import torch, torch.nn as nn, torch.distributed as dist
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, average_precision_score
import numpy as np
from PIL import Image
import random

tile_table = [4, 8, 2, 12, 4, 3, 6, 4, 16, 6, 4, 1, 6, 8, 10, 6]

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

def _center_crop_if_large(pil_img, crop=512):
    w, h = pil_img.size
    if w > crop or h > crop:
        l = max(0, (w - crop)//2); t = max(0, (h - crop)//2)
        return pil_img.crop((l,t,l+crop,t+crop))
    return pil_img

def center_crop_and_resize(pil_img, target_size=256):
    width, height = pil_img.size
    if width > target_size or height > target_size:
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        cropped_img = pil_img.crop((left, top, right, bottom))
        return cropped_img
    elif width < target_size or height < target_size:
        # Resize 
        return pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    else:
        return pil_img

fbcnn_preprocess = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_and_resize(img, target_size=256)), 
        transforms.ToTensor(),
    ])

@torch.no_grad()
def _fbcnn_qf(fbcnn, pil_list, device):
    imgs_tensor_list = [
        fbcnn_preprocess(p) for p in pil_list
    ]
    imgs_batch = torch.stack(imgs_tensor_list)
    qf_batch = fbcnn.predict_qf(imgs_batch.to(device, non_blocking=True))
    return qf_batch

def train_one_epoch_ddp(model, fbcnn, loader, optimizer,
                        epoch:int, rank:int, world_size):
    model.train()
    device = torch.device("cuda", rank)
    loss_cls_fn  = nn.CrossEntropyLoss()
    loss_jpeg_fn = nn.MSELoss()
    loss_patch_fn= nn.BCEWithLogitsLoss()

    tot = torch.zeros(6, device=device)  # loss,cls,jpeg,patch,correct,count
    bar = tqdm(loader, desc=f"Train EP {epoch}", disable=(rank!=0))

    pos = 0
    for imgs, raws, labels, patch_labels in bar:
        pos += 1
        jpeg_qf = _fbcnn_qf(fbcnn, raws, device)          # [B,1]

        # now_tiles = tile_table[random.randint(0, len(tile_table)-1)]
        now_tiles = tile_table[pos%len(tile_table)]

        imgs   = imgs.cuda(non_blocking=True)  # [Batch, 1+crop, c, w, h]
        labels = labels.cuda(non_blocking=True)
        patch_labels = patch_labels.cuda(non_blocking=True)
        
        # print(imgs.shape, patch_labels.shape)
        imgs = imgs[:, :1+now_tiles]
        patch_labels = patch_labels[:, :1+now_tiles]
        # print(imgs.shape, patch_labels.shape)

        logits, patch_logits, jpeg_pred = model(imgs)
        loss_cls   = loss_cls_fn(logits, labels)
        loss_jpeg  = loss_jpeg_fn(jpeg_pred, jpeg_qf)
        loss_patch = loss_patch_fn(patch_logits.squeeze(-1), patch_labels)

        loss = loss_cls + loss_jpeg + loss_patch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bsz = labels.size(0)
            correct = (logits.argmax(1) == labels).sum()
            tot += torch.tensor([loss.item()*bsz,
                                 loss_cls.item()*bsz,
                                 loss_jpeg.item()*bsz,
                                 loss_patch.item()*bsz,
                                 correct, bsz], device=device)

    # --- all_reduce ---
    if world_size > 1:
        dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    loss, cls, jpeg, patch, correct, count = tot.cpu().numpy()
    stats = {
        "loss":  loss   / count,
        "cls":   cls    / count,
        "jpeg":  jpeg   / count,
        "patch": patch  / count,
        "acc":   correct/ count,
    }
    return stats

@torch.no_grad()
def evaluate_ddp(model, loader, rank, world_size):
    device = torch.device("cuda", rank)
    model.eval()

    y_true_local, y_score_local = [], []
    for imgs, _, labels, _ in tqdm(loader, desc="Eval", disable=(rank!=0)):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)[0]
        prob   = torch.softmax(logits, 1)[:, 1]          # P(fake)

        y_true_local.append(labels.cpu().numpy())
        y_score_local.append(prob.cpu().numpy())

    y_true_local  = np.concatenate(y_true_local)
    y_score_local = np.concatenate(y_score_local)

    if world_size > 1:
        gather_true  = [None] * world_size
        gather_score = [None] * world_size
        dist.all_gather_object(gather_true,  y_true_local)
        dist.all_gather_object(gather_score, y_score_local)

        if rank != 0:
            return None

        y_true  = np.concatenate(gather_true)
        y_score = np.concatenate(gather_score)
    else:                    # single card
        y_true  = y_true_local
        y_score = y_score_local

    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    ap_score  = average_precision_score(y_true, y_score)

    accs      = [accuracy_score(y_true, y_score >= t) for t in thr]
    best_idx  = int(np.argmax(accs))
    best_thr  = thr[best_idx]
    best_acc  = accs[best_idx]

    
    acc_05    = accuracy_score(y_true, y_score >= 0.5)
    y_pred = (y_score >= 0.5).astype(int)
    acc_real = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
    acc_fake = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])

    return {
        "acc": acc_05, "acc_real": acc_real, "acc_fake": acc_fake,
        "auc": auc_score, "ap": ap_score,
        "best_thr": best_thr, "best_acc": best_acc,
    }