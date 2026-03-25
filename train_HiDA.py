"""
export CUDA_VISIBLE_DEVICES=1,2 \
torchrun --standalone --nproc-per-node=1 train_HiDA.py --config configs/train.yaml
"""
import os, yaml, time, argparse, random, logging, datetime, numpy as np
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from clip import clip
from model.HiDA import HiDANet
from utils.network_fbcnn import FBCNN as FBCNN
from utils.dataset_source import dataset_info
from utils.tile_dataset_srm           import TiledRFDataset,  collate_img_list_to_tensor_test
from utils.tile_dataset_srm_paired_patch_enhance import (
    TiledRFDataset_paired, collate_img_list_to_tensor_paried_train)

from utils.train_test_tools_ddp_float_tile import (
    train_one_epoch_ddp, evaluate_ddp)

import shutil

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(cfg, rank):
    if rank != 0:
        logging.basicConfig(level=logging.ERROR)
        return

    os.makedirs(cfg['save_dir'], exist_ok=True)
    fmt = "%(asctime)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_file = os.path.join(cfg['save_dir'], "train.log")

    logging.basicConfig(filename=log_file, filemode="w",
                        level=logging.INFO, format=fmt, datefmt=datefmt)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)

    logging.info("===== Configuration =====")
    for k,v in cfg.items():
        logging.info(f"{k}: {v}")
    logging.info("=========================")

def get_train_loader(cfg, resolution, epoch, world_size, rank):
    train_reals, train_fakes = dataset_info[cfg['train_set']]
    train_ds = TiledRFDataset_paired(
        train_reals[0], train_fakes[0],
        tile_size=resolution,
        set_size=cfg['train_set_size'],
        shuffle_seed=cfg['seed']+epoch,
        jpeg_augment=cfg['jpeg_augment'], jpeg_p=cfg['jpeg_p'],
        resize_augment=cfg['resize_augment'], resize_p=cfg['resize_p'],
        blur_augment=cfg['blur_augment'], blur_p=cfg['blur_p'],
        is_train=True,
        default_train_tiles=cfg['default_train_tiles'],
        replace_p=cfg['replace_p'],
        replace_ratio=cfg['replace_ratio'],
        replace_grid=cfg['replace_grid'])
    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    train_loader  = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg['batch_size']//2,
        sampler=train_sampler,
        num_workers=cfg['num_workers'],
        collate_fn=collate_img_list_to_tensor_paried_train,
        pin_memory=True,
        drop_last=True)
    return train_ds, train_loader, train_sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file")
    args_cmd = parser.parse_args()

    with open(args_cmd.config, "r") as f:
        cfg = yaml.safe_load(f)


    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bb_safe = cfg['backbone'].replace("/","-").replace("@","-")
    cfg['save_dir'] = os.path.join("train_logs",
                                   f"{now}_{bb_safe}_{cfg['test_set']}_float")
    os.makedirs(cfg['save_dir'], exist_ok=True)
    if int(os.environ["RANK"]) == 0:
        shutil.copy(args_cmd.config, os.path.join(cfg['save_dir'], "config.yaml"))

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    setup_logger(cfg, rank)
    set_seed(cfg['seed'] + rank)                   

    clip_model, _ = clip.load(cfg['backbone'], device="cpu") 
    model = HiDANet(clip_model.visual,
                         cfg['transformer_heads'], 2,
                         cfg['transformer_layers'],
                         dropout=cfg['dropout'],
                         tile_heads=cfg['tile_heads'],
                         tile_layers=cfg['tile_layers']
                        ).cuda()
    model = torch.compile(model)
    if world_size > 1:
        model.core = DDP(
            model.core,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    fbcnn = FBCNN(in_nc=3, out_nc=3, nc=[64,128,256,512], nb=4,
                  act_mode='R').cuda()
    fbcnn_ckpt = cfg['fbcnn_ckpt']
    fbcnn.load_state_dict(torch.load(fbcnn_ckpt, map_location="cpu"), strict=True)
    fbcnn.eval()
    for param in fbcnn.parameters():
        param.requires_grad = False
    
    eff_lr = cfg['learning_rate'] * cfg['batch_size'] * world_size / 256
    if rank == 0:
        logging.info(f"Effective lr: {eff_lr} Effective batch size: {world_size * cfg['batch_size']}")
    if world_size > 1:
        optimizer = torch.optim.AdamW(model.core.module.parameters(),
                                  lr=eff_lr, weight_decay=cfg['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.core.parameters(),
                                  lr=eff_lr, weight_decay=cfg['weight_decay'])

    
    clip_res = clip_model.visual.input_resolution
   
    test_loaders = {}
    test_reals, test_fakes = dataset_info[cfg['test_set']]
    each_num = cfg['test_set_size'] // len(test_reals)
    for real, fake in zip(test_reals, test_fakes):
        name = real[0]
        if cfg['test_set'] == 'DRCT_test':
            name = fake[0]
        test_ds = TiledRFDataset(
            real, fake, tile_size=clip_res,
            set_size=each_num, shuffle_seed=cfg['seed'],
            balance=False, 
            is_train=False)
        test_sampler = DistributedSampler(test_ds, world_size, rank, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(
            test_ds,
            batch_size=cfg['test_batch_size'],
            sampler=test_sampler,
            num_workers=4,
            collate_fn=collate_img_list_to_tensor_test,
            pin_memory=True)
        test_loaders[name] = test_loader

    best_acc = 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        train_ds, train_loader, train_sampler = get_train_loader(cfg, clip_res, epoch, world_size, rank)
        if rank == 0 and epoch == 1:
            logging.info(f"Loaded dataset | train={len(train_ds)*2}  test≈{each_num*len(test_loaders)}") 

        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch_ddp(model, fbcnn, train_loader,
                                          optimizer, epoch, rank, world_size)
        if rank == 0:
            logging.info(f"[Train EP {epoch}] " + " ".join(f"{k}:{v:.4f}" for k,v in train_stats.items()))

        # evaluate
        logging.info(f"[Evaluate EP {epoch}]")
        all_metrics = {}
        for name, loader in test_loaders.items():
            metrics = evaluate_ddp(model, loader, rank, world_size)
            if rank == 0:
                all_metrics[name] = metrics
                logging.info(f"[Eval-{name}] " + " ".join(f"{k}:{v:.4f}" for k,v in metrics.items()))

        # save model
        if rank == 0:
            metric_keys = list(next(iter(all_metrics.values())).keys())
            avg_metrics = {
                k: np.mean([m[k] for m in all_metrics.values()])
                for k in metric_keys
            }

            logging.info("=== Average over all test sets ===")
            for k, v in avg_metrics.items():
                logging.info(f"{k}: {v:.4f}")
            logging.info("="*40)

            cur_acc = avg_metrics['acc']
            if world_size > 1:
                save_model = model.core.module
            else:
                save_model = model.core
            torch.save(save_model.state_dict(),
                    os.path.join(cfg['save_dir'], f"epoch_{epoch:02d}.pth"))
            if cur_acc > best_acc:
                best_acc = cur_acc
                torch.save(save_model.state_dict(),
                        os.path.join(cfg['save_dir'], "best.pth"))
                logging.info(f"New best ACC={best_acc:.4f} @ epoch {epoch}")

    dist.destroy_process_group()
    if rank == 0:
        logging.info("Training finished.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
