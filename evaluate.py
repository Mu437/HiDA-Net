import os, yaml, time, argparse, random, logging, datetime, numpy as np
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from clip import clip
from model.HiDA import HiDANet 
from utils.network_fbcnn import FBCNN as FBCNN
from utils.dataset_source import dataset_info
from utils.tile_dataset_srm import TiledRFDataset,  collate_img_list_to_tensor_test
from utils.train_test_tools_ddp_float_tile import evaluate_ddp

import shutil
'''
export CUDA_VISIBLE_DEVICES=4 \
torchrun --standalone --nproc-per-node=1 evaluate.py --config configs/test.yaml --weights /path/to/pth'''

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
    log_file = os.path.join(cfg['save_dir'], "test.log")

    logging.basicConfig(filename=log_file, filemode="w",
                        level=logging.INFO, format=fmt, datefmt=datefmt)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)

    logging.info("===== Configuration =====")
    for k,v in cfg.items():
        logging.info(f"{k}: {v}")
    logging.info("=========================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (e.g. best.pth)")
    args_cmd = parser.parse_args()

    with open(args_cmd.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_path = args_cmd.weights
    test_resize = None
    test_webp = None
    test_jpeg = None
    test_blur = None

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bb_safe = cfg['backbone'].replace("/","-").replace("@","-")
    cfg['save_dir'] = os.path.join("evaluate_logs",
                                   f"{now}_{bb_safe}_{cfg['test_set']}")
    if test_resize is not None:
        cfg['save_dir'] += "_resize"+str(test_resize)
        print('resize', test_resize)
    if test_webp is not None:
        cfg['save_dir'] += "_webp"+str(test_webp)
        print('webp', test_webp)
    if test_jpeg is not None:
        cfg['save_dir'] += "_jpeg"+str(test_jpeg)
        print('jpeg', test_jpeg)
    if test_blur is not None:
        cfg['save_dir'] += "_blur"+str(test_blur)
        print('blur', test_blur)
    
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
                        )
    model.core.load_state_dict(torch.load(model_path))
    model = model.cuda()
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

    # train loader
    clip_res = clip_model.visual.input_resolution

    # test loader
    test_loaders = {}
    test_reals, test_fakes = dataset_info[cfg['test_set']]
    each_num = cfg['test_set_size'] // len(test_reals)
    for real, fake in zip(test_reals, test_fakes):
        name = real[0]
        if cfg['test_set'] == 'DRCT_test':
            name = fake[0]
        test_ds = TiledRFDataset(
            real, fake, tile_size=clip_res,
            jpeg_augment = test_jpeg, jpeg_p=1,
            webp_augment=test_webp, webp_p=1,
            resize_augment= test_resize, resize_p=1,
            blur_augment=test_blur, blur_p=1,
            set_size=each_num, shuffle_seed=cfg['seed'],
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
    
    logging.info(f"{model_path}")
    # evaluate
    all_metrics = {}
    for name, loader in test_loaders.items():
        print(f"Begin {name} Eval, Total={len(loader)}")
        metrics = evaluate_ddp(model, loader, rank, world_size)
        if rank == 0:
            if metrics is not None:
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

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
