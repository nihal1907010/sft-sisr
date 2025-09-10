#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-wise validation for x4 SR with crop evaluation.
- Minimal dependencies: PyTorch, torchvision (optional), PIL, numpy, scikit-image, TensorBoard
- No BasicSR utils; only imports EDSR architecture to stay checkpoint-compatible.
"""

import os
import re
import csv
import glob
import math
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# --- (optional) tqdm progress bar
try:
    from tqdm import tqdm
except ImportError:  # fallback if tqdm not installed
    tqdm = lambda x, **k: x

# ====== MODEL IMPORT ==========================================================
# Keep this one-liner so you can load your trained EDSR checkpoints.
# If you want ZERO basicsr, I can inline a pure-PyTorch EDSR here instead.
from basicsr.archs.edsr_arch import EDSR as EDSRNet
# ============================================================================


# ========= Utilities ==========================================================
def lr_to_gt_name(lr_name: str, scale: int) -> str:
    """Map '0801x4.png' -> '0801.png' (strip trailing 'x{scale}')."""
    return re.sub(rf'x{scale}(?=\.[^.]+$)', '', lr_name)

def list_pairs(lr_dir: str, gt_dir: str, scale: int) -> List[Tuple[str, str]]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    lr_files = sorted([f for f in os.listdir(lr_dir) if os.path.splitext(f)[1].lower() in exts])
    pairs, missing = [], []
    for lf in lr_files:
        gf = lr_to_gt_name(lf, scale)
        lr_fp = os.path.join(lr_dir, lf)
        gt_fp = os.path.join(gt_dir, gf)
        if os.path.exists(gt_fp):
            pairs.append((lr_fp, gt_fp))
        else:
            missing.append((lf, gf))
    print(f"[Pairs] Found {len(pairs)} LR/GT pairs. Missing GT for {len(missing)} LR files.")
    if missing:
        print("  Examples of missing mappings:", missing[:5])
    return pairs

def pil_to_numpy_rgb01(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 numpy HxWxC in [0,1]."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

def crop_border_np(arr: np.ndarray, b: int) -> np.ndarray:
    if b <= 0:
        return arr
    if arr.ndim == 3:
        return arr[b:-b, b:-b, :]
    return arr[b:-b, b:-b]

def rgb_to_y(arr01: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> Y (BT.601) [0,1], returns HxW."""
    return (0.299 * arr01[..., 0] + 0.587 * arr01[..., 1] + 0.114 * arr01[..., 2]).astype(np.float32)

def compute_metrics(gt: np.ndarray, sr: np.ndarray, crop: int, y_only: bool, data_range: float):
    gt_c = crop_border_np(gt, crop)
    sr_c = crop_border_np(sr, crop)
    if y_only:
        gt_eval = rgb_to_y(gt_c)
        sr_eval = rgb_to_y(sr_c)
        psnr_v = psnr_fn(gt_eval, sr_eval, data_range=data_range)
        ssim_v = ssim_fn(gt_eval, sr_eval, data_range=data_range)
        l1_v = float(np.mean(np.abs(gt_eval - sr_eval)))
    else:
        gt_eval = gt_c
        sr_eval = sr_c
        psnr_v = psnr_fn(gt_eval, sr_eval, data_range=data_range)
        ssim_v = ssim_fn(gt_eval, sr_eval, channel_axis=2, data_range=data_range)
        l1_v = float(np.mean(np.abs(gt_eval - sr_eval)))
    return float(psnr_v), float(ssim_v), float(l1_v)

def safe_load_state_dict(model: nn.Module, state):
    """Load common checkpoint formats (ema/params/state_dict/raw)."""
    if isinstance(state, dict):
        if "params_ema" in state:
            model.load_state_dict(state["params_ema"], strict=True)
            return "params_ema"
        if "params" in state:
            model.load_state_dict(state["params"], strict=True)
            return "params"
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
            return "state_dict"
    model.load_state_dict(state, strict=False)
    return "raw"

def sample_lr_boxes(w: int, h: int, crop: int, strategy: str, num: int, rng: np.random.Generator):
    """Return a list of (x, y, crop, crop) boxes in LR coords."""
    if w < crop or h < crop:
        return []
    boxes = []
    if strategy == "center":
        x = (w - crop) // 2
        y = (h - crop) // 2
        boxes.append((x, y, crop, crop))
    elif strategy == "random":
        for _ in range(num):
            x = int(rng.integers(0, w - crop + 1))
            y = int(rng.integers(0, h - crop + 1))
            boxes.append((x, y, crop, crop))
    else:
        raise ValueError(f"Unknown CROP_STRATEGY: {strategy}")
    return boxes
# ============================================================================


# =========== Dataset / DataLoader ============================================
class PairedLRHRDataset(Dataset):
    """
    Simple paired dataset:
        - LR files live under VAL_LR_DIR
        - GT files live under VAL_GT_DIR
        - File mapping uses 'namex{scale}.ext' -> 'name.ext'
    Yields per-item dict with numpy arrays [0,1] HxWxC for convenience.
    """
    def __init__(self, lr_dir: str, gt_dir: str, scale: int):
        super().__init__()
        self.scale = scale
        self.pairs = list_pairs(lr_dir, gt_dir, scale)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lr_fp, gt_fp = self.pairs[idx]
        lr_np = pil_to_numpy_rgb01(Image.open(lr_fp))
        gt_np = pil_to_numpy_rgb01(Image.open(gt_fp))
        return {
            "lr": lr_np,      # H_lr x W_lr x 3, [0,1]
            "gt": gt_np,      # H_hr x W_hr x 3, [0,1]
            "lr_path": lr_fp,
            "gt_path": gt_fp,
        }


# ============ Core evaluation ================================================
def run_crops_in_batches(model: nn.Module,
                         device: torch.device,
                         lr_crops_np: List[np.ndarray],
                         batch_size: int) -> List[np.ndarray]:
    """Run list of LR crops (HxWx3 in [0,1]) through model in batches; return list of SR crops (HxWx3 in [0,1])."""
    sr_list = []
    with torch.no_grad():
        for i in range(0, len(lr_crops_np), batch_size):
            batch_np = lr_crops_np[i:i+batch_size]                     # list of HxWx3
            batch_t = torch.stack(
                [torch.from_numpy(x).permute(2, 0, 1) for x in batch_np], dim=0
            ).to(device)                                               # B x 3 x H x W, [0,1]
            sr_t = model(batch_t).clamp_(0, 1).cpu()                   # B x 3 x (H*S) x (W*S)
            for b in range(sr_t.size(0)):
                sr_np = sr_t[b].permute(1, 2, 0).numpy()               # HxWx3
                sr_list.append(sr_np)
    return sr_list


# ============ Main ============================================================
def main():
    parser = argparse.ArgumentParser(description="Batch-wise SR validation (minimal PyTorch)")
    # Data / IO
    parser.add_argument("--val_lr_dir", type=str, default="datasets/Valid/DIV2K/LRx4")
    parser.add_argument("--val_gt_dir", type=str, default="datasets/Valid/DIV2K/HR")
    parser.add_argument("--ckpt_dir",   type=str, default="experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models")
    parser.add_argument("--ckpt_glob",  type=str, default="*.pth")
    parser.add_argument("--csv_out",    type=str, default="val_metrics_crops.csv")
    parser.add_argument("--tb_logdir",  type=str, default="tb_logs/validation")

    # Eval protocol
    parser.add_argument("--scale",      type=int, default=4)
    parser.add_argument("--test_y",     action="store_true", help="compute PSNR/SSIM on Y channel")
    parser.add_argument("--data_range", type=float, default=1.0)
    parser.add_argument("--crop_border", type=int, default=None, help="border crop for metrics; default=scale")

    # Crops
    parser.add_argument("--lr_crop", type=int, default=64, help="LR crop size")
    parser.add_argument("--crop_strategy", type=str, default="center", choices=["center", "random"])
    parser.add_argument("--num_crops_per_image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1907010)

    # Batching / runtime
    parser.add_argument("--batch_size", type=int, default=64, help="batch size over LR crops")
    parser.add_argument("--num_workers", type=int, default=0)

    # Model hyperparams (must match training)
    parser.add_argument("--num_in_ch", type=int, default=3)
    parser.add_argument("--num_out_ch", type=int, default=3)
    parser.add_argument("--num_feat", type=int, default=192)
    parser.add_argument("--num_block", type=int, default=6)
    parser.add_argument("--upscale",  type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--hw",       type=int, default=8)
    parser.add_argument("--ww",       type=int, default=8)
    parser.add_argument("--img_range", type=float, default=255.0)
    parser.add_argument("--rgb_mean",  nargs=3, type=float, default=[0.4488, 0.4371, 0.4040])

    args = parser.parse_args()

    SCALE = args.scale
    CROP_BORDER = args.crop_border if args.crop_border is not None else SCALE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Dataset & DataLoader (batch_size=1 full image; we batch *crops* ourselves)
    dataset = PairedLRHRDataset(args.val_lr_dir, args.val_gt_dir, SCALE)
    if len(dataset) == 0:
        print("No LR/GT pairs found. Check directories/mapping.")
        return
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Model
    net = EDSRNet(
        num_in_ch=args.num_in_ch,
        num_out_ch=args.num_out_ch,
        num_feat=args.num_feat,
        num_block=args.num_block,
        upscale=args.upscale,
        num_heads=args.num_heads,
        hw=args.hw,
        ww=args.ww,
        img_range=args.img_range,
        rgb_mean=args.rgb_mean,
    ).to(device).eval()

    # Checkpoints
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, args.ckpt_glob)))
    if not ckpt_paths:
        print(f"No checkpoints found in: {args.ckpt_dir} (pattern: {args.ckpt_glob})")
        return
    print(f"[Checkpoints] Found {len(ckpt_paths)} files")

    # Logging
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    os.makedirs(args.tb_logdir, exist_ok=True)
    tb = SummaryWriter(args.tb_logdir)

    rng = np.random.default_rng(args.seed)
    results = []
    global_step = 0

    with torch.no_grad():
        for ckpt in tqdm(ckpt_paths, desc="Validating checkpoints"):
            # Load weights
            try:
                state = torch.load(ckpt, map_location=device)
            except Exception as e:
                print(f"[Warn] Failed to load {ckpt}: {e}")
                continue
            key_used = safe_load_state_dict(net, state)
            print(f"[Eval] {os.path.basename(ckpt)} (loaded: {key_used})")

            psnr_imgs, ssim_imgs, l1_imgs = [], [], []

            for batch in loader:
                lr_np = batch["lr"][0].numpy()  # H_lr x W_lr x 3 in [0,1]
                gt_np = batch["gt"][0].numpy()  # H_hr x W_hr x 3 in [0,1]
                lr_path = batch["lr_path"][0]
                gt_path = batch["gt_path"][0]

                H_lr, W_lr = lr_np.shape[:2]
                boxes = sample_lr_boxes(W_lr, H_lr, args.lr_crop, args.crop_strategy, args.num_crops_per_image, rng)
                if not boxes:
                    continue

                # collect LR/GT crops (numpy 0..1)
                lr_crops_np, gt_crops_np = [], []
                for (x, y, cw, ch) in boxes:
                    lr_crop = lr_np[y:y+ch, x:x+cw, :]
                    x_hr, y_hr = x * SCALE, y * SCALE
                    cw_hr, ch_hr = cw * SCALE, ch * SCALE
                    gt_crop = gt_np[y_hr:y_hr+ch_hr, x_hr:x_hr+cw_hr, :]
                    lr_crops_np.append(lr_crop.astype(np.float32))
                    gt_crops_np.append(gt_crop.astype(np.float32))

                # forward in *batches* of crops
                sr_crops_np = run_crops_in_batches(net, device, lr_crops_np, args.batch_size)

                # metrics per crop -> per image
                psnr_crops, ssim_crops, l1_crops = [], [], []
                for gt_crop, sr_crop in zip(gt_crops_np, sr_crops_np):
                    p, s, l = compute_metrics(
                        gt_crop, sr_crop, crop=CROP_BORDER,
                        y_only=args.test_y, data_range=args.data_range
                    )
                    psnr_crops.append(p); ssim_crops.append(s); l1_crops.append(l)

                if psnr_crops:
                    psnr_img = float(np.mean(psnr_crops))
                    ssim_img = float(np.mean(ssim_crops))
                    l1_img   = float(np.mean(l1_crops))
                    psnr_imgs.append(psnr_img); ssim_imgs.append(ssim_img); l1_imgs.append(l1_img)

                    # TB per-image
                    tb.add_scalar("val_image/psnr", psnr_img, global_step)
                    tb.add_scalar("val_image/ssim", ssim_img, global_step)
                    tb.add_scalar("val_image/l1",   l1_img,   global_step)
                    tb.add_text("val_image/name", f"{os.path.basename(lr_path)}->{os.path.basename(gt_path)}", global_step)
                    global_step += 1

            if psnr_imgs:
                row = {
                    "checkpoint": os.path.basename(ckpt),
                    "psnr": float(np.mean(psnr_imgs)),
                    "ssim": float(np.mean(ssim_imgs)),
                    "l1": float(np.mean(l1_imgs)),
                }
                results.append(row)
                print(f"  -> PSNR: {row['psnr']:.4f}, SSIM: {row['ssim']:.5f}, L1: {row['l1']:.6f}")
                # TB per-checkpoint
                tb.add_scalar("val_ckpt/psnr", row["psnr"], global_step)
                tb.add_scalar("val_ckpt/ssim", row["ssim"], global_step)
                tb.add_scalar("val_ckpt/l1",   row["l1"],   global_step)
                tb.add_text("val_ckpt/name", row["checkpoint"], global_step)
            else:
                print("  -> No valid crops/images evaluated.")

    if not results:
        print("No results computed.")
        tb.close()
        return

    # Save CSV
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "psnr", "ssim", "l1"])
        w.writeheader()
        w.writerows(results)
    print(f"[CSV] Wrote: {args.csv_out}")

    # Best by PSNR
    best = max(results, key=lambda x: x["psnr"])
    print("Best by PSNR:", best)

    tb.flush()
    tb.close()


if __name__ == "__main__":
    main()
