#!/usr/bin/env python3
"""
Batch-by-batch validation for EDSR checkpoints (BasicSR variant with extra args):
- Tracks dataset-mean L1 (RGB), L1 (Y), PSNR (Y), SSIM (Y) for each saved model
- Uses Y-channel (YCbCr, BT.601 "studio range") with standard HR crop (crop = scale)
- Minimal BasicSR dependency: only imports the EDSR arch
- Reproducible (fixed seeds, deterministic)
- TensorBoard logging (tb_logger): metrics are logged at global_step = parsed iteration from ckpt filename
- Assumes all LR images in the set are the same size so batching works
"""

import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # must be set before torch initializes CUDA
import re
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ====================== BasicSR arch import ======================
try:
    from basicsr.archs.edsr_arch import EDSR
except Exception as e:
    raise ImportError(
        "Failed to import EDSR from basicsr. Please `pip install basicsr` and make sure it matches your checkpoints."
    ) from e


# ====================== Reproducibility ======================
def set_repro(seed: int = 2023):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def worker_init_fn(worker_id):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)


# ====================== I/O helpers ======================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str) -> List[str]:
    files = [f for f in glob(os.path.join(folder, "*")) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """PIL (RGB) -> float tensor in [0,1], shape CxHxW."""
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def extract_iter_from_name(path: str) -> int:
    nums = [int(x) for x in re.findall(r"(\d+)", os.path.basename(path))]
    return max(nums) if nums else 0


# ====================== Model helpers ======================
def build_edsr(args) -> nn.Module:
    """
    Build EDSR with the signature you provided:
    EDSR(num_in_ch=3, num_out_ch=3, num_feat=192, num_block=6, upscale=4,
         num_heads=6, hw=8, ww=8, img_range=255., rgb_mean=[...])
    """
    return EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=args.num_feat,
        num_block=args.num_block,
        upscale=args.scale,
        num_heads=args.num_heads,
        hw=args.hw,
        ww=args.ww,
        img_range=args.img_range,
        rgb_mean=args.rgb_mean,
    )


def strip_key_prefix(state_dict, prefixes=("module.", "net_g.", "model.", "generator.")):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    candidate = None
    for key in ["params_ema", "state_dict", "network_g", "model", "net_g", "params"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            candidate = ckpt[key]
            break
    if candidate is None and isinstance(ckpt, dict):
        candidate = ckpt
    if candidate is None or not isinstance(candidate, dict):
        raise RuntimeError(f"Could not find a state_dict in: {ckpt_path}")

    candidate = strip_key_prefix(candidate)
    missing, unexpected = model.load_state_dict(candidate, strict=False)
    if missing or unexpected:
        print(f"[Warn] Non-strict load for '{os.path.basename(ckpt_path)}'")
        if missing:
            print("  Missing:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  Unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")


# ====================== Dataset ======================
class PairDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str):
        self.lr_paths = list_images(lr_dir)
        if not self.lr_paths:
            raise FileNotFoundError(f"No LR images found in {lr_dir}")

        hr_map = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(hr_dir)}
        pairs = []
        for lp in self.lr_paths:
            name = os.path.splitext(os.path.basename(lp))[0]
            if name not in hr_map:
                raise FileNotFoundError(f"Missing HR image for LR '{name}' in {hr_dir}")
            pairs.append((lp, hr_map[name], name))
        self.pairs = pairs

        # Enforce identical LR size for batching
        sizes = []
        for lp, _, _ in pairs:
            with Image.open(lp) as im:
                sizes.append(im.size)  # (W,H)
        # if len(set(sizes)) != 1:
        #     raise ValueError(f"All LR images must be identical size for batching, got: {set(sizes)}")
        self.lr_size = sizes[0]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lp, hp, name = self.pairs[idx]
        with Image.open(lp) as li:
            lr = pil_to_tensor01(li.convert("RGB"))
        with Image.open(hp) as hi:
            hr = pil_to_tensor01(hi.convert("RGB"))
        return lr, hr, name


# ====================== Metrics (Y-channel, standard crop) ======================
def rgb01_to_y_255(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB in [0,1] to Y channel in [0,255] using ITU-R BT.601 studio-range:
      Y = 16 + 65.481*R + 128.553*G + 24.966*B  (with R,G,B in [0,1], Y in [0,255])
    Input:  N x 3 x H x W  (or 3 x H x W with batch dim added by caller)
    Output: N x 1 x H x W  in [0,255]
    """
    assert rgb.dim() == 4 and rgb.size(1) == 3, "Expected shape N x 3 x H x W"
    r = rgb[:, 0:1, ...]
    g = rgb[:, 1:2, ...]
    b = rgb[:, 2:3, ...]
    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    return y


def crop_border(x: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


class L1LossNoReduce(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, x, y):
        return self.l1(x, y)  # N x C x H x W


def psnr_from_mse(mse: torch.Tensor, max_val: float) -> torch.Tensor:
    return 10.0 * torch.log10((max_val ** 2) / (mse + 1e-12))


def gaussian_window(window_size: int, sigma: float, channel: int, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)  # 1 x W
    kernel_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)   # 1 x 1 x H x W
    kernel = kernel_2d.repeat(channel, 1, 1, 1)         # C x 1 x H x W
    return kernel


def ssim_on_y01(y1_01: torch.Tensor, y2_01: torch.Tensor, window_size=11, sigma=1.5, eps=1e-12) -> torch.Tensor:
    """
    SSIM on Y channel in [0,1]; returns per-image values (N,).
    """
    N, C, H, W = y1_01.shape
    assert C == 1
    device = y1_01.device
    window = gaussian_window(window_size, sigma, 1, device)
    padding = window_size // 2

    mu1 = torch.conv2d(y1_01, window, padding=padding, groups=1)
    mu2 = torch.conv2d(y2_01, window, padding=padding, groups=1)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.conv2d(y1_01 * y1_01, window, padding=padding, groups=1) - mu1_sq
    sigma2_sq = torch.conv2d(y2_01 * y2_01, window, padding=padding, groups=1) - mu2_sq
    sigma12   = torch.conv2d(y1_01 * y2_01, window, padding=padding, groups=1) - mu1_mu2

    C1 = (0.01 ** 2)  # since data_range=1
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
    return ssim_map.mean(dim=(1, 2, 3))


# ====================== Validation ======================
@torch.no_grad()
def validate_one_ckpt(
    ckpt_path: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale: int,
    use_amp: bool,
    feed_img255: bool,
    img_range: float,
) -> Tuple[float, float, float, float]:
    """
    Returns dataset means: (L1_RGB, L1_Y, PSNR_Y, SSIM_Y)
    """
    model.eval()
    load_checkpoint_into_model(model, ckpt_path)
    model.to(device)

    l1_metric = L1LossNoReduce()

    l1_sum = 0.0
    l1y_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    n_img = 0

    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast

    for lr, hr, _names in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        lr_in = lr * img_range if feed_img255 else lr
        with amp_ctx():
            sr = model(lr_in)
        sr = sr / img_range if feed_img255 else sr
        sr = torch.clamp(sr, 0.0, 1.0)

        # Standard HR crop (border = scale)
        sr_c = crop_border(sr, scale)
        hr_c = crop_border(hr, scale)

        # L1 on RGB (after crop)
        l1_per = l1_metric(sr_c, hr_c).mean(dim=(1, 2, 3))  # per-image

        # PSNR / SSIM on Y (+ L1 on Y)
        y_sr_255 = rgb01_to_y_255(sr_c)
        y_hr_255 = rgb01_to_y_255(hr_c)

        mse_y = ((y_sr_255 - y_hr_255) ** 2).mean(dim=(1, 2, 3))
        psnr_y = psnr_from_mse(mse_y, max_val=255.0)

        y_sr_01 = torch.clamp(y_sr_255 / 255.0, 0.0, 1.0)
        y_hr_01 = torch.clamp(y_hr_255 / 255.0, 0.0, 1.0)
        ssim_y  = ssim_on_y01(y_sr_01, y_hr_01)

        # L1 on Y (in [0,1])
        l1_y_per = l1_metric(y_sr_01, y_hr_01).mean(dim=(1, 2, 3))

        bs = lr.size(0)
        l1_sum   += l1_per.sum().item()
        l1y_sum  += l1_y_per.sum().item()
        psnr_sum += psnr_y.sum().item()
        ssim_sum += ssim_y.sum().item()
        n_img    += bs

    return l1_sum / n_img, l1y_sum / n_img, psnr_sum / n_img, ssim_sum / n_img


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(
        description="Validate EDSR checkpoints: L1 (RGB), L1 (Y), PSNR/SSIM (Y, BT.601), standard crop = scale."
    )
    # Data / IO
    parser.add_argument("--lr_dir",      type=str, default="datasets/Valid/DIV2K/LRx4_randomcrop",     help="Folder with LR images (all same size).")
    parser.add_argument("--hr_dir",      type=str, default="datasets/Valid/DIV2K/HR_randomcrop",     help="Folder with HR images.")
    parser.add_argument("--models_dir",  type=str, default="experiments/thesis_v100/models", help="Folder with checkpoints (.pth).")
    parser.add_argument("--log_dir",     type=str, default="runs/valid/conv", help="TensorBoard log directory.")
    # Eval
    parser.add_argument("--scale",       type=int, default=4,        help="Upscale factor (2/3/4). Used for model and HR crop.")
    parser.add_argument("--batch_size",  type=int, default=1,        help="Batch size for LR images.")
    parser.add_argument("--num_workers", type=int, default=4,        help="DataLoader workers.")
    parser.add_argument("--device",      type=str, default="cuda",   help="'cuda' or 'cpu'.")
    parser.add_argument("--amp",         action="store_true",        help="Use autocast mixed precision for inference.")
    parser.add_argument("--seed",        type=int, default=10,     help="Deterministic seed.")
    parser.add_argument("--feed_img255", action="store_true",
                        help="Multiply LR by img_range before forward and divide SR after; enable if your training did this outside the model.")
    # EDSR hyper-params (match your trained checkpoints)
    parser.add_argument("--num_feat",  type=int,   default=64,   help="EDSR feature channels.")
    parser.add_argument("--num_block", type=int,   default=4,     help="EDSR residual blocks.")
    parser.add_argument("--num_heads", type=int,   default=4,     help="Heads for your EDSR variant (if used).")
    parser.add_argument("--hw",        type=int,   default=8,     help="Head/window height.")
    parser.add_argument("--ww",        type=int,   default=8,     help="Head/window width.")
    parser.add_argument("--img_range", type=float, default=255.0, help="Model image range (commonly 255.0).")
    parser.add_argument("--rgb_mean",  type=float, nargs=3, default=[0.4488, 0.4371, 0.4040],
                        help="RGB mean (0-1 range).")

    args = parser.parse_args()
    set_repro(args.seed)

    # Dataset & loader
    dataset = PairDataset(args.lr_dir, args.hr_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # Device & model shell (weights reloaded per ckpt)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_edsr(args)

    # Checkpoints
    ckpts = [p for p in glob(os.path.join(args.models_dir, "*.pth"))]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints (*.pth) found in {args.models_dir}")
    ckpts.sort(key=lambda p: (extract_iter_from_name(p), p))

    # TensorBoard
    tb_logger = SummaryWriter(log_dir=args.log_dir)

    print(f"Found {len(ckpts)} checkpoints. Starting validation...")
    for ck in ckpts:
        it = extract_iter_from_name(ck)
        print(f"Evaluating: {os.path.basename(ck)} (iter={it})")
        l1_rgb, l1_y, psnr_y, ssim_y = validate_one_ckpt(
            ckpt_path=ck,
            model=model,
            loader=loader,
            device=device,
            scale=args.scale,
            use_amp=args.amp,
            feed_img255=args.feed_img255,
            img_range=args.img_range,
        )

        tb_logger.add_scalar("valid/L1_RGB", l1_rgb, global_step=it)
        tb_logger.add_scalar("valid/L1_Y",   l1_y,   global_step=it)
        tb_logger.add_scalar("valid/PSNR_Y", psnr_y, global_step=it)
        tb_logger.add_scalar("valid/SSIM_Y", ssim_y, global_step=it)

        print(f"  L1_RGB: {l1_rgb:.6f} | L1_Y: {l1_y:.6f} | PSNR_Y: {psnr_y:.4f} dB | SSIM_Y: {ssim_y:.6f}")


    tb_logger.close()
    print("Done.")


if __name__ == "__main__":
    main()
