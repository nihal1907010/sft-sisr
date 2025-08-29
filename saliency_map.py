#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saliency maps (input-gradient) for EDSR (BasicSR)

Outputs:
  - sr.png
  - saliency_gray_lr.png
  - saliency_gray_sr.png
  - saliency_color.png
  - saliency_overlay.png
  - index.txt

Requirements:
  pip install torch torchvision opencv-python numpy
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Your EDSR (uses your BasicBlock internally) ----
from basicsr.archs.edsr_arch import EDSR


# =========================
# ======= CONFIG ==========
# =========================
OUTDIR = "saliency_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Your exact paths ----
LR_PATH   = "datasets/Test/Set5/LRx4/baby.png"
HR_PATH   = "datasets/Test/Set5/HR/baby.png"  # used if MODE='loss'
CKPT_PATH = "experiments/v201_f64_b8_h8_hw8_b4a3_1em3_300k_5em1_75k150k225k_x4/models/net_g_latest.pth"

# ---- Your exact model params ----
MODEL_KW = dict(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=8,
    upscale=4,
    num_heads=8,
    hw=8,
    ww=8,
    img_range=255.,
    rgb_mean=[0.4488, 0.4371, 0.4040],
)

# Scalar target for backprop:
MODE = "loss"          # 'loss' | 'pixel' | 'patch'
PIXEL_XY = None        # SR coords (x,y) if MODE='pixel' (None=center)
PATCH_XYWH = None      # SR coords (x,y,w,h) if MODE='patch' (None=center patch)

# Visualization tuning (conservative to avoid washout)
P_LOW, P_HIGH = 5.0, 95.0   # percentile clip for contrast
GAMMA = 1.0                 # 1.0 neutral; <1 brightens, >1 darkens
CMAP_ID = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
ALPHA_OVERLAY = 0.40


# =========================
# ======= HELPERS =========
# =========================
def read_image(path: str, to_rgb=True) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = img[..., None]
    if to_rgb and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError("Expected HxWxC")
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img

def to_tensor(img: np.ndarray) -> torch.Tensor:
    # HWC [0,1] -> BCHW float
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().float()

def save_png(path: str, img01: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.clip(img01, 0, 1)
    a = (a * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, a)

def colorize_and_overlay(sr_rgb: np.ndarray, cam_01: np.ndarray, alpha=ALPHA_OVERLAY):
    """
    Robust: resizes cam_01 to SR size if needed, then returns (heat_rgb, overlay_rgb).
    """
    H, W = sr_rgb.shape[:2]
    if cam_01.shape[:2] != (H, W):
        cam_01 = cv2.resize(cam_01, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_u8 = (np.clip(cam_01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, CMAP_ID)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    sr_u8 = (np.clip(sr_rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    overlay = cv2.addWeighted(sr_u8, 1.0 - alpha, heat_rgb, alpha, 0.0)
    return heat_rgb, overlay

def postprocess_01(x: np.ndarray, p_low=P_LOW, p_high=P_HIGH, gamma=GAMMA) -> np.ndarray:
    x = x.astype(np.float32)
    m = np.isfinite(x)
    if not m.all():
        if m.any():
            med = float(np.median(x[m]))
            x[~m] = med
        else:
            return np.full_like(x, 0.5, dtype=np.float32)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        # fallback to max-norm
        x = x - np.nanmin(x)
        mx = np.nanmax(x)
        return np.full_like(x, 0.5, dtype=np.float32) if mx <= 1e-8 else np.clip(x / mx, 0, 1)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, gamma, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)

def disable_inplace(m: nn.Module):
    if isinstance(m, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU)):
        if hasattr(m, "inplace") and m.inplace:
            m.inplace = False

def load_state_flex(model: nn.Module, ckpt_path: str) -> None:
    print(f"[Info] Loading checkpoint: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["params_ema", "params", "state_dict", "net", "model"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    missing, unexpected = model.load_state_dict(obj, strict=False)
    print(f"[Info] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:    print("  missing (first 10):", missing[:10])
    if unexpected: print("  unexpected (first 10):", unexpected[:10])

def make_scalar_target(
    sr: torch.Tensor,
    hr: torch.Tensor | None,
    mode: str,
    pixel_xy: tuple[int, int] | None = None,
    patch_xywh: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    if mode == "loss":
        if hr is None:
            raise ValueError("MODE='loss' requires HR_PATH.")
        return F.l1_loss(sr, hr)
    B, C, H, W = sr.shape
    if mode == "pixel":
        if pixel_xy is None:
            y, x = H // 2, W // 2
        else:
            x = int(np.clip(pixel_xy[0], 0, W - 1))
            y = int(np.clip(pixel_xy[1], 0, H - 1))
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        return sr[..., y0:y1, x0:x1].mean()
    if mode == "patch":
        if patch_xywh is None:
            cy, cx = H // 2, W // 2
            h, w = min(32, H), min(32, W)
            y0, x0 = max(0, cy - h // 2), max(0, cx - w // 2)
        else:
            x0, y0, w, h = patch_xywh
            x0 = int(np.clip(x0, 0, W - 1))
            y0 = int(np.clip(y0, 0, H - 1))
            w = int(np.clip(w, 1, W - x0))
            h = int(np.clip(h, 1, H - y0))
        return sr[..., y0:y0 + h, x0:x0 + w].mean()
    raise ValueError(f"Unknown MODE: {mode}")

def compute_saliency(model: nn.Module, lr_t: torch.Tensor,
                     hr_t: torch.Tensor | None,
                     mode: str, pixel_xy=None, patch_xywh=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      saliency_01: (Hlr,Wlr) in [0,1]
      sr_rgb: (Hsr,Wsr,3) in [0,1]
    """
    # Make a grad-enabled clone of input
    x = lr_t.clone().detach().requires_grad_(True)

    # Forward
    sr = model(x)

    # Objective
    scalar = make_scalar_target(sr, hr_t, mode=mode, pixel_xy=pixel_xy, patch_xywh=patch_xywh)

    # Backprop
    model.zero_grad(set_to_none=True)
    scalar.backward()

    # Gradient wrt input
    grad = x.grad  # (1,3,Hlr,Wlr)
    if grad is None:
        raise RuntimeError("No gradient captured for input (check requires_grad).")

    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    sal = grad.abs().max(dim=1, keepdim=False)[0]  # (1,H,W) -> (H,W)
    sal = sal[0].detach().cpu().numpy()

    # Normalize (percentile-based)
    sal01 = postprocess_01(sal, p_low=P_LOW, p_high=P_HIGH, gamma=GAMMA)

    # SR for overlay (to HxWxC, [0,1])
    sr_np = sr.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    sr_np = np.clip(sr_np, 0.0, 1.0)

    return sal01, sr_np


# =========================
# ========= MAIN ==========
# =========================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load LR/HR
    lr_img = ensure_3ch(read_image(LR_PATH, to_rgb=True))
    lr_t = to_tensor(lr_img)

    hr_t = None
    if MODE == "loss":
        hr_img = ensure_3ch(read_image(HR_PATH, to_rgb=True))
        h_lr, w_lr = lr_img.shape[:2]
        h_hr, w_hr = hr_img.shape[:2]
        scale = MODEL_KW.get("upscale", 4)
        if (h_lr * scale != h_hr) or (w_lr * scale != w_hr):
            raise SystemExit(f"ERROR: HR must be LR×{scale}. LR=({h_lr},{w_lr}) HR=({h_hr},{w_hr})")
        hr_t = to_tensor(hr_img)

    # Build model + weights
    model = EDSR(**MODEL_KW).to(DEVICE)
    load_state_flex(model, CKPT_PATH)
    model.eval()
    model.apply(disable_inplace)

    # Move tensors
    lr_t = lr_t.to(DEVICE)
    if hr_t is not None:
        hr_t = hr_t.to(DEVICE)

    # Compute saliency (LR space) + SR image
    sal01_lr, sr_rgb = compute_saliency(
        model, lr_t, hr_t,
        mode=MODE, pixel_xy=PIXEL_XY, patch_xywh=PATCH_XYWH
    )

    # Save SR
    save_png(os.path.join(OUTDIR, "sr.png"), sr_rgb)

    # Save saliency at LR
    gray3_lr = np.repeat(sal01_lr[..., None], 3, axis=2)
    save_png(os.path.join(OUTDIR, "saliency_gray_lr.png"), gray3_lr)

    # Prepare SR-sized saliency for color/overlay and SR-gray
    Hsr, Wsr = sr_rgb.shape[:2]
    sal01_sr = cv2.resize(sal01_lr, (Wsr, Hsr), interpolation=cv2.INTER_LINEAR)
    gray3_sr = np.repeat(sal01_sr[..., None], 3, axis=2)
    save_png(os.path.join(OUTDIR, "saliency_gray_sr.png"), gray3_sr)

    # Heatmap + overlay at SR size (robust function resizes if needed)
    heat_rgb, overlay_rgb = colorize_and_overlay(sr_rgb, sal01_sr, alpha=ALPHA_OVERLAY)
    save_png(os.path.join(OUTDIR, "saliency_color.png"), np.clip(heat_rgb.astype(np.float32)/255.0, 0, 1))
    save_png(os.path.join(OUTDIR, "saliency_overlay.png"), np.clip(overlay_rgb.astype(np.float32)/255.0, 0, 1))

    # Index
    with open(os.path.join(OUTDIR, "index.txt"), "w", encoding="utf-8") as f:
        f.write("Saliency map (input gradient magnitude) for EDSR\n")
        f.write(f"Mode: {MODE} | p_low={P_LOW} p_high={P_HIGH} gamma={GAMMA}\n")
        f.write(f"LR: {LR_PATH}\n")
        if MODE == "loss":
            f.write(f"HR: {HR_PATH}\n")
        f.write("\nFiles:\n")
        f.write("  sr.png\n")
        f.write("  saliency_gray_lr.png\n")
        f.write("  saliency_gray_sr.png\n")
        f.write("  saliency_color.png\n")
        f.write("  saliency_overlay.png\n")

    print("\n[Done]")
    print("  Output dir:", os.path.abspath(OUTDIR))
    print("  Files listed in index.txt")


if __name__ == "__main__":
    main()
