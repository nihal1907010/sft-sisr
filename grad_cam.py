#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDSR Grad-CAM (BasicSR) - no CLI args
-------------------------------------
Edit the CONFIG section below and run:
    python edsr_gradcam_nocli.py

Requirements:
    pip install torch torchvision basicsr opencv-python pillow numpy
"""

import os
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- BasicSR EDSR ----
from basicsr.archs.edsr_arch import EDSR


# =========================
# ======= CONFIGURE =======
# =========================
OUTDIR = "gradcam_out"     # where to save outputs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# I/O
LR_PATH = "datasets/Test/Set5/LRx4/baby.png"      # required
HR_PATH = "datasets/Test/Set5/HR/baby.png"      # required for MODE='loss', else can be None
CKPT_PATH = "experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models/net_g_latest.pth"    # optional; set to None to use random weights

# Model
UPSCALE = 4  # 2, 3, or 4

# Grad-CAM target mode: 'loss' | 'pixel' | 'patch'
MODE = "loss"

# For MODE='pixel' (SR coordinates): (x, y)
PIXEL_XY: Optional[Tuple[int, int]] = (128, 128)

# For MODE='patch' (SR coordinates): (x, y, w, h)
PATCH_XYWH: Optional[Tuple[int, int, int, int]] = (100, 100, 48, 48)


# =========================
# ====== IMPLEMENTS =======
# =========================

def read_image(path: str, to_rgb=True) -> np.ndarray:
    """Read image as float32 in [0,1], HxWxC."""
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = img[..., None]
    if to_rgb and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def ensure_3ch(img: np.ndarray) -> np.ndarray:
    """Make sure image is 3-channel (repeat grayscale if needed)."""
    if img is None:
        return None
    if img.ndim != 3:
        raise ValueError("Expected HxWxC image")
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWxC [0,1] -> 1xCxHxW float32"""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.float()


def save_png(path: str, img: np.ndarray) -> None:
    """Save RGB or gray float/uint8 image with automatic conversion."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = img
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 1)
        a = (a * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, a)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fh = target_layer.register_forward_hook(self._save_activation)
        # Backward hook compatibility
        try:
            self.bh = target_layer.register_full_backward_hook(self._save_gradient)
        except Exception:
            self.bh = target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] has gradient w.r.t. module output
        self.gradients = grad_out[0]

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor, eps=1e-8) -> torch.Tensor:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def __call__(self, out_scalar: torch.Tensor, upsize_to: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Return CAM in [0,1], shape (B,1,H,W)."""
        self.model.zero_grad(set_to_none=True)
        out_scalar.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)             # (B,C,1,1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (B,1,H,W)
        cam = self._normalize(cam)
        if upsize_to is not None:
            cam = F.interpolate(cam, size=upsize_to, mode="bilinear", align_corners=False)
        return cam


def find_last_conv_in_body(model: nn.Module) -> nn.Module:
    """Return the last nn.Conv2d inside model.body (robust default for EDSR)."""
    last = None
    for m in model.body.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Could not find a Conv2d inside model.body")
    return last


def load_state_flex(model: nn.Module, ckpt_path: Optional[str]) -> None:
    if ckpt_path is None:
        print("[Info] No checkpoint provided; using random weights.")
        return
    print(f"[Info] Loading checkpoint: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["params_ema", "params", "state_dict", "net", "model"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise RuntimeError("Unsupported checkpoint format.")
    missing, unexpected = model.load_state_dict(obj, strict=False)
    print(f"[Info] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:   print("  missing (truncated):", missing[:10])
    if unexpected:print("  unexpected (truncated):", unexpected[:10])


def make_scalar_target(
    sr: torch.Tensor,
    hr: Optional[torch.Tensor],
    mode: str,
    pixel_xy: Optional[Tuple[int, int]] = None,
    patch_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    """
    - mode='loss'  : requires hr; scalar = MSE(sr, hr)
    - mode='pixel' : scalar = mean over SR patch around (x,y)
    - mode='patch' : scalar = mean over SR[y:y+h, x:x+w]
    """
    if mode == "loss":
        if hr is None:
            raise ValueError("MODE='loss' requires HR_PATH to be set.")
        return F.mse_loss(sr, hr)

    B, C, H, W = sr.shape

    if mode == "pixel":
        if pixel_xy is None:
            y, x = H // 2, W // 2
        else:
            x, y = int(pixel_xy[0]), int(pixel_xy[1])
            x = int(np.clip(x, 0, W - 1))
            y = int(np.clip(y, 0, H - 1))
        # 3x3 neighborhood mean for stability
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        return sr[..., y0:y1, x0:x1].mean()

    if mode == "patch":
        if patch_xywh is None:
            cy, cx = H // 2, W // 2
            h, w = min(32, H), min(32, W)
            y0, x0 = max(0, cy - h // 2), max(0, cx - w // 2)
        else:
            x0, y0, w, h = map(int, patch_xywh)
            x0 = int(np.clip(x0, 0, W - 1))
            y0 = int(np.clip(y0, 0, H - 1))
            w = int(np.clip(w, 1, W - x0))
            h = int(np.clip(h, 1, H - y0))
        return sr[..., y0:y0 + h, x0:x0 + w].mean()

    raise ValueError(f"Unknown MODE: {mode}")


def colorize_and_overlay(sr_rgb: np.ndarray, cam: np.ndarray, alpha=0.45):
    """Return (heatmap_rgb_uint8, overlay_rgb_uint8)."""
    heat_u8 = (np.clip(cam, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)

    sr_u8 = (np.clip(sr_rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    overlay_rgb = cv2.addWeighted(sr_u8, 1.0 - alpha, heat_color_rgb, alpha, 0.0)
    return heat_color_rgb, overlay_rgb


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load LR/HR
    lr_img = ensure_3ch(read_image(LR_PATH, to_rgb=True))
    lr_t = to_tensor(lr_img)  # 1x3xhxw

    hr_t = None
    if MODE == "loss":
        if HR_PATH is None:
            raise SystemExit("ERROR: MODE='loss' requires HR_PATH.")
        hr_img = ensure_3ch(read_image(HR_PATH, to_rgb=True))
        # size check: HR must be LR × UPSCALE
        h_lr, w_lr = lr_img.shape[:2]
        h_hr, w_hr = hr_img.shape[:2]
        if (h_lr * UPSCALE != h_hr) or (w_lr * UPSCALE != w_hr):
            raise SystemExit(f"ERROR: HR size must be LR size × UPSCALE={UPSCALE}. "
                             f"Got LR=({h_lr},{w_lr}) HR=({h_hr},{w_hr}).")
        hr_t = to_tensor(hr_img)

    # Build EDSR and load weights
    model = EDSR(num_in_ch=3,
             num_out_ch=3,
             num_feat=192,
             num_block=6,
             upscale=4,
             num_heads=6,
             hw=8,
             ww=8,
             img_range=255.,
             rgb_mean=[0.4488, 0.4371, 0.4040])
    load_state_flex(model, CKPT_PATH)
    model.eval().to(DEVICE)

    # Target layer (last Conv in body). Register hooks BEFORE forward.
    target_layer = find_last_conv_in_body(model)
    cam_engine = GradCAM(model, target_layer)

    # Move tensors
    lr_t = lr_t.to(DEVICE)
    if hr_t is not None:
        hr_t = hr_t.to(DEVICE)

    # Forward (need grads)
    sr_t = model(lr_t)  # 1x3xHsrxWsr
    Hsr, Wsr = sr_t.shape[-2:]

    # Scalar to backprop
    out_scalar = make_scalar_target(
        sr_t, hr_t,
        mode=MODE,
        pixel_xy=PIXEL_XY,
        patch_xywh=PATCH_XYWH,
    )

    # Grad-CAM
    cam_t = cam_engine(out_scalar, upsize_to=(Hsr, Wsr))  # 1x1xHsrxWsr in [0,1]
    cam_engine.remove_hooks()

    # To numpy
    sr_np = sr_t.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    sr_np = np.clip(sr_np, 0.0, 1.0)
    cam_np = cam_t.detach().float().cpu().squeeze(0).squeeze(0).numpy()

    # Heatmap + overlay
    heat_rgb, overlay_rgb = colorize_and_overlay(sr_np, cam_np, alpha=0.45)

    # Save
    out_sr = os.path.join(OUTDIR, "sr.png")
    out_cam_gray = os.path.join(OUTDIR, "cam_gray.png")
    out_cam_color = os.path.join(OUTDIR, "cam_jet.png")
    out_overlay = os.path.join(OUTDIR, "overlay.png")
    out_cam_npy = os.path.join(OUTDIR, "cam_raw.npy")

    save_png(out_sr, sr_np)
    save_png(out_cam_gray, np.stack([cam_np] * 3, axis=-1))  # 3-ch gray for easy viewing
    save_png(out_cam_color, heat_rgb.astype(np.uint8))
    save_png(out_overlay, overlay_rgb.astype(np.uint8))
    np.save(out_cam_npy, cam_np.astype(np.float32))

    print("\n[Saved]")
    print("  SR image     :", out_sr)
    print("  CAM (gray)   :", out_cam_gray)
    print("  CAM (JET)    :", out_cam_color)
    print("  Overlay      :", out_overlay)
    print("  CAM (raw npy):", out_cam_npy)
    print("\nDone.")


if __name__ == "__main__":
    main()
