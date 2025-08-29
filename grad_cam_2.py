#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM over EVERY layer of your EDSR (BasicSR), robust to multi-output layers.

Outputs:
  - sr.png
  - cam_gray_<layer>.png, cam_jet_<layer>.png, overlay_<layer>.png  (for layers with grad)
  - index.txt (lists all layers, marks SKIPPED when no grad was captured)

Run:
    python grad_cam_all_layers.py
"""

import os
import re
from typing import Tuple, Optional, List, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Your EDSR implementation =====
from basicsr.archs.edsr_arch import EDSR


# =========================
# ======= CONFIG ==========
# =========================
OUTDIR = "gradcam_all_layers_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- YOUR PATHS (as requested) ----
LR_PATH  = "datasets/Test/Set5/LRx4/baby.png"  # required
HR_PATH  = "datasets/Test/Set5/HR/baby.png"    # required for MODE='loss'
CKPT_PATH = "experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models/net_g_latest.pth"

# ---- Model params (as requested) ----
MODEL_KW = dict(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=192,
    num_block=6,
    upscale=4,
    num_heads=6,
    hw=8,
    ww=8,
    img_range=255.,
    rgb_mean=[0.4488, 0.4371, 0.4040],
)

# Grad-CAM target
MODE = "loss"             # 'loss' | 'pixel' | 'patch'
PIXEL_XY = None           # (x, y) in SR coords if MODE='pixel'
PATCH_XYWH = None         # (x, y, w, h) in SR coords if MODE='patch'

# What modules to CAM
INCLUDE_CONV = True
INCLUDE_BASICBLOCK = True
MAX_TARGETS: Optional[int] = None  # limit if desired (e.g., 80)


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
    img = img.astype(np.float32) / 255.0
    return img

def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError("Expected HxWxC image")
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img

def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().float()

def save_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = img
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 1)
        a = (a * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, a)

def colorize_and_overlay(sr_rgb: np.ndarray, cam: np.ndarray, alpha=0.45):
    """
    sr_rgb: HxWx3 float [0,1]
    cam   : HxW float [0,1]
    -> (heatmap_rgb_uint8, overlay_rgb_uint8)
    """
    heat_u8 = (np.clip(cam, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    sr_u8 = (np.clip(sr_rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    overlay = cv2.addWeighted(sr_u8, 1.0 - alpha, heat_rgb, alpha, 0.0)
    return heat_rgb, overlay

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)

def disable_inplace(m: nn.Module):
    # Prevent in-place activation quirks that break grad hooks
    if isinstance(m, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU)):
        if hasattr(m, "inplace") and m.inplace:
            m.inplace = False

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
    if missing:    print("  missing (first 10):", missing[:10])
    if unexpected: print("  unexpected (first 10):", unexpected[:10])

def first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """
    Return the first Tensor we can find inside obj (Tensor / tuple / list / dict),
    preferring one that requires_grad if possible.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        cand_req = [x for x in obj if isinstance(x, torch.Tensor) and x.requires_grad]
        if cand_req:
            return cand_req[0]
        cand_any = [x for x in obj if isinstance(x, torch.Tensor)]
        return cand_any[0] if cand_any else None
    if isinstance(obj, dict):
        # prefer values named 'out' or similar
        for k in ["out", "output", "y"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
        for v in obj.values():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                return v
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    return None


class GradCAM:
    """
    Robust Grad-CAM:
      - Forward hook captures the module's forward output (supports Tensor / tuple / list / dict)
      - A tensor-level grad hook is attached to that captured tensor
    If a layer's output truly doesn't carry grad (e.g., detached branch), this will result in
    no gradients captured; the caller should skip that layer.
    """
    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fh = None
        self._th = None

        def fwd_hook(module, inp, out):
            act = first_tensor(out)
            self.activations = act
            # clean previous tensor hook if any
            if self._th is not None:
                try:
                    self._th.remove()
                except Exception:
                    pass
                self._th = None
            if isinstance(act, torch.Tensor) and act.requires_grad:
                def _grad_hook(grad):
                    self.gradients = grad.detach().clone()
                self._th = act.register_hook(_grad_hook)

        self._fh = target_module.register_forward_hook(fwd_hook)

    def remove(self):
        if self._fh is not None:
            try: self._fh.remove()
            except Exception: pass
            self._fh = None
        if self._th is not None:
            try: self._th.remove()
            except Exception: pass
            self._th = None

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor, eps=1e-8) -> torch.Tensor:
        xmin = x.amin(dim=(-2, -1), keepdim=True)
        xmax = x.amax(dim=(-2, -1), keepdim=True)
        return (x - xmin) / (xmax - xmin + eps)

    def __call__(self, scalar: torch.Tensor, upsize_to: Tuple[int, int]) -> Optional[torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        scalar.backward()

        if self.gradients is None or self.activations is None or not isinstance(self.activations, torch.Tensor):
            # No grad captured for this target (detached path or unused module)
            return None

        # activations / gradients: (B,C,h,w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # (B,C,1,1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = self._normalize(cam)
        cam = F.interpolate(cam, size=upsize_to, mode="bilinear", align_corners=False)
        return cam  # (B,1,H,W) in [0,1]


def make_scalar_target(
    sr: torch.Tensor,
    hr: Optional[torch.Tensor],
    mode: str,
    pixel_xy: Optional[Tuple[int, int]] = None,
    patch_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    if mode == "loss":
        if hr is None:
            raise ValueError("MODE='loss' requires HR_PATH.")
        return F.mse_loss(sr, hr)

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

def collect_targets(model: nn.Module) -> List[tuple]:
    targets = []
    seen = set()
    for name, m in model.named_modules():
        add = False
        if INCLUDE_CONV and isinstance(m, nn.Conv2d):
            add = True
        if INCLUDE_BASICBLOCK and (m.__class__.__name__ == "BasicBlock"):
            add = True
        if add and (id(m) not in seen):
            seen.add(id(m))
            targets.append((name, m))
    return targets


# =========================
# ========= MAIN ==========
# =========================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Load LR/HR
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

    # 2) Build model + load weights
    model = EDSR(**MODEL_KW).to(DEVICE)
    load_state_flex(model, CKPT_PATH)
    model.eval()
    model.apply(disable_inplace)  # avoid in-place ReLU issues

    # 3) Move tensors
    lr_t = lr_t.to(DEVICE)
    if hr_t is not None:
        hr_t = hr_t.to(DEVICE)

    # 4) Collect targets
    targets = collect_targets(model)
    if MAX_TARGETS is not None:
        targets = targets[:MAX_TARGETS]
    print(f"[Info] Number of targets to CAM: {len(targets)}")

    index_lines = []
    sr_np_cached = None

    # 5) Loop over targets
    for idx, (name, module) in enumerate(targets, 1):
        print(f"[{idx}/{len(targets)}] Layer: {name}")
        tag = sanitize(name)

        cam_engine = GradCAM(model, module)

        # Forward after hook registration
        sr = model(lr_t)
        Hsr, Wsr = sr.shape[-2:]

        scalar = make_scalar_target(sr, hr_t, mode=MODE, pixel_xy=PIXEL_XY, patch_xywh=PATCH_XYWH)

        cam = cam_engine(scalar, upsize_to=(Hsr, Wsr))
        cam_engine.remove()

        if cam is None:
            index_lines.append(f"{idx:03d}  {name}   SKIPPED (no grad captured)")
            # Clean grads and continue
            model.zero_grad(set_to_none=True)
            continue

        cam_np = cam.detach().cpu().squeeze(0).squeeze(0).numpy()
        sr_np = sr.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        sr_np = np.clip(sr_np, 0.0, 1.0)
        sr_np_cached = sr_np

        heat_rgb, overlay_rgb = colorize_and_overlay(sr_np, cam_np, alpha=0.45)

        out_cam_gray = os.path.join(OUTDIR, f"cam_gray_{tag}.png")
        out_cam_jet  = os.path.join(OUTDIR, f"cam_jet_{tag}.png")
        out_overlay  = os.path.join(OUTDIR, f"overlay_{tag}.png")
        save_png(out_cam_gray, np.stack([cam_np]*3, axis=-1))
        save_png(out_cam_jet,  heat_rgb.astype(np.uint8))
        save_png(out_overlay,  overlay_rgb.astype(np.uint8))

        index_lines.append(
            f"{idx:03d}  {name}   OK\n"
            f"      cam_gray={os.path.basename(out_cam_gray)}  "
            f"cam_jet={os.path.basename(out_cam_jet)}  "
            f"overlay={os.path.basename(out_overlay)}"
        )

        model.zero_grad(set_to_none=True)

    # 6) Save SR once
    if sr_np_cached is None:
        sr_np_cached = model(lr_t).detach().cpu().squeeze(0).permute(1,2,0).numpy()
        sr_np_cached = np.clip(sr_np_cached, 0.0, 1.0)
    save_png(os.path.join(OUTDIR, "sr.png"), sr_np_cached)

    # 7) Write index
    with open(os.path.join(OUTDIR, "index.txt"), "w", encoding="utf-8") as f:
        f.write("Grad-CAM outputs per layer\n")
        f.write(f"Model: EDSR | Mode: {MODE}\n\n")
        for line in index_lines:
            f.write(line + "\n")

    print("\n[Done]")
    print("  Output dir:", os.path.abspath(OUTDIR))
    print("  Files listed in index.txt")


if __name__ == "__main__":
    main()
