#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM for each component inside every BasicBlock of EDSR (BasicSR)

Saves (per component):
  - cam_gray_body<idx>_<component>.png
  - cam_color_body<idx>_<component>.png
  - overlay_body<idx>_<component>.png
Also:
  - sr.png
  - index.txt (OK / SKIPPED per target)

Requirements:
  pip install torch torchvision opencv-python numpy
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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
OUTDIR = "gradcam_basicblock_components_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Your exact paths ----
LR_PATH  = "datasets/Test/Urban100/LRx4/img001.png"
HR_PATH  = "datasets/Test/Urban100/HR/img001.png"
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

# Grad-CAM target scalar:
MODE = "pixel"          # 'loss' | 'pixel' | 'patch'
PIXEL_XY = None        # SR coords, if MODE='pixel'
PATCH_XYWH = None      # SR coords, if MODE='patch'

# Visualization tuning (more conservative to avoid washout)
P_LOW, P_HIGH = 5.0, 95.0   # percentile clip for contrast
GAMMA = 1.0                 # 1.0 = neutral; <1 brightens
CAM_RELU = True             # standard Grad-CAM positive-only
CMAP_ID = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
ALPHA_OVERLAY = 0.40

# Which submodules (inside each BasicBlock) to CAM:
COMPONENT_PATHS = [
    # projections
    "proj0", "proj1",

    # spatial/frequency encoders (top-level)
    "spatial_encoder1", "freq_encoder1",
    "spatial_encoder2", "freq_encoder2",

    # attention INSIDE encoders (have (h,w) inputs; safe to reshape)
    # "spatial_encoder1.spatialattention",
    # "spatial_encoder2.spatialattention",
    # "freq_encoder1.spatialattention1",
    # "freq_encoder1.spatialattention2",
    # "freq_encoder1.spatialattention3",
    # "freq_encoder2.spatialattention1",
    # "freq_encoder2.spatialattention2",
    # "freq_encoder2.spatialattention3",

    # cross encoder (and its attentions, both have (h,w) inputs)
    "cross_encoder",
    # "cross_encoder.spatial_crossattention",
    # "cross_encoder.freq_crossattention",

    # band mixers
    "freq_mix1", "freq_mix2",

    # fusion (and the internal convs)
    "fusion",
    # "fusion.0", "fusion.2", "fusion.4",
]


# =========================
# ======= HELPERS =========
# =========================

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\\-]+", "_", name)

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
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().float()

def save_png(path: str, img01: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.clip(img01, 0, 1)
    a = (a * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, a)

def colorize_and_overlay(sr_rgb: np.ndarray, cam_01: np.ndarray, alpha=ALPHA_OVERLAY):
    cam_u8 = (np.clip(cam_01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, CMAP_ID)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    sr_u8 = (np.clip(sr_rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    overlay = cv2.addWeighted(sr_u8, 1.0 - alpha, heat_rgb, alpha, 0.0)
    return heat_rgb, overlay

def postprocess_cam_01(cam_up: np.ndarray, p_low=P_LOW, p_high=P_HIGH, gamma=GAMMA) -> np.ndarray:
    # Handle NaN/Inf early
    x = cam_up.astype(np.float32)
    m = np.isfinite(x)
    if not m.all():
        # replace non-finite with median of finite
        if m.any():
            med = np.median(x[m])
            x[~m] = med
        else:
            return np.full_like(x, 0.5, dtype=np.float32)

    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.full_like(x, 0.5, dtype=np.float32)

    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, gamma, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)

def disable_inplace(m: nn.Module):
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

def get_attr_path(root: nn.Module, path: str) -> Optional[nn.Module]:
    cur = root
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur if isinstance(cur, nn.Module) else None


# =========================
# ===== Grad-CAM Core =====
# =========================

def _extract_hw_from_inputs(inp_tuple: Tuple[Any, ...]) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to pull (h, w) from the module's forward inputs.
    We look from the RIGHT (many signatures are (..., h, w)).
    """
    h_in, w_in = None, None
    # flatten containers just in case (but typical inputs are tensors and ints)
    flat: List[Any] = []
    for it in inp_tuple:
        if isinstance(it, (list, tuple)):
            flat.extend(list(it))
        else:
            flat.append(it)
    # scan from right to left for ints
    ints: List[int] = []
    for it in reversed(flat):
        if isinstance(it, int) and it > 0:
            ints.append(int(it))
            if len(ints) >= 2:
                break
    if len(ints) == 2:
        # we collected [w, h] from right to left; reverse to (h,w)
        w_in, h_in = ints[0], ints[1]
        return h_in, w_in
    return None, None


class FlexGradCAM:
    """
    Safe Grad-CAM that:
      - Captures forward output tensor
      - If 3D (B,N,C), reshapes to (B,C,H,W) using H/W from inputs
      - If 4D and looks like NHWC using known (h,w), permutes to NCHW
      - Uses tensor-level grad hook (no fragile backward hooks)
    """
    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.act_raw: Optional[torch.Tensor] = None  # raw activation (as returned)
        self.grad_raw: Optional[torch.Tensor] = None
        self.h_in: Optional[int] = None
        self.w_in: Optional[int] = None
        self._fh = None
        self._th = None

        def fwd_hook(module, inp, out):
            # Extract (h, w) if present among inputs
            h_guess, w_guess = _extract_hw_from_inputs(inp)
            if h_guess is not None and w_guess is not None:
                self.h_in, self.w_in = int(h_guess), int(w_guess)

            # Activation tensor (first tensor if container)
            act = self._first_tensor(out)
            self.act_raw = act

            # Reset prior tensor hook if any
            if self._th is not None:
                try: self._th.remove()
                except Exception: pass
                self._th = None

            if isinstance(act, torch.Tensor) and act.requires_grad:
                def _grad_hook(grad):
                    self.grad_raw = grad.detach().clone()
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

    @staticmethod
    def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            for x in obj:
                if isinstance(x, torch.Tensor):
                    return x
        if isinstance(obj, dict):
            for k in ["out", "output", "y"]:
                if k in obj and isinstance(obj[k], torch.Tensor):
                    return obj[k]
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    return v
        return None

    def _as_4d(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert raw activation/gradient to (B,C,H,W).
        Handles NCHW, NHWC (with known h/w), and (B,N,C) with known h/w.
        """
        if t is None or not isinstance(t, torch.Tensor):
            return None

        # (B,C,H,W) or (B,H,W,C)
        if t.dim() == 4:
            B, D1, D2, D3 = t.shape  # interpret generically
            # If we know (h,w), try to detect layout unambiguously:
            if self.h_in is not None and self.w_in is not None:
                h, w = int(self.h_in), int(self.w_in)
                # NCHW -> dims (B,C,H,W): D2==h and D3==w
                if D2 == h and D3 == w:
                    return t
                # NHWC -> dims (B,H,W,C): D1==h and D2==w
                if D1 == h and D2 == w:
                    return t.permute(0, 3, 1, 2).contiguous()
            # Fallback: assume NCHW (common for convs)
            return t

        # (B,N,C) or (B,C,N): need (h,w)
        if t.dim() == 3 and (self.h_in is not None and self.w_in is not None):
            B, A, B_or_C = t.shape
            h, w = int(self.h_in), int(self.w_in)
            if A == h * w:
                # (B, H*W, C) -> (B,C,H,W)
                C = B_or_C
                return t.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
            if B_or_C == h * w:
                # (B, C, H*W) -> (B,C,H,W)
                C = A
                return t.reshape(B, C, h, w).contiguous()

        # Nothing we can do
        return None

    def __call__(self, scalar: torch.Tensor, upsize_to: Tuple[int, int]) -> Optional[np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        scalar.backward()

        if self.act_raw is None or self.grad_raw is None:
            return None

        # Convert both to (B,C,H,W)
        act4 = self._as_4d(self.act_raw)
        grd4 = self._as_4d(self.grad_raw)
        if act4 is None or grd4 is None:
            return None

        # Guard against NaNs/Infs
        if not torch.isfinite(act4).all():
            act4 = torch.nan_to_num(act4, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(grd4).all():
            grd4 = torch.nan_to_num(grd4, nan=0.0, posinf=0.0, neginf=0.0)

        # Grad-CAM weights and map
        weights = grd4.mean(dim=(2, 3), keepdim=True)           # (B,C,1,1)
        cam = (weights * act4).sum(dim=1, keepdim=True)         # (B,1,h,w)
        if CAM_RELU:
            cam = torch.relu(cam)

        # Upsample then postprocess (percentile+gamma)
        cam_up = F.interpolate(cam, size=upsize_to, mode="bilinear", align_corners=False)
        cam_np = cam_up.detach().float().cpu().squeeze(0).squeeze(0).numpy()
        cam_01 = postprocess_cam_01(cam_np)
        return cam_01


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
    model.apply(disable_inplace)

    # 3) Move tensors
    lr_t = lr_t.to(DEVICE)
    if hr_t is not None:
        hr_t = hr_t.to(DEVICE)

    # 4) Find BasicBlocks
    blocks: List[Tuple[int, nn.Module]] = []
    for i, m in enumerate(model.body.children()):
        if m.__class__.__name__ == "BasicBlock":
            blocks.append((i, m))
    if not blocks:
        print("No BasicBlock modules found under model.body")
        return
    print(f"[Info] Found {len(blocks)} BasicBlocks")

    index_lines: List[str] = []
    sr_np_cached = None

    # 5) Iterate over blocks and their components
    for blk_idx, blk in blocks:
        for rel_path in COMPONENT_PATHS:
            target = get_attr_path(blk, rel_path)
            if target is None:
                continue

            full_name = f"body.{blk_idx}.{rel_path}"
            tag = _sanitize(full_name)
            print(f"[Block {blk_idx}] Component: {rel_path}")

            cam_engine = FlexGradCAM(model, target)

            # Forward AFTER registering hook
            sr = model(lr_t)
            Hsr, Wsr = sr.shape[-2:]

            scalar = make_scalar_target(sr, hr_t, mode=MODE, pixel_xy=PIXEL_XY, patch_xywh=PATCH_XYWH)
            cam01 = cam_engine(scalar, upsize_to=(Hsr, Wsr))
            cam_engine.remove()

            if cam01 is None:
                index_lines.append(f"{full_name:45s}  SKIPPED (no spatial CAM / no grad)")
                model.zero_grad(set_to_none=True)
                continue

            # SR numpy for overlays
            sr_np = sr.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            sr_np = np.clip(sr_np, 0.0, 1.0)
            sr_np_cached = sr_np

            # Save visualizations
            gray3 = np.repeat(cam01[..., None], 3, axis=2)
            heat_rgb, overlay_rgb = colorize_and_overlay(sr_np, cam01, alpha=ALPHA_OVERLAY)

            out_gray = os.path.join(OUTDIR, f"cam_gray_{tag}.png")
            out_col  = os.path.join(OUTDIR, f"cam_color_{tag}.png")
            out_ovr  = os.path.join(OUTDIR, f"overlay_{tag}.png")
            save_png(out_gray, gray3)
            save_png(out_col,  np.clip(heat_rgb.astype(np.float32)/255.0,0,1))
            save_png(out_ovr,  np.clip(overlay_rgb.astype(np.float32)/255.0,0,1))

            index_lines.append(
                f"{full_name:45s}  OK  "
                f"gray={os.path.basename(out_gray)}  "
                f"color={os.path.basename(out_col)}  "
                f"overlay={os.path.basename(out_ovr)}"
            )

            model.zero_grad(set_to_none=True)

    # 6) Save SR once
    if sr_np_cached is None:
        sr_np_cached = model(lr_t).detach().cpu().squeeze(0).permute(1,2,0).numpy()
        sr_np_cached = np.clip(sr_np_cached, 0.0, 1.0)
    save_png(os.path.join(OUTDIR, "sr.png"), sr_np_cached)

    # 7) Write index
    with open(os.path.join(OUTDIR, "index.txt"), "w", encoding="utf-8") as f:
        f.write("Grad-CAM per BasicBlock component (percentile-normalized)\n")
        f.write(f"Model: EDSR | Mode: {MODE} | p_low={P_LOW} p_high={P_HIGH} gamma={GAMMA}\n\n")
        for line in index_lines:
            f.write(line + "\n")

    print("\n[Done]")
    print("  Output dir:", os.path.abspath(OUTDIR))
    print("  Files listed in index.txt")


if __name__ == "__main__":
    main()
