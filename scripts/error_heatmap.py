# ================================================================
# Super-Resolution Error Visualization for BasicSR - EDSR
# ---------------------------------------------------------------
# Configure the five variables below and run the script.
# It will:
#  - load an EDSR generator from a BasicSR checkpoint
#  - process LR/HR image pairs
#  - produce SR outputs, error heatmaps, and side-by-side panels
#  - compute PSNR/SSIM (RGB + Y) and save a CSV report
# ================================================================

import os
import glob
import cv2
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# -------------------------
# 1) USER CONFIG (EDIT ME)
# -------------------------
LR_FOLDER     = "datasets/Test/Set5/LRx4"         # folder with LR images (png/jpg)
HR_FOLDER     = "datasets/Test/Set5/HR"         # folder with GT HR images (same basenames)
CKPT_PATH     = "experiments/v201_f64_b8_h8_hw8_b4a3_1em3_300k_5em1_75k150k225k_x4/models/net_g_latest.pth"  # BasicSR checkpoint (net_g / params_ema / state_dict)
OUTPUT_ROOT   = "./viz_out_edsr"             # where outputs and CSV go
UPSCALE       = 4                              # 2/3/4 typically

# If your checkpoint does not store model config, set these to match your model.
# They will be auto-inferred if the ckpt has enough info; otherwise this fallback is used.
EDSR_CONFIG = dict(
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

# Shave border for metrics (commonly = UPSCALE for SR benchmarks)
SHAVE = UPSCALE

# -------------------------
# Helper: load EDSR (BasicSR)
# -------------------------
def build_edsr(arch_cfg):
    # EDSR lives in basicsr.archs.edsr_arch
    try:
        from basicsr.archs.edsr_arch import EDSR
    except Exception as e:
        raise RuntimeError(
            "Could not import EDSR from basicsr. "
            "Install basicsr: pip install basicsr"
        ) from e
    # net = EDSR(
    #     num_in_ch=arch_cfg.get("num_in_ch", 3),
    #     num_out_ch=arch_cfg.get("num_out_ch", 3),
    #     num_feat=arch_cfg.get("num_feat", 64),
    #     num_block=arch_cfg.get("num_block", 16),
    #     res_scale=arch_cfg.get("res_scale", 1.0),
    #     upscale=arch_cfg.get("upscale", UPSCALE),
    # )
    net = EDSR(**EDSR_CONFIG)
    return net

def _strip_prefix(state_dict, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def _pick_state_dict(ckpt):
    # Try common BasicSR layouts
    for key in ["params_ema", "net_g", "state_dict", "model", "params"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # Otherwise assume whole file is a state dict
    return ckpt

def _infer_arch_cfg_from_ckpt(ckpt, fallback):
    # Try to extract arch/options if present (many BasicSR ckpts save opt in 'opt' as a dict/json)
    cfg = dict(fallback)
    if "opt" in ckpt:
        try:
            opt = ckpt["opt"]
            if isinstance(opt, (str, bytes)):
                opt = json.loads(opt)
            # Common places: opt['network_g'] or opt['network_g']['type']=='EDSR'
            ng = opt.get("network_g", {})
            if isinstance(ng, dict) and (ng.get("type", "").lower() == "edsr"):
                for k in ["num_in_ch", "num_out_ch", "num_feat", "num_block", "res_scale", "upscale"]:
                    if k in ng:
                        cfg[k] = ng[k]
        except Exception:
            pass
    return cfg

def load_model(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch_cfg = _infer_arch_cfg_from_ckpt(ckpt, EDSR_CONFIG)
    net = build_edsr(arch_cfg)
    sd = _pick_state_dict(ckpt)

    # Handle nested prefixes (common: 'net_g.' or 'module.')
    for pref in ["net_g.", "model.", "module."]:
        if any(k.startswith(pref) for k in sd.keys()):
            sd = _strip_prefix(sd, pref)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    net.eval().to(device)
    return net, device, arch_cfg

# -------------------------
# Image utilities
# -------------------------
def imread_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_tensor(img_rgb):
    # HWC [0,255] -> BCHW float [0,1]
    x = torch.from_numpy(img_rgb).float() / 255.0  # HWC
    x = x.permute(2, 0, 1).unsqueeze(0)           # BCHW
    return x

def to_uint8_rgb(t):
    # BCHW float [0,1] -> HWC uint8
    x = t.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def rgb_to_y(img_rgb):
    # ITU-R BT.601 conversion (same used in many SR benchmarks)
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y  # range roughly [16, 235]

def shave_to_common(a, b, shave):
    # center crop both so they share same (H-2*shave, W-2*shave)
    H = min(a.shape[0], b.shape[0]) - 2 * shave
    W = min(a.shape[1], b.shape[1]) - 2 * shave
    H = max(H, 0); W = max(W, 0)
    def center_crop(x):
        hh, ww = x.shape[:2]
        top = (hh - H) // 2
        left = (ww - W) // 2
        return x[top:top+H, left:left+W]
    return center_crop(a), center_crop(b)

# -------------------------
# Visualization utilities
# -------------------------
def error_heatmap_abs(sr_rgb, hr_rgb, clip_min=0, clip_max=255):
    # |SR - HR| averaged over channels -> normalize -> apply JET
    err = np.mean(np.abs(sr_rgb.astype(np.float32) - hr_rgb.astype(np.float32)), axis=2)
    err = (err - err.min()) / (err.max() - err.min() + 1e-8)
    heat = cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat

def error_heatmap_sq(sr_rgb, hr_rgb):
    err = np.mean((sr_rgb.astype(np.float32) - hr_rgb.astype(np.float32)) ** 2, axis=2)
    err = (err - err.min()) / (err.max() - err.min() + 1e-8)
    heat = cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat

def tile_h(images, pad=6, pad_val=255):
    h = max(im.shape[0] for im in images)
    rows = []
    for im in images:
        if im.shape[0] < h:
            # pad vertically to h
            top = (h - im.shape[0]) // 2
            bottom = h - im.shape[0] - top
            im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[pad_val]*3)
        rows.append(im)
    sep = np.full((h, pad, 3), pad_val, np.uint8)
    out = rows[0]
    for im in rows[1:]:
        out = np.concatenate([out, sep, im], axis=1)
    return out

def annotate(img_rgb, text, font_scale=0.5, thickness=1):
    out = img_rgb.copy()
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return out

# -------------------------
# Metrics
# -------------------------
def metrics_rgb(sr_rgb, hr_rgb, shave=0):
    if shave > 0:
        sr_c, hr_c = shave_to_common(sr_rgb, hr_rgb, shave)
    else:
        sr_c, hr_c = sr_rgb, hr_rgb
    # PSNR/SSIM on RGB [0..255]
    psnr_val = psnr(hr_c, sr_c, data_range=255)
    ssim_val = ssim(hr_c, sr_c, channel_axis=2, data_range=255)
    return psnr_val, ssim_val

def metrics_y(sr_rgb, hr_rgb, shave=0):
    if shave > 0:
        sr_c, hr_c = shave_to_common(sr_rgb, hr_rgb, shave)
    else:
        sr_c, hr_c = sr_rgb, hr_rgb
    ys = rgb_to_y(sr_c)
    yh = rgb_to_y(hr_c)
    # many papers subtract 16 and use range 219; psnr implementation needs data_range
    psnr_val = psnr(yh, ys, data_range=235 - 16)
    ssim_val = ssim(yh, ys, data_range=235 - 16)
    return psnr_val, ssim_val

# -------------------------
# Main loop
# -------------------------
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out_sr      = os.path.join(OUTPUT_ROOT, "sr")
    out_abs     = os.path.join(OUTPUT_ROOT, "heat_abs")
    out_sq      = os.path.join(OUTPUT_ROOT, "heat_sq")
    out_panel   = os.path.join(OUTPUT_ROOT, "panel")
    for d in [out_sr, out_abs, out_sq, out_panel]:
        os.makedirs(d, exist_ok=True)

    # Load model
    model, device, arch_cfg = load_model(CKPT_PATH)
    print("Loaded EDSR with config:", arch_cfg)

    # Gather image pairs by basename (case-insensitive, ignore extensions)
    def index_folder(folder):
        idx = {}
        for p in glob.glob(os.path.join(folder, "*")):
            if os.path.isdir(p):
                continue
            base = os.path.splitext(os.path.basename(p))[0].lower()
            idx[base] = p
        return idx

    lr_map = index_folder(LR_FOLDER)
    hr_map = index_folder(HR_FOLDER)
    keys = sorted(set(lr_map.keys()) & set(hr_map.keys()))
    if not keys:
        raise RuntimeError("No matching LR/HR pairs by basename. Check your folders.")

    rows = []
    for k in keys:
        lr_path = lr_map[k]
        hr_path = hr_map[k]
        try:
            lr = imread_rgb(lr_path)
            hr = imread_rgb(hr_path)
        except Exception as e:
            print(f"[skip] {k}: {e}")
            continue

        # Run model
        with torch.no_grad():
            x = to_tensor(lr).to(device)
            # Some EDSR variants expect input divisible by scale; pad if needed
            _, _, H, W = x.shape
            s = arch_cfg.get("upscale", UPSCALE)
            pad_h = (math.ceil(H / 1) - H) % 1  # usually 0 for EDSR
            pad_w = (math.ceil(W / 1) - W) % 1  # (kept for completeness)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            y = model(x)
            sr = to_uint8_rgb(y)

        # If size mismatch to HR, center-crop to common size for fair comps
        Hc = min(sr.shape[0], hr.shape[0])
        Wc = min(sr.shape[1], hr.shape[1])
        def center_crop_to(img, Hc, Wc):
            top = (img.shape[0] - Hc) // 2
            left = (img.shape[1] - Wc) // 2
            return img[top:top+Hc, left:left+Wc]
        sr_c = center_crop_to(sr, Hc, Wc)
        hr_c = center_crop_to(hr, Hc, Wc)

        # Heatmaps
        heat_abs = error_heatmap_abs(sr_c, hr_c)
        heat_sq  = error_heatmap_sq(sr_c, hr_c)

        # Metrics
        psnr_rgb, ssim_rgb = metrics_rgb(sr_c, hr_c, shave=SHAVE)
        psnr_y,   ssim_y   = metrics_y(sr_c, hr_c, shave=SHAVE)

        # Save files
        base = os.path.splitext(os.path.basename(hr_path))[0]
        cv2.imwrite(os.path.join(out_sr,    f"{base}_SR.png"), cv2.cvtColor(sr_c, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_abs,   f"{base}_ABS.png"), cv2.cvtColor(heat_abs, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_sq,    f"{base}_SQ.png"),  cv2.cvtColor(heat_sq,  cv2.COLOR_RGB2BGR))

        # Side-by-side panel: HR | SR | |SR-HR| | (SR-HR)^2
        panel = tile_h([
            annotate(hr_c,  "HR"),
            annotate(sr_c,  f"EDSR x{arch_cfg.get('upscale', UPSCALE)}"),
            annotate(heat_abs, "|SR-HR| heat"),
            annotate(heat_sq,  "Squared error heat"),
        ])
        cv2.imwrite(os.path.join(out_panel, f"{base}_panel.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        rows.append(dict(
            image=base,
            H=sr_c.shape[0], W=sr_c.shape[1],
            psnr_rgb=psnr_rgb, ssim_rgb=ssim_rgb,
            psnr_y=psnr_y, ssim_y=ssim_y
        ))
        print(f"[ok] {base}: PSNR_Y={psnr_y:.3f}, SSIM_Y={ssim_y:.4f}")

    # Save CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTPUT_ROOT, "metrics.csv"), index=False)
        print("Saved:", os.path.join(OUTPUT_ROOT, "metrics.csv"))
    else:
        print("No images processed.")

if __name__ == "__main__":
    main()
