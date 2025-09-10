# gradcam_v2.py
import os
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# CONFIG (your values)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_LR_IMAGE = "datasets/Test/Urban100/LRx4/img001.png"     # path to LR input
OPTIONAL_HR_GT  = "datasets/Test/Urban100/HR/img001.png"      # path to HR GT, or None
CHECKPOINT_PATH = "experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models/net_g_latest.pth"
OUTPUT_DIR      = "gradcam_v2"

EDSR_PARAMS = dict(
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

# If your EDSR class isn't at basicsr.archs.edsr_arch, set a custom import like:
# MODEL_IMPORT_PATH = ("path.to.module", "EDSR")
MODEL_IMPORT_PATH: Optional[Tuple[str, str]] = None


# =========================
# Utilities
# =========================
def _import_edsr(import_path: Optional[Tuple[str, str]]):
    """Import the EDSR class."""
    if import_path is not None:
        mod, cls = import_path
        m = __import__(mod, fromlist=[cls])
        return getattr(m, cls)

    # Try the standard BasicSR path
    try:
        from basicsr.archs.edsr_arch import EDSR as EDSR_from_basicsr
        return EDSR_from_basicsr
    except Exception:
        pass

    # Fallback: if EDSR is in the current project as EDSR.py
    try:
        from EDSR import EDSR as EDSR_local
        return EDSR_local
    except Exception as e:
        raise ImportError(
            "Could not import EDSR. Set MODEL_IMPORT_PATH properly or ensure "
            "basicsr.archs.edsr_arch.EDSR is available."
        ) from e


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # (H,W,3) -> (1,3,H,W) in [0,1]
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    # (1,3,H,W) or (3,H,W) -> (H,W,3) uint8
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def colorize_and_overlay(cam: np.ndarray, base_img_uint8: np.ndarray, alpha: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """
    cam: (H,W) in [0,1]; base_img_uint8: (H,W,3) uint8
    returns: (heatmap_uint8, overlay_uint8)
    """
    cmap = plt.get_cmap("jet")
    cam_color = cmap(cam)[:, :, :3]  # float [0,1]
    heat_uint8 = (cam_color * 255.0 + 0.5).astype(np.uint8)
    overlay = (alpha * cam_color + (1 - alpha) * (base_img_uint8.astype(np.float32) / 255.0))
    overlay_uint8 = (overlay * 255.0 + 0.5).astype(np.uint8)
    return heat_uint8, overlay_uint8


# =========================
# Grad-CAM (per BasicBlock)
# =========================
class GradCAMPerBlock:
    """
    Registers forward activations and backward gradients on every BasicBlock in model.body.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self.gradients: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.blocks = self._find_basic_blocks()

        for idx, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_hook(self._make_fwd_hook(idx)))
            self.handles.append(block.register_full_backward_hook(self._make_bwd_hook(idx)))

    def _find_basic_blocks(self) -> List[nn.Module]:
        body = getattr(self.model, "body", None)
        if body is None:
            raise RuntimeError("Model has no attribute 'body'. Can't locate BasicBlocks.")

        # Prefer to match by class name "BasicBlock"
        blocks = [m for m in body.modules() if m.__class__.__name__.lower() == "basicblock"]

        # If not found (e.g., body is a flat nn.Sequential of blocks), fall back to direct children
        if not blocks and isinstance(body, nn.Sequential):
            blocks = list(body)

        if not blocks:
            raise RuntimeError("Could not find BasicBlock modules inside model.body.")
        return blocks

    def _make_fwd_hook(self, idx: int):
        def fwd_hook(module, inp, out):
            self.activations[idx] = out
        return fwd_hook

    def _make_bwd_hook(self, idx: int):
        def bwd_hook(module, grad_input, grad_output):
            self.gradients[idx] = grad_output[0]
        return bwd_hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def compute_cam(self, idx: int, up_to_size: Tuple[int, int]) -> torch.Tensor:
        """
        Return (1,1,H,W) Grad-CAM in [0,1], upsampled to up_to_size.
        """
        if idx not in self.activations or idx not in self.gradients:
            raise RuntimeError(f"Missing activations/gradients for block {idx}. "
                               f"Did you register hooks before forward and call backward?")
        A = self.activations[idx]        # (N,C,h,w)
        dA = self.gradients[idx]         # (N,C,h,w)

        weights = dA.mean(dim=[2, 3], keepdim=True)     # (N,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)    # (N,1,h,w)
        cam = F.relu(cam)

        # Normalize each sample and upsample
        cams = []
        for n in range(cam.size(0)):
            c = cam[n:n+1]
            c_min, c_max = c.min(), c.max()
            if (c_max - c_min) > 1e-12:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = torch.zeros_like(c)
            c = F.interpolate(c, size=up_to_size, mode="bilinear", align_corners=False)
            cams.append(c)
        return torch.cat(cams, dim=0)    # (N,1,H,W)


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    # 1) Load model
    EDSR = _import_edsr(MODEL_IMPORT_PATH)
    model = EDSR(**EDSR_PARAMS).to(DEVICE).eval()

    # 2) Load checkpoint (robust to various state_dict layouts)
    if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state = ckpt.get("params_ema", ckpt.get("state_dict", ckpt))
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "net_g" in state:
                model.load_state_dict(state["net_g"], strict=False)
            else:
                model.load_state_dict(state, strict=False)

    # 3) Prepare input (EDSR internally normalizes using img_range/mean)
    lr_img = load_image_rgb(INPUT_LR_IMAGE)
    x = pil_to_tensor(lr_img).to(DEVICE)  # (1,3,H,W) in [0,1]

    # 4) Register Grad-CAM hooks BEFORE forward
    cammer = GradCAMPerBlock(model)

    # 5) Forward + L1 loss target
    with torch.set_grad_enabled(True):
        y = model(x)  # (1,3,H_hr,W_hr)

        if OPTIONAL_HR_GT and os.path.isfile(OPTIONAL_HR_GT):
            gt_img = load_image_rgb(OPTIONAL_HR_GT)
            gt = pil_to_tensor(gt_img).to(DEVICE)
            if gt.shape[-2:] != y.shape[-2:]:
                gt = F.interpolate(gt, size=y.shape[-2:], mode="bicubic", align_corners=False)
            loss = F.l1_loss(y, gt)
        else:
            # fallback target so grads exist
            loss = y.abs().mean()

        model.zero_grad(set_to_none=True)
        loss.backward()

    # 6) Save SR output
    sr_uint8 = tensor_to_uint8_img(y)
    Image.fromarray(sr_uint8).save(os.path.join(OUTPUT_DIR, "sr_output.png"))

    # 7) Compute & save per-block CAMs
    H_hr, W_hr = sr_uint8.shape[:2]
    num_blocks = len(cammer.blocks)
    print(f"Found {num_blocks} BasicBlocks. Saving Grad-CAMs into: {OUTPUT_DIR}")

    for i in range(num_blocks):
        cam = cammer.compute_cam(i, up_to_size=(H_hr, W_hr))  # (1,1,H,W)
        cam_np = cam[0, 0].detach().cpu().numpy()             # (H,W) in [0,1]

        heat_u8, overlay_u8 = colorize_and_overlay(cam_np, sr_uint8, alpha=0.35)

        heat_path = os.path.join(OUTPUT_DIR, f"block_{i+1:02d}_heatmap.png")
        over_path = os.path.join(OUTPUT_DIR, f"block_{i+1:02d}_overlay.png")
        Image.fromarray(heat_u8).save(heat_path)
        Image.fromarray(overlay_u8).save(over_path)
        print(f"[Block {i+1:02d}] -> {heat_path}, {over_path}")

    # 8) Clean up hooks
    cammer.remove()
    print("Done.")

if __name__ == "__main__":
    main()
