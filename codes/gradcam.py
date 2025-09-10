import os
import glob
import cv2
import torch
import numpy as np
from importlib import import_module
from torchvision import transforms

# -------------------
# Grad-CAM
# -------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None

        def fwd_hook(_m, _in, out):
            self.activations = out.detach()

        def bwd_hook(_m, _gin, gout):
            self.gradients = gout[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        if hasattr(target_layer, "register_full_backward_hook"):
            target_layer.register_full_backward_hook(bwd_hook)
        else:  # very old PyTorch fallback
            target_layer.register_backward_hook(bwd_hook)

    def generate(self, input_tensor, scalar_objective=None):
        output = self.model(input_tensor)
        if scalar_objective is None:
            scalar_objective = output.mean()  # image regression -> mean works
        self.model.zero_grad(set_to_none=True)
        scalar_objective.backward(retain_graph=False)

        # weights: GAP over gradients
        w = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [N,C,1,1]
        cam = torch.sum(w * self.activations, dim=1)              # [N,H,W]
        cam = torch.relu(cam)

        # normalize per-sample to [0,1]
        cams = []
        for i in range(cam.shape[0]):
            c = cam[i]
            c = c - c.min()
            c = c / (c.max() + 1e-8)
            cams.append(c)
        cam = torch.stack(cams, dim=0)
        return output, cam

# -------------------
# IO helpers
# -------------------
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
to_tensor = transforms.ToTensor()

def safe_imread_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def overlay_cam_rgb(image_rgb: np.ndarray, cam_01: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam_01, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255.0 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = np.float32(heatmap_rgb) / 255.0
    base = np.float32(image_rgb) / 255.0
    out = (heatmap_rgb + base)
    out = out / (out.max() + 1e-8)
    return np.uint8(255.0 * out)

def list_images(folder: str):
    files = []
    for ext in VALID_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files.sort()
    return files

# -------------------
# Non-inplace forward patch for EDSR
# -------------------
def patch_forward_no_inplace(model):
    """
    Replaces the model.forward with a copy that avoids in-place ops.
    Assumes the EDSR structure you've shown: conv_first -> body -> conv_after_body
    -> Upsample -> conv_last with mean/range normalization.
    """
    # Capture the submodules used in forward
    conv_first = model.conv_first
    body = model.body
    conv_after_body = model.conv_after_body
    upsample = model.upsample
    conv_last = model.conv_last
    img_range = model.img_range
    # model.mean exists and is reshaped/type_as(x) in original forward;
    # we’ll do the same here (no in-place math).

    def forward_no_inplace(x):
        mean = model.mean.type_as(x)
        x = (x - mean) * img_range
        x1 = conv_first(x)
        res = conv_after_body(body(x1))
        res = res + x1  # <-- non-inplace residual add
        out = conv_last(upsample(res))
        out = out / img_range + mean
        return out

    # Bind the method
    model.forward = forward_no_inplace

# -------------------
# Runner
# -------------------
def run_gradcam(img_folder, save_folder, checkpoint_path,
                arch_import="basicsr.archs.edsr_arch", device="cuda"):

    os.makedirs(save_folder, exist_ok=True)
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Import EDSR from your module
    arch_mod = import_module(arch_import)
    EDSR = getattr(arch_mod, "EDSR")

    # Build model with your params
    model = EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=192,
        num_block=6,
        upscale=4,
        num_heads=6,
        hw=8,
        ww=8,
        img_range=255.,
        rgb_mean=[0.4488, 0.4371, 0.4040]
    ).to(device).eval()

    # Load checkpoint robustly
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("params") or ckpt.get("params_ema") or ckpt
    model.load_state_dict(state, strict=False)

    # ---- Patch forward to avoid in-place residual add
    patch_forward_no_inplace(model)

    # Target layer for CAM (post-body conv is a good choice)
    target_layer = model.conv_after_body
    cam_engine = GradCAM(model, target_layer)

    # Process folder
    files = list_images(img_folder)
    if not files:
        print(f"No images found in: {img_folder}")
        return

    for path in files:
        fname = os.path.basename(path)
        img = safe_imread_rgb(path)
        tensor = to_tensor(img).unsqueeze(0).to(device)

        # Generate CAM
        _out, cam = cam_engine.generate(tensor)
        cam_map = cam[0].detach().cpu().numpy()

        # Save overlay
        overlay = overlay_cam_rgb(img, cam_map)
        save_path = os.path.join(save_folder, fname)
        cv2.imwrite(save_path, overlay[:, :, ::-1])  # back to BGR for OpenCV
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # ==== Set your paths here ====
    img_folder = "datasets/Test/Set5/LRx4"
    save_folder = "data/Set5/heatmap"
    checkpoint_path = "experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models/net_g_latest.pth"
    # =============================

    run_gradcam(img_folder, save_folder, checkpoint_path)
