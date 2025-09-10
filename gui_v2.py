import os
import time
import cv2
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from basicsr.archs.edsr_arch import EDSR
from basicsr.utils import tensor2img

# -------------------- Config --------------------
UPSCALE = 4  # keep in sync with model.upscale
SAVE_BASE = "runs"

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Model ---------------------
model = EDSR(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=192,
    num_block=6,
    upscale=UPSCALE,
    num_heads=6,
    hw=8,
    ww=8,
    img_range=255.,
    rgb_mean=[0.4488, 0.4371, 0.4040]
).to(device)

checkpoint = torch.load(
    'experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models/net_g_latest.pth',
    map_location=device
)
model.load_state_dict(checkpoint['params'])
model.eval()

# -------------------- Utils ---------------------
def make_run_dir(base=SAVE_BASE):
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def norm01(x: torch.Tensor):
    mn = x.min()
    mx = x.max()
    if float(mx - mn) < 1e-8:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)

def tensor_to_rgb_np(t: torch.Tensor) -> np.ndarray:
    """
    t: [1,C,H,W] -> HxWx3 uint8 RGB (first 3 channels, per-channel min-max normalized).
    """
    assert t.dim() == 4 and t.size(0) == 1
    t = t.detach().cpu().float()[0]  # [C,H,W]
    C, H, W = t.shape
    if C >= 3:
        r, g, b = norm01(t[0]), norm01(t[1]), norm01(t[2])
        img = torch.stack([r, g, b], dim=0)
    else:
        g = norm01(t[0])
        img = torch.stack([g, g, g], dim=0)
    img = (img.clamp(0,1) * 255).byte().permute(1,2,0).numpy()  # HxWx3 RGB
    return img

def save_rgb(path, rgb_np):
    # cv2.imwrite expects BGR
    cv2.imwrite(path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

def overlay_heatmap_on_image(heatmap_01: np.ndarray, base_rgb: np.ndarray, alpha=0.45):
    """
    heatmap_01: HxW float in [0,1]
    base_rgb:  HxWx3 uint8 RGB
    """
    heat_uint8 = (heatmap_01 * 255).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    blend = (alpha * heat_color_rgb + (1 - alpha) * base_rgb).clip(0, 255).astype(np.uint8)
    return blend

# -------------------- Grad-CAM Hooks ---------------------
def register_gradcam_hooks(model, activations, gradients):
    """
    Forward hooks store activations; we also attach a grad hook on the activation tensor.
    """
    handles = []

    def fwd_hook(name):
        def _hook(module, inp, out):
            activations[name] = out
            def _save_grad(grad):
                gradients[name] = grad
            out.register_hook(_save_grad)
        return _hook

    # conv_first
    if hasattr(model, "conv_first"):
        handles.append(model.conv_first.register_forward_hook(fwd_hook("01_conv_first")))

    # body blocks (Sequential)
    if hasattr(model, "body"):
        for idx, block in enumerate(model.body):
            handles.append(block.register_forward_hook(fwd_hook(f"02_body_block_{idx:03d}")))

    # conv_after_body
    if hasattr(model, "conv_after_body"):
        handles.append(model.conv_after_body.register_forward_hook(fwd_hook("03_conv_after_body")))

    # upsample (whole module)
    if hasattr(model, "upsample"):
        handles.append(model.upsample.register_forward_hook(fwd_hook("04_upsample")))

    # conv_last
    if hasattr(model, "conv_last"):
        handles.append(model.conv_last.register_forward_hook(fwd_hook("05_conv_last")))

    return handles

def compute_gradcam_map(act: torch.Tensor, grad: torch.Tensor):
    """
    act:  [1,C,H,W] activation
    grad: [1,C,H,W] gradient d(L1)/d(act)
    Returns [H,W] in [0,1]
    """
    # GAP over spatial dims -> weights for each channel
    weights = grad.mean(dim=(2,3), keepdim=True)  # [1,C,1,1]
    cam = (weights * act).sum(dim=1, keepdim=False)  # [1,H,W] -> sum over C
    cam = cam.squeeze(0)  # [H,W]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam  # [H,W] 0..1

# -------------------- Inference (with L1-based Grad-CAM) ---------------------
def inference(img_bgr):
    run_dir = make_run_dir()

    # Save input
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    save_rgb(os.path.join(run_dir, "00_input.png"), img_rgb)

    # To tensor [1,3,H,W] in [0,1]
    inp = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

    # Register hooks
    activations, gradients = {}, {}
    handles = register_gradcam_hooks(model, activations, gradients)

    # Forward pass (with grads enabled)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(inp)  # [1,3,H_out,W_out]

    # Build L1 target: bicubic upsample of input to SR size (detached)
    _, _, H_out, W_out = out.shape
    target = F.interpolate(inp, size=(H_out, W_out), mode='bicubic', align_corners=False).detach()

    # Save bicubic target for reference
    target_img = tensor2img(target.detach().cpu())  # RGB uint8
    save_rgb(os.path.join(run_dir, "01_bicubic_target.png"), target_img)

    # L1 loss objective and backward
    loss = F.l1_loss(out, target, reduction='mean')
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Remove hooks
    for h in handles:
        h.remove()

    # Save per-layer feature and Grad-CAM (heatmap + overlay on SR output)
    # Prepare SR output as overlay base
    out_img = tensor2img(out.detach().cpu())  # RGB uint8
    save_rgb(os.path.join(run_dir, "ZZ_output.png"), out_img)

    # Also keep a numpy version and its size
    sr_base = out_img  # H_out x W_out x 3 (uint8)

    for name in sorted(activations.keys()):
        act = activations[name]                 # [1,C,h,w]
        grad = gradients.get(name, None)        # may be None
        # Feature visualization (first 3 channels)
        try:
            feat_img = tensor_to_rgb_np(act)
            save_rgb(os.path.join(run_dir, f"{name}.png"), feat_img)
        except Exception:
            pass

        if grad is not None:
            try:
                cam = compute_gradcam_map(act, grad)    # [h,w] 0..1 (torch)
                cam_np = cam.detach().cpu().numpy()
                # Resize CAM to SR output size for overlay
                cam_np_resized = cv2.resize(cam_np, (W_out, H_out), interpolation=cv2.INTER_CUBIC)

                # # Save raw heatmap (grayscale 0..255 as RGB)
                # heat_rgb = (cam_np_resized * 255).astype(np.uint8)
                # heat_rgb = np.stack([heat_rgb, heat_rgb, heat_rgb], axis=-1)
                # save_rgb(os.path.join(run_dir, f"{name}_gradcam_heatmap.png"), heat_rgb)

                # Save colored heatmap using a colormap
                heat_uint8 = (cam_np_resized * 255).astype(np.uint8)  # HxW uint8
                # Use TURBO if available (more perceptually uniform); fall back to JET
                colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
                heat_color_bgr = cv2.applyColorMap(heat_uint8, colormap)        # HxWx3 BGR
                heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
                save_rgb(os.path.join(run_dir, f"{name}_gradcam_heatmap.png"), heat_color_rgb)


                # Save overlay on SR output
                overlay = overlay_heatmap_on_image(cam_np_resized, sr_base, alpha=0.45)
                save_rgb(os.path.join(run_dir, f"{name}_gradcam_overlay.png"), overlay)
            except Exception:
                pass

    # Return SR result to UI
    return out_img

# -------------------- UI ---------------------
gr.Interface(
    fn=inference,
    inputs=gr.Image(type="numpy"),
    outputs="image",
    title="BasicSR Model Tester (GPU) — Grad-CAM (L1 loss) after each layer"
).launch()
