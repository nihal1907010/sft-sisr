
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # BasicSR registry
    from basicsr.utils.registry import LOSS_REGISTRY
except Exception:
    # Minimal fallback registry to keep importable outside BasicSR for testing
    class _Reg(dict):
        def register(self, cls):
            self[cls.__name__] = cls
            return cls
    LOSS_REGISTRY = _Reg()

# ========== Utilities ==========

def to_y_channel(x):
    """Convert RGB (in [0,1]) to Y (luminance) with BT.601 coefficients.
    If input is already single-channel, return as-is.
    x: (N,C,H,W) with C=1 or 3.
    """
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1, ...], x[:, 1:2, ...], x[:, 2:2+1, ...]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

# ========== Charbonnier Loss ==========

class CharbonnierLoss(nn.Module):
    """Robust L1 (Charbonnier) loss"""
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = torch.sqrt((pred - target) ** 2 + self.eps)
        if self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        else:
            return diff

# ========== SSIM Loss (single-scale) ==========

def _gaussian_kernel(window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    win2d = g.T @ g
    win2d = win2d / win2d.sum()
    return win2d

def ssim_map(x, y, window_size=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """Compute SSIM map for images in [0,1]. Assumes x,y in NCHW with same shape."""
    N, C, H, W = x.shape
    win = _gaussian_kernel(window_size, sigma, device=x.device, dtype=x.dtype).expand(C, 1, -1, -1)
    padding = window_size // 2

    mu_x = F.conv2d(x, win, padding=padding, groups=C)
    mu_y = F.conv2d(y, win, padding=padding, groups=C)
    mu_x2, mu_y2 = mu_x * mu_x, mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, win, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=padding, groups=C) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return num / (den + 1e-12)

class SSIMLoss(nn.Module):
    """1 - SSIM on luminance (optional) or on full channels"""
    def __init__(self, window_size=11, sigma=1.5, on_y=True):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.on_y = on_y

    def forward(self, pred, target):
        x, y = pred, target
        if self.on_y:
            x, y = to_y_channel(x), to_y_channel(y)
        ssim = ssim_map(x, y, self.window_size, self.sigma).mean()
        return 1.0 - ssim

# ========== Wavelet High-Frequency Loss ==========

# Expect user's helper to exist in their project:
# from v201 import swt_forward_haar2d
# We'll import lazily inside forward to avoid hard dependency at import-time.
class WaveletHFLoss(nn.Module):
    """L1/L2 loss on LH/HL/HH subbands from stationary wavelet transform (Haar)."""
    def __init__(self, weights=(1.0, 1.0, 1.0), p=1):
        super().__init__()
        assert len(weights) == 3, "weights must be (LH, HL, HH)"
        self.w = weights
        self.p = p

    def forward(self, pred, target):
        try:
            from basicsr.archs.v201 import swt_forward_haar2d  # user-provided
        except Exception as e:
            raise ImportError(
                "WaveletHFLoss requires swt_forward_haar2d in v201.py. "
                "Please make sure it is importable."
            ) from e

        _pll, plh, phl, phh = swt_forward_haar2d(pred)
        _tll, tlh, thl, thh = swt_forward_haar2d(target)

        losses = []
        if self.w[0] != 0:
            losses.append(self.w[0] * (plh - tlh).abs().pow(self.p).mean())
        if self.w[1] != 0:
            losses.append(self.w[1] * (phl - thl).abs().pow(self.p).mean())
        if self.w[2] != 0:
            losses.append(self.w[2] * (phh - thh).abs().pow(self.p).mean())
        return sum(losses) if len(losses) > 0 else pred.new_tensor(0.0)

# ========== Perceptual Loss (VGG features) ==========

class VGGPerceptualLoss(nn.Module):
    """Simple VGG feature L1 loss.
    Note: For training you likely want pretrained weights. Set use_pretrained=True
    and ensure torchvision can load them in your environment.
    """
    def __init__(self, layer="relu3_1", use_pretrained=True, input_range=(0.0, 1.0), normalize=True):
        super().__init__()
        try:
            from torchvision import models
        except Exception as e:
            raise ImportError("torchvision is required for VGGPerceptualLoss") from e

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT if use_pretrained else None).features

        layer_map = {
            "relu1_1": 1,  "relu1_2": 3,
            "relu2_1": 6,  "relu2_2": 8,
            "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
            "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
            "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
        }
        assert layer in layer_map, f"Unsupported VGG layer: {layer}"
        self.vgg = vgg[: layer_map[layer] + 1].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.input_range = input_range
        self.normalize = normalize

    def _preprocess(self, x):
        # Ensure 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # Map from input_range to [0,1]
        lo, hi = self.input_range
        if not (abs(lo) == 0.0 and abs(hi - 1.0) < 1e-6):
            x = (x - lo) / (hi - lo + 1e-12)
            x = torch.clamp(x, 0.0, 1.0)
        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def _features(self, x):
        return self.vgg(self._preprocess(x))

    def forward(self, pred, target):
        f_p = self._features(pred)
        f_t = self._features(target)
        return F.l1_loss(f_p, f_t)

# ========== Total Variation (TV) ==========

def tv_loss(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw

# ========== Combined Loss ==========

@LOSS_REGISTRY.register()
class CombinedSRLoss(nn.Module):
    """Combined loss for super-resolution usable as BasicSR pixel_opt.
    Components:
      - Charbonnier     (w_charb)
      - SSIM (1-SSIM)   (w_ssim)
      - Wavelet HF      (w_wavelet)
      - Perceptual VGG  (w_perceptual)
      - Total Variation (w_tv)

    Example BasicSR YAML (put under train.pixel_opt):
      pixel_opt:
        type: CombinedSRLoss
        w_charb: 1.0
        w_ssim: 0.2
        w_wavelet: 0.5
        w_perceptual: 0.01
        w_tv: 1.0e-6
        wavelet_weights: [1.0, 1.0, 1.0]
        ssim_on_y: true
        vgg_layer: relu3_1
        vgg_use_pretrained: true
        vgg_input_range: [0.0, 1.0]
        vgg_normalize: true
    """
    def __init__(self,
                 w_charb=1.0,
                 w_ssim=0.2,
                 w_wavelet=0.5,
                 w_perceptual=0.01,
                 w_tv=1e-6,
                 # SSIM
                 ssim_on_y=True,
                 ssim_window_size=11,
                 ssim_sigma=1.5,
                 # Wavelet
                 wavelet_weights=(1.0, 1.0, 1.0),
                 wavelet_p=1,
                 # Perceptual
                 vgg_layer="relu3_1",
                 vgg_use_pretrained=True,
                 vgg_input_range=(0.0, 1.0),
                 vgg_normalize=True):
        super().__init__()

        self.w_charb = float(w_charb)
        self.w_ssim = float(w_ssim)
        self.w_wavelet = float(w_wavelet)
        self.w_perc = float(w_perceptual)
        self.w_tv = float(w_tv)

        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss(window_size=ssim_window_size, sigma=ssim_sigma, on_y=bool(ssim_on_y))
        self.wav = WaveletHFLoss(weights=tuple(wavelet_weights), p=int(wavelet_p))
        self.perc = VGGPerceptualLoss(layer=vgg_layer,
                                      use_pretrained=bool(vgg_use_pretrained),
                                      input_range=tuple(vgg_input_range),
                                      normalize=bool(vgg_normalize))

    def forward(self, pred, target):
        losses = {}
        total = pred.new_tensor(0.0)

        if self.w_charb != 0:
            Lc = self.charb(pred, target)
            losses['charb'] = Lc.detach()
            total = total + self.w_charb * Lc

        if self.w_ssim != 0:
            Ls = self.ssim(pred, target)
            losses['ssim'] = Ls.detach()
            total = total + self.w_ssim * Ls

        if self.w_wavelet != 0:
            Lw = self.wav(pred, target)
            losses['waveletHF'] = Lw.detach()
            total = total + self.w_wavelet * Lw

        if self.w_perc != 0:
            Lp = self.perc(pred, target)
            losses['perceptual'] = Lp.detach()
            total = total + self.w_perc * Lp

        if self.w_tv != 0:
            Lt = tv_loss(pred)
            losses['tv'] = Lt.detach()
            total = total + self.w_tv * Lt

        # You can log `losses` dict in your training loop if your runner supports it.
        return total
