import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram



# ======================= Frequency-aware composite loss =======================
# L = L_pix + L_wav + 0.1 * L_grad + 0.05 * L_fft + 0.1 * L_cyc
# ------------------------------------------------------------------------------

import math
try:
    from pytorch_wavelets import DWTForward
except Exception as _e:
    DWTForward = None  # we will error out cleanly in __init__

def _charbonnier(x, eps: float = 1e-3):
    return torch.sqrt(x * x + eps * eps)

def _sobel_grad(img: torch.Tensor) -> torch.Tensor:
    # img: (b, c, h, w)
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    kx = kx.repeat(img.size(1), 1, 1, 1)
    ky = ky.repeat(img.size(1), 1, 1, 1)
    gx = F.conv2d(img, kx, padding=1, groups=img.size(1))
    gy = F.conv2d(img, ky, padding=1, groups=img.size(1))
    return torch.sqrt(gx * gx + gy * gy + 1e-12)

def _fft_log_mag(x: torch.Tensor) -> torch.Tensor:
    # x: (b, c, h, w) -> (b, c, h, w//2+1)
    X = torch.fft.rfftn(x, dim=(-2, -1), norm='backward')
    mag = torch.abs(X)
    return torch.log(mag + 1e-8)

def _radial_weight(h: int, w: int, device, alpha: float = 1.0) -> torch.Tensor:
    # produces (1,1,h, w//2+1)
    yy = torch.arange(h, device=device).float()
    xx = torch.arange(w // 2 + 1, device=device).float()
    # torch>=1.10 supports indexing='ij'; if not available, default behavior is 'ij' for 2 tensors
    try:
        Y, X = torch.meshgrid(yy, xx, indexing='ij')
    except TypeError:
        Y, X = torch.meshgrid(yy, xx)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    rr = rr / (rr.max() + 1e-8)
    return (rr ** alpha).unsqueeze(0).unsqueeze(0)

@LOSS_REGISTRY.register()
class FrequencyAwareLoss(nn.Module):
    """
    Composite frequency-aware SR loss:

        L = L_pix + L_wav + w_grad * L_grad + w_fft * L_fft + w_cyc * L_cyc

    where
      - L_pix       : Charbonnier (robust L1) on RGB
      - L_wav       : Wavelet band losses: λ_LL * |LL̂-LL| + λ_H * (|LĤ-LH| + |HL̂-HL| + |HĤ-HH|)
      - L_grad      : L1 on Sobel gradient magnitude maps
      - L_fft       : L1 on log-magnitude FFT, radially weighted toward high-freq
      - L_cyc       : downsample(pred) vs LR input (bicubic by default)

    Args:
        loss_weight (float): global multiplier (BasicSR convention).
        wave (str)         : wavelet type for pytorch_wavelets (default: 'haar').
        charbonnier_eps    : epsilon for Charbonnier pixel loss.
        w_ll, w_h          : weights within L_wav (LL vs high-frequency).
        w_grad, w_fft      : weights for gradient and FFT terms.
        w_cyc              : weight for cycle consistency term.
        downscale (int)    : scale factor for L_cyc; must match your SR scale.
        cycle_mode (str)   : interpolation mode for downsampling ('bicubic' recommended).
    """
    def __init__(self,
                 loss_weight: float = 1.0,
                 wave: str = 'haar',
                 charbonnier_eps: float = 1e-3,
                 w_ll: float = 0.2,
                 w_h: float = 0.6,
                 w_grad: float = 0.1,
                 w_fft: float = 0.05,
                 w_cyc: float = 0.1,
                 downscale: int = 2,
                 cycle_mode: str = 'bicubic'):
        super().__init__()
        if DWTForward is None:
            raise ImportError(
                'pytorch_wavelets is required for FrequencyAwareLoss. '
                'Install with: pip install pytorch_wavelets'
            )
        self.loss_weight = loss_weight
        self.charbonnier_eps = charbonnier_eps
        self.w_ll = w_ll
        self.w_h  = w_h
        self.w_grad = w_grad
        self.w_fft  = w_fft
        self.w_cyc  = w_cyc
        self.downscale = downscale
        self.cycle_mode = cycle_mode

        # single-level DWT, symmetric padding (matches your block)
        self.dwt = DWTForward(J=1, wave=wave, mode='symmetric')

    # ---------------- components ----------------
    def _L_pix(self, pred, gt):
        return _charbonnier(pred - gt, self.charbonnier_eps).mean()

    def _L_wav(self, pred, gt):
        LLp, Hp = self.dwt(pred)     # Hp: list len=1 -> (b, c, 3, h/2, w/2)
        LLg, Hg = self.dwt(gt)
        Hp, Hg = Hp[0], Hg[0]
        Lll = F.l1_loss(LLp, LLg)
        Lh  = (F.l1_loss(Hp[:, :, 0], Hg[:, :, 0]) +
               F.l1_loss(Hp[:, :, 1], Hg[:, :, 1]) +
               F.l1_loss(Hp[:, :, 2], Hg[:, :, 2]))
        return self.w_ll * Lll + self.w_h * Lh

    def _L_grad(self, pred, gt):
        return F.l1_loss(_sobel_grad(pred), _sobel_grad(gt))

    def _L_fft(self, pred, gt):
        b, c, h, w = pred.shape
        W = _radial_weight(h, w, pred.device, alpha=1.0)  # emphasize high-freq
        L = 0.0
        P = _fft_log_mag(pred)
        G = _fft_log_mag(gt)
        # average over channels
        L = torch.mean(W * torch.abs(P - G))
        return L

    def _L_cyc(self, pred, lr):
        if (lr is None) or (self.w_cyc <= 0):
            return pred.new_zeros(())
        # downsample pred to LR size; use bicubic by default
        h, w = lr.shape[-2:]
        align = False if 'linear' in self.cycle_mode else None
        down = F.interpolate(pred, size=(h, w), mode=self.cycle_mode, align_corners=align)
        return F.l1_loss(down, lr)

    # --------------- BasicSR entrypoint ---------------
    def forward(self, pred, target, weight=None, **kwargs):
        """
        pred   : (N, C, H, W) SR prediction
        target : (N, C, H, W) HR GT
        weight : (unused; kept for BasicSR compatibility)
        kwargs : may contain 'lr' (or 'lq') for the cycle term
        """
        # fetch LR for cycle consistency if provided
        lr = kwargs.get('lr', kwargs.get('lq', None))

        L = self._L_pix(pred, target)
        L = L + self._L_wav(pred, target)

        if self.w_grad > 0:
            L = L + self.w_grad * self._L_grad(pred, target)
        if self.w_fft > 0:
            L = L + self.w_fft * self._L_fft(pred, target)
        if self.w_cyc > 0:
            L = L + self.w_cyc * self._L_cyc(pred, lr)

        return self.loss_weight * L

