import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sep_conv2d_same(x: torch.Tensor, krow: torch.Tensor, kcol: torch.Tensor,
                     dilation: int = 1, mode: str = "circular") -> torch.Tensor:
    """
    Separable 2D correlation (row then col) with 'same' output size,
    using circular padding to preserve shift-invariance and invertibility.
    """
    C = x.shape[1]
    # Horizontal pass (1 x K)
    kcol = kcol.view(1, 1, 1, -1).repeat(C, 1, 1, 1)          # (C,1,1,KW)
    xh = F.pad(x, (0, (kcol.shape[-1]-1)*dilation, 0, 0), mode=mode)
    xh = F.conv2d(xh, kcol, stride=1, padding=0, dilation=(1, dilation), groups=C)

    # Vertical pass (K x 1)
    krow = krow.view(1, 1, -1, 1).repeat(C, 1, 1, 1)          # (C,1,KH,1)
    xv = F.pad(xh, (0, 0, 0, (krow.shape[-2]-1)*dilation), mode=mode)
    xv = F.conv2d(xv, krow, stride=1, padding=0, dilation=(dilation, 1), groups=C)
    return xv


class SWTForward(nn.Module):
    """
    1-level stationary wavelet transform (SWT) using Haar filters.
    Outputs (LL, LH, HL, HH) with the same HxW as input.
    """
    def __init__(self, dilation: int = 1, padding_mode: str = "circular"):
        super().__init__()
        self.dilation = dilation
        self.mode = padding_mode
        s = math.sqrt(2.0)
        # Haar analysis filters
        lo = torch.tensor([1/s,  1/s], dtype=torch.float32)   # low-pass
        hi = torch.tensor([-1/s, 1/s], dtype=torch.float32)   # high-pass
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    @torch.no_grad()
    def _filters(self, x: torch.Tensor):
        # Put buffers on the correct device/dtype at runtime
        lo = self.lo.to(dtype=x.dtype, device=x.device)
        hi = self.hi.to(dtype=x.dtype, device=x.device)
        return lo, hi

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        returns: (LL, LH, HL, HH), each (B, C, H, W)
        """
        lo, hi = self._filters(x)
        LL = _sep_conv2d_same(x, lo, lo, dilation=self.dilation, mode=self.mode)
        LH = _sep_conv2d_same(x, lo, hi, dilation=self.dilation, mode=self.mode)
        HL = _sep_conv2d_same(x, hi, lo, dilation=self.dilation, mode=self.mode)
        HH = _sep_conv2d_same(x, hi, hi, dilation=self.dilation, mode=self.mode)
        return LL, LH, HL, HH


class SWTInverse(nn.Module):
    """
    Exact inverse for the 1-level SWT above (Haar).
    Takes (LL, LH, HL, HH) and reconstructs x.
    """
    def __init__(self, dilation: int = 1, padding_mode: str = "circular"):
        super().__init__()
        self.dilation = dilation
        self.mode = padding_mode
        s = math.sqrt(2.0)
        # Haar synthesis filters (time-reversed analysis)
        lor = torch.tensor([1/s,  1/s], dtype=torch.float32).flip(0)  # == [1/s,1/s]
        hir = torch.tensor([-1/s, 1/s], dtype=torch.float32).flip(0)  # == [1/s,-1/s]
        self.register_buffer("lor", lor)
        self.register_buffer("hir", hir)

    @torch.no_grad()
    def _filters(self, ref: torch.Tensor):
        lor = self.lor.to(dtype=ref.dtype, device=ref.device)
        hir = self.hir.to(dtype=ref.dtype, device=ref.device)
        return lor, hir

    def forward(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
        """
        LL, LH, HL, HH: each (B, C, H, W)
        returns: x_recon (B, C, H, W)
        """
        lor, hir = self._filters(LL)

        # Synthesis: sum of separable correlations with synthesis filters
        rec = (
            _sep_conv2d_same(LL, lor, lor, dilation=self.dilation, mode=self.mode) +
            _sep_conv2d_same(LH, lor, hir, dilation=self.dilation, mode=self.mode) +
            _sep_conv2d_same(HL, hir, lor, dilation=self.dilation, mode=self.mode) +
            _sep_conv2d_same(HH, hir, hir, dilation=self.dilation, mode=self.mode)
        )

        # With even-length (Haar) filters and 'circular' padding, the correlation
        # introduces a 1-pixel positive shift in both dims; compensate and scale.
        rec = torch.roll(rec, shifts=(1, 1), dims=(-2, -1))
        rec = 0.25 * rec  # normalization for Haar (sum of 4 branches)
        return rec


# # ---------- Quick test: reconstruction error ----------
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     B, C, H, W = 2, 3, 128, 128
#     x = torch.randn(B, C, H, W, device=device)

#     fwd = SWTForward(dilation=1, padding_mode="circular").to(device)
#     inv = SWTInverse(dilation=1, padding_mode="circular").to(device)

#     LL, LH, HL, HH = fwd(x)
#     x_rec = inv(LL, LH, HL, HH)

#     mse = (x - x_rec).pow(2).mean().item()
#     mae = (x - x_rec).abs().mean().item()
#     max_err = (x - x_rec).abs().max().item()
#     print(f"MSE: {mse:.3e} | MAE: {mae:.3e} | Max abs err: {max_err:.3e}")

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse

device = "cuda" if torch.cuda.is_available() else "cpu"

def imageTowindows(x, hw, ww): # b, c, h, w -> b', n, c
  b, c, h, w = x.shape
  x = x.reshape(b, c, h // hw, hw, w // ww, ww)
  x = x.permute(0, 2, 4, 1, 3, 5)
  x = x.reshape(-1, hw * ww, c)
  return x

def imageTowindows_bnc(x, h, w, hw, ww): # b, n, c -> b', n, c
  b, n, c = x.shape
  x = x.transpose(-1, -2) # b, c, n
  x = x.reshape(b, c, h, w) # b, c, h, w
  x = imageTowindows(x, hw, ww) # b', n, c
  return x

def windowsToImage(x, h, w, hw, ww): # b', n, c -> b, h, w, c
  b = int(x.shape[0] // (h * w / hw / ww))
  x = x.reshape(b, h // hw, w // ww, hw, ww, -1)
  x = x.permute(0, 1, 3, 2, 4, 5)
  x = x.reshape(b, h, w, -1)
  return x

class SpatialWindowSelfAttention(nn.Module):
  def __init__(self, dim, num_heads, hw, ww):
    super().__init__()

    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.hw = hw
    self.ww = ww

    self.wqkv = nn.Linear(dim, 3 * dim)
    self.wp = nn.Linear(dim, dim)

    self.relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * self.hw - 1) * (2 * self.ww - 1), self.num_heads)
    )

    coords_h = torch.arange(hw) # hw
    coords_w = torch.arange(ww) # ww
    coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, hw, ww
    coords_flatten = torch.flatten(coords, 1) # 2, hw * ww
    relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1) # 2, hw * ww, hw * ww
    relative_coords = relative_coords.permute(1, 2, 0) # hw * ww, hw * ww, 2
    relative_coords[:, :, 0] += hw - 1
    relative_coords[:, :, 1] += ww - 1
    relative_coords[:, :, 0] *= 2 * ww - 1
    relative_position_index = relative_coords.sum(-1) # hw * ww, hw * ww
    self.register_buffer('relative_position_index', relative_position_index)


  def forward(self, x, h, w): # batch, height * width, channel

    original_h, original_w = h, w

    pad_b = (self.hw - h % self.hw) % self.hw
    pad_r = (self.ww - w % self.ww) % self.ww

    h += pad_b
    w += pad_r

    b, _, c = x.shape # batch, height * width, channel

    x = x.reshape(b, original_h, original_w, c).permute(0, 3, 1, 2) # batch, channel, height, width
    x = F.pad(x, [0, pad_r, 0, pad_b])
    x = x.permute(0, 2, 3, 1).reshape(b, -1, c) # batch, height * width, channel

    qkv = self.wqkv(x) # batch, height * width, 3 * channel
    qkv = qkv.reshape(b, -1, 3, c) # batch, height * width, 3, channel
    qkv = qkv.permute(2, 0, 1, 3) # 3, batch, height * width, channel
    q, k, v = qkv[0], qkv[1], qkv[2] # batch, height * width, channel
    q, k, v = imageTowindows_bnc(q, h, w, self.hw, self.ww), imageTowindows_bnc(k, h, w, self.hw, self.ww), imageTowindows_bnc(v, h, w, self.hw, self.ww) # batch', height_win * width_win, channels
    q = q.reshape(-1, self.hw * self.ww, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch', num_heads, height_win * width_win, channel / num_heads
    k = k.reshape(-1, self.hw * self.ww, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch', num_heads, height_win * width_win, channel / num_heads
    v = v.reshape(-1, self.hw * self.ww, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch', num_heads, height_win * width_win, channel / num_heads
    attention_scores = torch.matmul(q, k.transpose(-1, -2)) # batch', num_heads, height_win * width_win, height_win * width_win
    scaled_scores = attention_scores / pow(self.head_dim, 0.5) # batch', num_heads, height_win * width_win, height_win * width_win

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)] # (hw * ww) * (hw * ww), num_heads
    relative_position_bias = relative_position_bias.reshape(self.hw * self.ww, self.hw * self.ww, -1) # hw * ww, hw * ww, num_heads
    relative_position_bias = relative_position_bias.permute(2, 0, 1) # num_heads, hw * ww, hw * ww
    scaled_scores += relative_position_bias.unsqueeze(0) # batch', num_heads, hw * ww, hw * ww

    attention_weights = F.softmax(scaled_scores , -1) # batch', num_heads, height_win * width_win, height_win * width_win
    y = torch.matmul(attention_weights , v) # batch', num_heads, height_win * width_win, channel / num_heads
    y = y.transpose(1, 2) # batch', height_win * width_win, num_heads, channel / num_heads
    y = y.reshape(-1, self.hw * self.ww, c) # batch', height_win * width_win, channel
    y = windowsToImage(y, h, w, self.hw, self.ww) # batch, height, width, channel
    out = self.wp(y) # batch, height, width, channel
    out = out[:, :original_h, :original_w, :]
    out = out.reshape(b, -1, c) # batch, height * width, channel
    return out

class SpatialWindowCrossAttention(nn.Module):
  def __init__(self, dim, kexpand, num_heads, hw, ww):
    super().__init__()

    self.kexpand = kexpand
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.hw = hw
    self.ww = ww

    self.qproj = nn.Linear(dim, dim)
    self.kproj = nn.Linear(self.kexpand * dim, dim)
    self.vproj = nn.Linear(dim, dim)
    self.wp = nn.Linear(dim, dim)

    self.relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * self.hw - 1) * (2 * self.ww - 1), self.num_heads)
    )

    coords_h = torch.arange(hw) # hw
    coords_w = torch.arange(ww) # ww
    coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, hw, ww
    coords_flatten = torch.flatten(coords, 1) # 2, hw * ww
    relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1) # 2, hw * ww, hw * ww
    relative_coords = relative_coords.permute(1, 2, 0) # hw * ww, hw * ww, 2
    relative_coords[:, :, 0] += hw - 1
    relative_coords[:, :, 1] += ww - 1
    relative_coords[:, :, 0] *= 2 * ww - 1
    relative_position_index = relative_coords.sum(-1) # hw * ww, hw * ww
    self.register_buffer('relative_position_index', relative_position_index)

  def forward(self, qq, kk, vv, h, w): # batch, height * width, channel

    original_h, original_w = h, w

    pad_b = (self.hw - h % self.hw) % self.hw
    pad_r = (self.ww - w % self.ww) % self.ww

    h += pad_b
    w += pad_r

    b, _, c = qq.shape # batch, height * width, channel

    qq = qq.reshape(b, original_h, original_w, c).permute(0, 3, 1, 2) # batch, channel, height, width
    qq = F.pad(qq, [0, pad_r, 0, pad_b])
    qq = qq.permute(0, 2, 3, 1).reshape(b, -1, c) # batch, height * width, channel

    kk = kk.reshape(b, original_h, original_w, self.kexpand * c).permute(0, 3, 1, 2) # batch, channel, height, width
    kk = F.pad(kk, [0, pad_r, 0, pad_b])
    kk = kk.permute(0, 2, 3, 1).reshape(b, -1, self.kexpand * c) # batch, height * width, channel

    vv = vv.reshape(b, original_h, original_w, c).permute(0, 3, 1, 2) # batch, channel, height, width
    vv = F.pad(vv, [0, pad_r, 0, pad_b])
    vv = vv.permute(0, 2, 3, 1).reshape(b, -1, c) # batch, height * width, channel


    q, k, v = self.qproj(qq), self.kproj(kk), self.vproj(vv) # batch, height * width, channel
    q, k, v = imageTowindows_bnc(q, h, w, self.hw, self.ww), imageTowindows_bnc(k, h, w, self.hw, self.ww), imageTowindows_bnc(v, h, w, self.hw, self.ww) # batch', height_win * width_win, channels
    q = q.reshape(-1, self.hw * self.ww, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch', num_heads, height_win * width_win, channel / num_heads
    k = k.reshape(-1, self.hw * self.ww, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch', num_heads, height_win * width_win, channel / num_heads
    v = v.reshape(-1, self.hw * self.ww, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch', num_heads, height_win * width_win, channel / num_heads
    attention_scores = torch.matmul(q, k.transpose(-1, -2)) # batch', num_heads, height_win * width_win, height_win * width_win
    scaled_scores = attention_scores / pow(self.head_dim, 0.5) # batch', num_heads, height_win * width_win, height_win * width_win

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)] # (hw * ww) * (hw * ww), num_heads
    relative_position_bias = relative_position_bias.reshape(self.hw * self.ww, self.hw * self.ww, -1) # hw * ww, hw * ww, num_heads
    relative_position_bias = relative_position_bias.permute(2, 0, 1) # num_heads, hw * ww, hw * ww
    scaled_scores += relative_position_bias.unsqueeze(0) # batch', num_heads, hw * ww, hw * ww

    attention_weights = F.softmax(scaled_scores , -1) # batch', num_heads, height_win * width_win, height_win * width_win
    y = torch.matmul(attention_weights , v) # batch', num_heads, height_win * width_win, channel / num_heads
    y = y.transpose(1, 2) # batch', height_win * width_win, num_heads, channel / num_heads
    y = y.reshape(-1, self.hw * self.ww, c) # batch', height_win * width_win, channel
    y = windowsToImage(y, h, w, self.hw, self.ww) # batch, heigh, width, channel
    out = self.wp(y) # batch, height, width, channel
    out = out[:, :original_h, :original_w, :]
    out = out.reshape(b, -1, c) # batch, height * width, channel
    return out

class ChannelSelfAttention(nn.Module):
  def __init__(self, dim, num_heads):
    super().__init__()

    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.wqkv = nn.Linear(dim, 3 * dim)
    self.wp = nn.Linear(dim, dim)

  def forward(self, x): # batch, height * width, channel
    b, n, c = x.shape
    qkv = self.wqkv(x) # batch, height * width, 3 * channel
    qkv = qkv.reshape(b, n, 3, c) # batch, height * width, 3, channel
    # qkv = qkv.reshape(3, b, n, c) # 3, batch, height * width, channel
    qkv = qkv.permute(2, 0, 1, 3) # 3, batch, height * width, channel
    q, k, v = qkv[0], qkv[1], qkv[2] # batch, height * width, channel
    q = q.reshape(b, n, self.num_heads, c // self.num_heads).reshape(b, self.num_heads, n, c // self.num_heads) # batch, num_heads, height * width, channel / num_heads
    k = k.reshape(b, n, self.num_heads, c // self.num_heads).reshape(b, self.num_heads, n, c // self.num_heads) # batch, num_heads, height * width, channel / num_heads
    v = v.reshape(b, n, self.num_heads, c // self.num_heads).reshape(b, self.num_heads, n, c // self.num_heads) # batch, num_heads, height * width, channel / num_heads
    attention_scores = torch.matmul(q.transpose(-1, -2), k) # batch, num_heads, channel / num_heads, channel / num_heads
    scaled_scores = attention_scores / pow(self.head_dim, 0.5) # batch, num_heads, channel / num_heads, channel / num_heads

    attention_weights = F.softmax(scaled_scores , -1) # batch, num_heads, channel / num_heads, channel / num_heads
    y = torch.matmul(attention_weights , v.transpose(-1, -2)) # batch, num_heads, channel / num_heads, height * width
    y = y.permute(0, 3, 1, 2) # batch, height * width, num_heads, channel / num_heads
    y = y.reshape(b, n, c) # batch, height * width, channel
    out = self.wp(y) # batch, height * width, channel
    return out

class MLP(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.hidden_dim = 4 * dim

    self.fc1 = nn.Linear(dim, self.hidden_dim)
    self.fc2 = nn.Linear(self.hidden_dim, dim)

  def forward(self, x):
    return self.fc2(F.gelu(self.fc1(x)))

class Encoder2(nn.Module):
  def __init__(self, dim, num_heads, hw, ww):
    super().__init__()

    self.layernorm1 = nn.LayerNorm(dim)
    self.layernorm2 = nn.LayerNorm(dim)
    self.layernorm3 = nn.LayerNorm(dim)

    self.spatialattention = SpatialWindowSelfAttention(dim, num_heads, hw, ww)
    self.channelattention = ChannelSelfAttention(dim, num_heads)
    self.mlp = MLP(dim)

  def forward(self, x):

    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(b, -1, c)

    x = x + self.spatialattention(self.layernorm1(x), h, w)
    x = x + self.channelattention(self.layernorm2(x))
    x = x + self.mlp(self.layernorm3(x))

    x = x.reshape(b, h, w, c)
    x = x.permute(0, 3, 1, 2)

    return x

class Encoder3(nn.Module):
  def __init__(self, dim, num_heads, hw, ww):
    super().__init__()

    self.layernorm1 = nn.LayerNorm(dim)
    self.layernorm2 = nn.LayerNorm(dim)
    self.layernorm3 = nn.LayerNorm(dim)
    self.layernorm4 = nn.LayerNorm(dim)

    self.kproj1 = nn.Linear(2 * dim, dim)
    self.kproj2 = nn.Linear(2 * dim, dim)
    self.kproj3 = nn.Linear(2 * dim, dim)
    self.kproj4 = nn.Linear(2 * dim, dim)

    self.spatialattention1 = SpatialWindowCrossAttention(dim, 2, num_heads, hw, ww)
    self.spatialattention2 = SpatialWindowCrossAttention(dim, 2, num_heads, hw, ww)
    self.spatialattention3 = SpatialWindowCrossAttention(dim, 2, num_heads, hw, ww)
    self.mlp = MLP(dim)

  def forward(self, ll, lh, hl, hh):

    b, c, h, w = ll.shape

    ll = ll.permute(0, 2, 3, 1)
    lh = lh.permute(0, 2, 3, 1)
    hl = hl.permute(0, 2, 3, 1)
    hh = hh.permute(0, 2, 3, 1)

    ll = ll.reshape(b, -1, c)
    lh = lh.reshape(b, -1, c)
    hl = hl.reshape(b, -1, c)
    hh = hh.reshape(b, -1, c)

    ll, lh = self.layernorm1(ll), self.layernorm1(lh)
    ll = ll + self.spatialattention1(ll, torch.cat([ll, lh], -1), lh, h, w)
    ll, hl = self.layernorm2(ll), self.layernorm2(hl)
    ll = ll + self.spatialattention2(ll, torch.cat([ll, hl], -1), hl, h, w)
    ll, hh = self.layernorm3(ll), self.layernorm3(hh)
    ll = ll + self.spatialattention3(ll, torch.cat([ll, hh], -1), hh, h, w)

    ll = ll + self.mlp(self.layernorm4(ll))

    ll = ll.reshape(b, h, w, c)
    ll = ll.permute(0, 3, 1, 2)

    return ll

class CrossEncoder(nn.Module):
    def __init__(self, dim, num_heads, hw, ww):
        super().__init__()

        self.spatial_layernorm1 = nn.LayerNorm(dim)
        self.spatial_layernorm2 = nn.LayerNorm(dim)
        self.spatial_layernorm3 = nn.LayerNorm(dim)
        self.spatial_crossattention = SpatialWindowCrossAttention(dim, 1, num_heads, hw, ww)
        self.spatial_mlp = MLP(dim)

        self.freq_layernorm1 = nn.LayerNorm(dim)
        self.freq_layernorm2 = nn.LayerNorm(dim)
        self.freq_layernorm3 = nn.LayerNorm(dim)
        self.freq_crossattention = SpatialWindowCrossAttention(dim, 1, num_heads, hw, ww)
        self.freq_mlp = MLP(dim)

    def forward(self, x_path0, x_path1, ll_path0, ll_path1):

        b, c, h, w = x_path0.shape

        x_path0 = x_path0.permute(0, 2, 3, 1).reshape(b, -1, c)
        x_path1 = x_path1.permute(0, 2, 3, 1).reshape(b, -1, c)

        x_path0 = self.spatial_layernorm1(x_path0)
        x_path1 = self.spatial_layernorm2(x_path1)

        x = x_path0 + self.spatial_crossattention(x_path0, x_path1, x_path1, h, w)
        x = x + self.spatial_mlp(self.spatial_layernorm3(x))

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        ll_path0 = ll_path0.permute(0, 2, 3, 1).reshape(b, -1, c)
        ll_path1 = ll_path1.permute(0, 2, 3, 1).reshape(b, -1, c)

        ll_path0 = self.freq_layernorm1(ll_path0)
        ll_path1 = self.freq_layernorm2(ll_path1)

        ll = ll_path1 + self.freq_crossattention(ll_path1, ll_path0, ll_path0, h, w)
        ll = ll + self.freq_mlp(self.freq_layernorm3(ll))

        ll = ll.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return x, ll

dwt = DWTForward(1, "haar", "symmetric").to(device)
idwt = DWTInverse("haar", "symmetric").to(device)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()

        out_dim = in_dim if out_dim == None else out_dim

        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.conv(x))

# ### encoder-2
# b, c, h, w, hw, ww = 8, 64, 31, 31, 8, 16
# num_heads = 4
# x = torch.randn(b, c, h, w).to(device)
# model = Encoder2(c, num_heads, hw, ww).to(device)
# y = model(x)
# print("Encoder-2:".ljust(30), x.shape, y.shape, x.shape == y.shape)

# ### encoder-3
# b, c, h, w, hw, ww = 8, 64, 31, 31, 8, 16
# num_heads = 4
# p = torch.randn(b, c, h, w).to(device)
# q = torch.randn(b, c, h, w).to(device)
# r = torch.randn(b, c, h, w).to(device)
# s = torch.randn(b, c, h, w).to(device)
# model = Encoder3(c, num_heads, hw, ww).to(device)
# y = model(p, q, r, s)
# print("Encoder-3:".ljust(30), p.shape, y.shape, p.shape == y.shape)

# ### wavelet transformation
# b, c, h, w = 8, 64, 32, 32
# x = torch.randn(b, c, h, w).to(device)
# ll, yh = dwt(x)
# lh, hl, hh = yh[0][:, :, 0, :, :], yh[0][:, :, 1, :, :], yh[0][:, :, 2, :, :]
# print("Forward Transformation:".ljust(30), hh.shape)
# y = idwt((ll, [torch.stack([lh, hl, hh], 2)]))
# print("Inverse Transformation:".ljust(30), x.shape, y.shape, x.shape == y.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# class BasicBlock(nn.Module):
#     def __init__(self, dim, num_heads, hw, ww):
#         super().__init__()

#         self.wave = "haar"
#         self.mode = "symmetric"

#         self.swt = SWTForward(dilation=1, padding_mode="reflect")
#         self.iswt = SWTInverse(dilation=1, padding_mode="reflect")

#         # self.spatial_conv1 = ConvBlock(dim)

#         self.spatial_encoder1 = Encoder2(dim, num_heads, hw, ww)
#         self.spatial_encoder2 = Encoder2(dim, num_heads, hw, ww)

#         # self.freq_conv1 = ConvBlock(dim)

#         self.freq_encoder1 = Encoder3(dim, num_heads, hw, ww)
#         self.freq_encoder2 = Encoder3(dim, num_heads, hw, ww)

#         self.cross_encoder = CrossEncoder(dim, num_heads, hw, ww)

#         self.fusion = nn.Sequential(
#            nn.Conv2d(2 * dim, dim, 3, 1, 1),
#            nn.GELU(),
#            nn.Conv2d(dim, dim, 1, 1, 0)
#         )

    # def forward(self, x):

    #     b, c, h, w = x.shape

    #     x = F.pad(x, [0, w % 2, 0, h % 2])

    #     # 0th stage
    #     x0_path0 = x
    #     ll0_path1, lh0_path1, hl0_path1, hh0_path1 = self.swt(x)

    #     # 1st stage
    #     x1_path0 = self.spatial_encoder1(x0_path0)
    #     # ll1_path1 = self.freq_encoder1(ll0_path1, lh0_path1, hl0_path1, hh0_path1)
    #     ll1_path1 = ll0_path1

    #     # 2nd stage
    #     x2_path0 = x1_path0
    #     ll2_path1 = ll1_path1

    #     # 3rd stage
    #     x2_path0, ll2_path1 = self.cross_encoder(x2_path0, ll2_path1, x2_path0, ll2_path1)

    #     ll3_path0, lh3_path0, hl3_path0, hh3_path0 = self.swt(x2_path0)
    #     x3_path1 = self.iswt(ll2_path1, lh0_path1, hl0_path1, hh0_path1)

    #     # 4th stage
    #     ll4_path0 = self.freq_conv3(ll3_path0)
    #     x4_path1 = self.spatial_conv3(x3_path1)

    #     # 5th stage
    #     # ll5_path0 = self.freq_encoder2(ll4_path0, lh3_path0, hl3_path0, hh3_path0)
    #     ll5_path0 = ll4_path0
    #     x5_path1 = self.spatial_encoder2(x4_path1)

    #     # 6th stage
    #     x6_path0 = self.iswt(ll5_path0, lh3_path0, hl3_path0, hh3_path0)


    #     # y = x + self.final_proj(torch.cat([x6_path0, x5_path1], 1))
    #     # new fusion
    #     y = x + self.fusion(torch.cat([x6_path0, x5_path1], 1))

    #     y = y[:, :, :h, :w]

    #     return y


# ###
# ### v112-noConv
# ###
# class BasicBlock(nn.Module):
#     def __init__(self, dim, num_heads, hw, ww):
#         super().__init__()

#         self.swt = SWTForward(dilation=1, padding_mode="reflect")
#         self.iswt = SWTInverse(dilation=1, padding_mode="reflect")

#         self.spatial_encoder1 = Encoder2(dim, num_heads, hw, ww)
#         self.spatial_encoder2 = Encoder2(dim, num_heads, hw, ww)

#         self.freq_encoder1 = Encoder3(dim, num_heads, hw, ww)
#         self.freq_encoder2 = Encoder3(dim, num_heads, hw, ww)

#         self.cross_encoder = CrossEncoder(dim, num_heads, hw, ww)

#         self.fusion = nn.Sequential(
#             nn.Conv2d(2 * dim, dim, 3, 1, 1),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 1, 1, 0)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = F.pad(x, [0, w % 2, 0, h % 2])

#         x1_path0 = x
#         ll1_path1, lh1_path1, hl1_path1, hh1_path1 = self.swt(x)

#         x2_path0 = self.spatial_encoder1(x1_path0)
#         ll2_path1 = self.freq_encoder1(ll1_path1, lh1_path1, hl1_path1, hh1_path1)

#         x3_path0, ll3_path1 = self.cross_encoder(x2_path0, ll2_path1, x2_path0, ll2_path1)

#         ll4_path0, lh4_path0, hl4_path0, hh4_path0 = self.swt(x3_path0)
#         x4_path1 = self.iswt(ll3_path1, lh1_path1, hl1_path1, hh1_path1)

#         ll5_path0 = self.freq_encoder2(ll4_path0, lh4_path0, hl4_path0, hh4_path0)
#         x5_path1 = self.spatial_encoder2(x4_path1)

#         x6_path0 = self.iswt(ll5_path0, lh4_path0, hl4_path0, hh4_path0)
#         x6_path1 = x5_path1

#         y = x + self.fusion(torch.cat([x6_path0, x6_path1], 1))
#         y = y[:, :, :h, :w]
#         return y

###
### v112-noConv-sameChannel
###
class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, hw, ww):
        super().__init__()

        self.swt = SWTForward(dilation=1, padding_mode="reflect")
        self.iswt = SWTInverse(dilation=1, padding_mode="reflect")

        self.proj0 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)

        self.spatial_encoder1 = Encoder2(dim // 2, num_heads, hw, ww)
        self.spatial_encoder2 = Encoder2(dim // 2, num_heads, hw, ww)

        self.freq_encoder1 = Encoder3(dim // 2, num_heads, hw, ww)
        self.freq_encoder2 = Encoder3(dim // 2, num_heads, hw, ww)

        self.cross_encoder = CrossEncoder(dim // 2, num_heads, hw, ww)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.pad(x, [0, w % 2, 0, h % 2])

        x0_path0 = self.proj0(x)
        x0_path1 = self.proj1(x)

        x1_path0 = x0_path0
        ll1_path1, lh1_path1, hl1_path1, hh1_path1 = self.swt(x0_path1)

        x2_path0 = self.spatial_encoder1(x1_path0)
        ll2_path1 = self.freq_encoder1(ll1_path1, lh1_path1, hl1_path1, hh1_path1)

        x3_path0, ll3_path1 = self.cross_encoder(x2_path0, ll2_path1, x2_path0, ll2_path1)

        ll4_path0, lh4_path0, hl4_path0, hh4_path0 = self.swt(x3_path0)
        x4_path1 = self.iswt(ll3_path1, lh1_path1, hl1_path1, hh1_path1)

        ll5_path0 = self.freq_encoder2(ll4_path0, lh4_path0, hl4_path0, hh4_path0)
        x5_path1 = self.spatial_encoder2(x4_path1)

        x6_path0 = self.iswt(ll5_path0, lh4_path0, hl4_path0, hh4_path0)
        x6_path1 = x5_path1

        y = x + self.fusion(torch.cat([x6_path0, x6_path1], 1))
        y = y[:, :, :h, :w]
        return y

# ###
# ### v113-noConv-sameChannel-noDownPath
# ###
# class BasicBlock(nn.Module):
#     def __init__(self, dim, num_heads, hw, ww):
#         super().__init__()

#         self.swt = SWTForward(dilation=1, padding_mode="reflect")
#         self.iswt = SWTInverse(dilation=1, padding_mode="reflect")

#         self.spatial_encoder1 = Encoder2(dim, num_heads, hw, ww)

#         self.freq_encoder2 = Encoder3(dim, num_heads, hw, ww)

#         self.fusion = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, 1, 0),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 1, 1, 0)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = F.pad(x, [0, w % 2, 0, h % 2])

#         x1_path0 = x

#         x2_path0 = self.spatial_encoder1(x1_path0)

#         x3_path0 = x2_path0

#         ll4_path0, lh4_path0, hl4_path0, hh4_path0 = self.swt(x3_path0)

#         ll5_path0 = self.freq_encoder2(ll4_path0, lh4_path0, hl4_path0, hh4_path0)

#         x6_path0 = self.iswt(ll5_path0, lh4_path0, hl4_path0, hh4_path0)

#         y = x + self.fusion(x6_path0)
#         y = y[:, :, :h, :w]
#         return y

# ### basic block
# b, c, h, w = 8, 64, 39, 31
# num_heads, hw, ww = 8, 8, 8
# x = torch.randn(b, c, h, w).to(device)
# model = BasicBlock(c, num_heads, hw, ww).to(device)
# y = model(x)
# print("Basic Block:".ljust(30), x.shape, y.shape, x.shape == y.shape)
