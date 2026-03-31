import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.v1_helper import _level1_swt2d, _level1_iswt2d

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
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")) # 2, hw, ww
    coords_flatten = torch.flatten(coords, 1) # 2, hw * ww
    relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1) # 2, hw * ww, hw * ww
    relative_coords = relative_coords.permute(1, 2, 0) # hw * ww, hw * ww, 2
    relative_coords[:, :, 0] += hw - 1
    relative_coords[:, :, 1] += ww - 1
    relative_coords[:, :, 0] *= 2 * ww - 1
    relative_position_index = relative_coords.sum(-1) # hw * ww, hw * ww
    self.register_buffer('relative_position_index', relative_position_index)

    self.attn_drop = nn.Dropout(0.0)
    self.proj_drop = nn.Dropout(0.0)

    nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    nn.init.xavier_uniform_(self.wqkv.weight)
    nn.init.zeros_(self.wqkv.bias)
    nn.init.xavier_uniform_(self.wp.weight)
    nn.init.zeros_(self.wp.bias)

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

    attention_weights = self.attn_drop(F.softmax(scaled_scores , -1)) # batch', num_heads, height_win * width_win, height_win * width_win
    y = torch.matmul(attention_weights , v) # batch', num_heads, height_win * width_win, channel / num_heads
    y = y.transpose(1, 2) # batch', height_win * width_win, num_heads, channel / num_heads
    y = y.reshape(-1, self.hw * self.ww, c) # batch', height_win * width_win, channel
    y = windowsToImage(y, h, w, self.hw, self.ww) # batch, height, width, channel
    out = self.proj_drop(self.wp(y)) # batch, height, width, channel
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
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")) # 2, hw, ww
    coords_flatten = torch.flatten(coords, 1) # 2, hw * ww
    relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1) # 2, hw * ww, hw * ww
    relative_coords = relative_coords.permute(1, 2, 0) # hw * ww, hw * ww, 2
    relative_coords[:, :, 0] += hw - 1
    relative_coords[:, :, 1] += ww - 1
    relative_coords[:, :, 0] *= 2 * ww - 1
    relative_position_index = relative_coords.sum(-1) # hw * ww, hw * ww
    self.register_buffer('relative_position_index', relative_position_index)

    self.attn_drop = nn.Dropout(0.0)
    self.proj_drop = nn.Dropout(0.0)

    nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    nn.init.xavier_uniform_(self.qproj.weight)
    nn.init.zeros_(self.qproj.bias)
    nn.init.xavier_uniform_(self.kproj.weight)
    nn.init.zeros_(self.kproj.bias)
    nn.init.xavier_uniform_(self.vproj.weight)
    nn.init.zeros_(self.vproj.bias)
    nn.init.xavier_uniform_(self.wp.weight)
    nn.init.zeros_(self.wp.bias)


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

    attention_weights = self.attn_drop(F.softmax(scaled_scores , -1)) # batch', num_heads, height_win * width_win, height_win * width_win
    y = torch.matmul(attention_weights , v) # batch', num_heads, height_win * width_win, channel / num_heads
    y = y.transpose(1, 2) # batch', height_win * width_win, num_heads, channel / num_heads
    y = y.reshape(-1, self.hw * self.ww, c) # batch', height_win * width_win, channel
    y = windowsToImage(y, h, w, self.hw, self.ww) # batch, heigh, width, channel
    out = self.proj_drop(self.wp(y)) # batch, height, width, channel
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

    self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    self.attn_drop = nn.Dropout(0.0)
    self.proj_drop = nn.Dropout(0.0)

    with torch.no_grad():
        self.temperature.fill_(1.0 / (self.head_dim ** 0.5))

    nn.init.xavier_uniform_(self.wqkv.weight)
    nn.init.zeros_(self.wqkv.bias)
    nn.init.xavier_uniform_(self.wp.weight)
    nn.init.zeros_(self.wp.bias)

  def forward(self, x): # batch, height * width, channel
    b, n, c = x.shape
    qkv = self.wqkv(x) # batch, height * width, 3 * channel
    qkv = qkv.reshape(b, n, 3, c) # batch, height * width, 3, channel
    # qkv = qkv.reshape(3, b, n, c) # 3, batch, height * width, channel
    qkv = qkv.permute(2, 0, 1, 3) # 3, batch, height * width, channel
    q, k, v = qkv[0], qkv[1], qkv[2] # batch, height * width, channel
    q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch, num_heads, height * width, channel / num_heads
    k = k.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch, num_heads, height * width, channel / num_heads
    v = v.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch, num_heads, height * width, channel / num_heads
    attention_scores = torch.matmul(q.transpose(-1, -2), k) # batch, num_heads, channel / num_heads, channel / num_heads
    scaled_scores = attention_scores * self.temperature # batch, num_heads, channel / num_heads, channel / num_heads

    attention_weights = self.attn_drop(F.softmax(scaled_scores , -1)) # batch, num_heads, channel / num_heads, channel / num_heads
    y = torch.matmul(attention_weights , v.transpose(-1, -2)) # batch, num_heads, channel / num_heads, height * width
    y = y.permute(0, 3, 1, 2) # batch, height * width, num_heads, channel / num_heads
    y = y.reshape(b, n, c) # batch, height * width, channel
    out = self.proj_drop(self.wp(y)) # batch, height * width, channel
    return out

class MLP(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.hidden_dim = 4 * dim

    self.fc1 = nn.Linear(dim, self.hidden_dim)
    self.fc2 = nn.Linear(self.hidden_dim, dim)

    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.zeros_(self.fc1.bias)
    nn.init.zeros_(self.fc2.weight)
    nn.init.zeros_(self.fc2.bias)

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

    self.layernorm11 = nn.LayerNorm(dim)
    self.layernorm21 = nn.LayerNorm(dim)
    self.layernorm31 = nn.LayerNorm(dim)
    self.layernorm12 = nn.LayerNorm(dim)
    self.layernorm22 = nn.LayerNorm(dim)
    self.layernorm32 = nn.LayerNorm(dim)
    self.layernorm4 = nn.LayerNorm(dim)

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

    ll_norm, lh_norm = self.layernorm11(ll), self.layernorm12(lh)
    ll = ll + self.spatialattention1(ll_norm, torch.cat([ll_norm, lh_norm], -1), lh_norm, h, w)
    ll_norm, hl_norm = self.layernorm21(ll), self.layernorm22(hl)
    ll = ll + self.spatialattention2(ll_norm, torch.cat([ll_norm, hl_norm], -1), hl_norm, h, w)
    ll_norm, hh_norm = self.layernorm31(ll), self.layernorm32(hh)
    ll = ll + self.spatialattention3(ll_norm, torch.cat([ll_norm, hh_norm], -1), hh_norm, h, w)

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

        x_path0_norm = self.spatial_layernorm1(x_path0)
        x_path1_norm = self.spatial_layernorm2(x_path1)

        x = x_path0 + self.spatial_crossattention(x_path0_norm, x_path1_norm, x_path1_norm, h, w)
        x = x + self.spatial_mlp(self.spatial_layernorm3(x))

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        ll_path0 = ll_path0.permute(0, 2, 3, 1).reshape(b, -1, c)
        ll_path1 = ll_path1.permute(0, 2, 3, 1).reshape(b, -1, c)

        ll_path0_norm = self.freq_layernorm1(ll_path0)
        ll_path1_norm = self.freq_layernorm2(ll_path1)

        ll = ll_path1 + self.freq_crossattention(ll_path1_norm, ll_path0_norm, ll_path0_norm, h, w)
        ll = ll + self.freq_mlp(self.freq_layernorm3(ll))

        ll = ll.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return x, ll

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

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, hw, ww, wavelet):
        super().__init__()

        self.proj0 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)

        self.spatial_encoder1 = Encoder2(dim // 2, num_heads, hw, ww)
        self.spatial_encoder2 = Encoder2(dim // 2, num_heads, hw, ww)

        self.freq_encoder1 = Encoder3(dim // 2, num_heads, hw, ww)
        self.freq_encoder2 = Encoder3(dim // 2, num_heads, hw, ww)

        self.freq_mix1 = nn.Conv2d(4 * dim // 2, 4 * dim // 2, 1, 1, 0, groups=dim // 2)
        self.freq_mix2 = nn.Conv2d(4 * dim // 2, 4 * dim // 2, 1, 1, 0, groups=dim // 2)

        self.cross_encoder = CrossEncoder(dim // 2, num_heads, hw, ww)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

        self.wavelet = wavelet

        self.res_scale = 0.1

        print("Block Version: Version 5; Wavelet Type: ", self.wavelet)

        def init_identity(m):
            with torch.no_grad():
                m.weight.zero_()
                m.bias.zero_()
                groups = m.in_channels // 4
                for g in range(groups):
                    for i in range(4):
                        m.weight[4*g+i, i, 0, 0] = 1.0

        init_identity(self.freq_mix1)
        init_identity(self.freq_mix2)

    def mix_bands(self, mixer, ll, lh, hl, hh):
        x = torch.cat([ll, lh, hl, hh], 1)
        x = mixer(x)
        return torch.chunk(x, 4, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.pad(x, [0, w % 2, 0, h % 2], mode="reflect")

        x0_path0 = self.proj0(x)
        x0_path1 = self.proj1(x)

        x1_path0 = x0_path0
        ll1_path1, lh1_path1, hl1_path1, hh1_path1 = _level1_swt2d(x0_path1, self.wavelet)

        x2_path0 = self.spatial_encoder1(x1_path0)
        ll2_path1 = self.freq_encoder1(ll1_path1, lh1_path1, hl1_path1, hh1_path1)

        x3_path0, ll3_path1 = self.cross_encoder(x2_path0, ll2_path1, x2_path0, ll2_path1)

        ll4_path0, lh4_path0, hl4_path0, hh4_path0 = _level1_swt2d(x3_path0, self.wavelet)
        ll3_path1, lh1_path1, hl1_path1, hh1_path1 = self.mix_bands(self.freq_mix1, ll3_path1, lh1_path1, hl1_path1, hh1_path1)
        x4_path1 = _level1_iswt2d(ll3_path1, lh1_path1, hl1_path1, hh1_path1, self.wavelet)

        ll5_path0 = self.freq_encoder2(ll4_path0, lh4_path0, hl4_path0, hh4_path0)
        x5_path1 = self.spatial_encoder2(x4_path1)

        ll5_path0, lh4_path0, hl4_path0, hh4_path0 = self.mix_bands(self.freq_mix2, ll5_path0, lh4_path0, hl4_path0, hh4_path0)
        x6_path0 = _level1_iswt2d(ll5_path0, lh4_path0, hl4_path0, hh4_path0, self.wavelet)
        x6_path1 = x5_path1

        y = x + self.res_scale * self.fusion(torch.cat([x6_path0, x6_path1], 1))
        y = y[:, :, :h, :w]
        return y


# ### basic block
# b, c, h, w = 8, 64, 39, 31
# num_heads, hw, ww = 8, 8, 8
# wavelet = 'gaus'
# x = torch.randn(b, c, h, w).to(device)
# model = BasicBlock(c, num_heads, hw, ww, wavelet).to(device)
# y = model(x)
# print("Nihal: Basic Block:".ljust(30), x.shape, y.shape, x.shape == y.shape)

