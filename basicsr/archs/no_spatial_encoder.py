import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse
from basicsr.archs.thesis_utils import ConvBlock, CrossEncoder

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
    self.layernorm3 = nn.LayerNorm(dim)

    self.spatialattention = SpatialWindowSelfAttention(dim, num_heads, hw, ww)
    self.mlp = MLP(dim)

  def forward(self, x):

    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(b, -1, c)

    x = x + self.spatialattention(self.layernorm1(x), h, w)
    x = x + self.mlp(self.layernorm3(x))

    x = x.reshape(b, h, w, c)
    x = x.permute(0, 3, 1, 2)

    return x

device = "cuda" if torch.cuda.is_available() else "cpu"

class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, hw, ww):
        super().__init__()

        self.wave = "haar"
        self.mode = "symmetric"

        self.dwt = DWTForward(1, self.wave, self.mode)
        self.idwt = DWTInverse(self.wave, self.mode)

        self.proj1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.proj2 = nn.Conv2d(dim, dim // 2, 1, 1, 0)

        self.spatial_conv3 = ConvBlock(dim // 2)

        self.spatial_encoder1 = Encoder2(dim // 2, num_heads, hw, ww)
        self.spatial_encoder2 = Encoder2(dim // 2, num_heads, hw, ww)

        self.freq_conv3 = ConvBlock(dim // 2)

        self.freq_encoder1 = Encoder2(dim // 2, num_heads, hw, ww)
        self.freq_encoder2 = Encoder2(dim // 2, num_heads, hw, ww)

        self.cross_encoder = CrossEncoder(dim // 2, num_heads, hw, ww)

        self.final_proj = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):

        b, c, h, w = x.shape

        x = F.pad(x, [0, w % 2, 0, h % 2])

        x1 = self.proj1(x)
        x2 = self.proj2(x)

        x0_path0 = x1
        LL, H = self.dwt(x2)
        ll0_path1, lh0_path1, hl0_path1, hh0_path1 = LL, H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]

        x1_path0 = self.spatial_encoder1(x0_path0)
        ll1_path1 = self.freq_encoder1(ll0_path1)

        x2_path0 = x1_path0
        ll2_path1 = ll1_path1

        LL, H = self.dwt(x2_path0)
        ll3_path0, lh3_path0, hl3_path0, hh3_path0 = LL, H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        x3_path1 = self.idwt((ll2_path1, [torch.stack([lh0_path1, hl0_path1, hh0_path1], 2)]))

        x2_path0, ll2_path1 = self.cross_encoder(x2_path0, x3_path1, ll3_path0, ll2_path1)

        LL, H = self.dwt(x2_path0)
        ll3_path0, lh3_path0, hl3_path0, hh3_path0 = LL, H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        x3_path1 = self.idwt((ll2_path1, [torch.stack([lh0_path1, hl0_path1, hh0_path1], 2)]))

        ll4_path0 = self.freq_conv3(ll3_path0)
        x4_path1 = self.spatial_conv3(x3_path1)

        ll5_path0 = self.freq_encoder2(ll4_path0)
        x5_path1 = self.spatial_encoder2(x4_path1)

        x6_path0 = self.idwt((ll5_path0, [torch.stack([lh3_path0, hl3_path0, hh3_path0], 2)]))

        y = x + self.final_proj(torch.cat([x6_path0, x5_path1], 1))

        y = y[:, :, :h, :w]

        return y

### basic block
b, c, h, w = 8, 64, 39, 31
num_heads, hw, ww = 8, 8, 8
x = torch.randn(b, c, h, w).to(device)
model = BasicBlock(c, num_heads, hw, ww).to(device)
y = model(x)
print("Basic Block:".ljust(30), x.shape, y.shape, x.shape == y.shape)
