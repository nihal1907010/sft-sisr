import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse
from basicsr.archs.thesis_utils import ConvBlock, Encoder2, Encoder3, CrossEncoder

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

        # self.spatial_conv1 = ConvBlock(dim)
        # self.spatial_conv2 = ConvBlock(dim)
        self.spatial_conv3 = ConvBlock(dim // 2)
        # self.spatial_conv4 = ConvBlock(dim)

        self.spatial_encoder1 = Encoder2(dim // 2, num_heads, hw, ww)
        self.spatial_encoder2 = Encoder2(dim // 2, num_heads, hw, ww)

        # self.freq_conv1 = ConvBlock(dim)
        # self.freq_conv2 = ConvBlock(dim)
        self.freq_conv3 = ConvBlock(dim // 2)
        # self.freq_conv4 = ConvBlock(dim)

        self.freq_encoder1 = Encoder3(dim // 2, num_heads, hw, ww)
        self.freq_encoder2 = Encoder3(dim // 2, num_heads, hw, ww)

        # self.cross_encoder = CrossEncoder(dim // 2, num_heads, hw, ww)

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
        ll1_path1 = self.freq_encoder1(ll0_path1, lh0_path1, hl0_path1, hh0_path1)

        x2_path0 = x1_path0
        ll2_path1 = ll1_path1

        LL, H = self.dwt(x2_path0)
        ll3_path0, lh3_path0, hl3_path0, hh3_path0 = LL, H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        x3_path1 = self.idwt((ll2_path1, [torch.stack([lh0_path1, hl0_path1, hh0_path1], 2)]))

        ll4_path0 = self.freq_conv3(ll3_path0)
        x4_path1 = self.spatial_conv3(x3_path1)

        ll5_path0 = self.freq_encoder2(ll4_path0, lh3_path0, hl3_path0, hh3_path0)
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
