import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from pytorch_wavelets import DWTForward, DWTInverse
from basicsr.archs.thesis_utils import ConvBlock, Encoder2, Encoder3, CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, hw, ww):
        super().__init__()

        self.spatial_conv1 = ConvBlock(dim)
        self.spatial_conv2 = ConvBlock(dim)

        self.spatial_encoder1 = Encoder2(dim, num_heads, hw, ww)
        self.spatial_encoder2 = Encoder2(dim, num_heads, hw, ww)

    def forward(self, x):

        b, c, h, w = x.shape

        x = F.pad(x, [0, w % 2, 0, h % 2])

        xx = self.spatial_conv1(x)
        xx = self.spatial_encoder1(xx)

        y = random.randint(0, 1)

        # if y == 1:
        xx = self.spatial_conv2(xx)
        xx = self.spatial_encoder2(xx)

        y = x + xx

        y = y[:, :, :h, :w]

        return y

# ### basic block
# b, c, h, w = 8, 64, 39, 31
# num_heads, hw, ww = 8, 8, 8
# x = torch.randn(b, c, h, w).to(device)
# model = BasicBlock(c, num_heads, hw, ww).to(device)
# y = model(x)
# print("Basic Block:".ljust(30), x.shape, y.shape, x.shape == y.shape)
