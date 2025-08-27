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

        self.freq_conv1 = ConvBlock(dim)
        self.freq_conv2 = ConvBlock(dim)

        self.freq_encoder1 = Encoder3(dim, num_heads, hw, ww)
        self.freq_encoder2 = Encoder3(dim, num_heads, hw, ww)

    def forward(self, x):

        b, c, h, w = x.shape

        x = F.pad(x, [0, w % 2, 0, h % 2])

        LL, H = self.dwt(x)
        ll, lh, hl, hh = LL, H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]

        ll = self.freq_conv1(ll)
        ll = self.freq_encoder1(ll, lh, hl, hh)

        ll = self.freq_conv2(ll)
        ll = self.freq_encoder2(ll, lh, hl, hh)

        y = x + self.idwt((ll, [torch.stack([lh, hl, hh], 2)]))

        y = y[:, :, :h, :w]

        return y

# ### basic block
# b, c, h, w = 8, 64, 39, 31
# num_heads, hw, ww = 8, 8, 8
# x = torch.randn(b, c, h, w).to(device)
# model = BasicBlock(c, num_heads, hw, ww).to(device)
# y = model(x)
# print("Basic Block:".ljust(30), x.shape, y.shape, x.shape == y.shape)
