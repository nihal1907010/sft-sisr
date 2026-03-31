import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
# thesis_v0_arch.py -- main
# from basicsr.archs.thesis_v0_arch import BasicBlock
# thesis_v1_arch.py
# from basicsr.archs.thesis_v1_arch import BasicBlock
# thesis_v100.py -- conv
# from basicsr.archs.thesis_v100 import BasicBlock
# thesis_v101.py -- without domain swap
# from basicsr.archs.thesis_v101 import BasicBlock
# thesis_v102.py -- only spatial path
# from basicsr.archs.thesis_v102 import BasicBlock
# thesis_v103.py -- only frequency path
# from basicsr.archs.thesis_v103 import BasicBlock
# thesis_v104.py -- without cross attention
# from basicsr.archs.thesis_v104 import BasicBlock
# thesis_v105.py -- no frequency encoder
# from basicsr.archs.thesis_v105 import BasicBlock
# thesis_v106.py
# from basicsr.archs.thesis_v106 import BasicBlock
# thesis_v107.py
# from basicsr.archs.thesis_v107 import BasicBlock
# thesis_v108.py
# from basicsr.archs.thesis_v109 import BasicBlock
# thesis_v110.py
# from basicsr.archs.thesis_v111 import BasicBlock
# thesis_v114.py
# from basicsr.archs.v114 import BasicBlock

# no spatial encoder
# from basicsr.archs.no_spatial_encoder import BasicBlock



# v201.py -- wide, deep
# from basicsr.archs.v201 import BasicBlock


from basicsr.archs.v4_09032026 import BasicBlock


from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 num_heads=4,
                 hw=8,
                 ww=8,
                 wavelet=None,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # thesis_v0_arch.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v1_arch.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v100.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v101.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v102.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v103.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v104.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v105.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v106.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v107.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v108.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # thesis_v110.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)
        # v114.py
        # self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww)


        # v201.py
        self.body = make_layer(BasicBlock, num_block, dim=num_feat, num_heads=num_heads, hw=hw, ww=ww, wavelet=wavelet)


        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
