from basicsr.archs.edsr_arch import EDSR

import torch

model = EDSR(num_in_ch=3,
             num_out_ch=3,
             num_feat=64,
             num_block=8,
             upscale=4,
             num_heads=8,
             hw=8,
             ww=8,
             wavelet='db1',
             img_range=255.,
             rgb_mean=[0.4488, 0.4371, 0.4040])

total_params = sum(p.numel() for p in model.parameters()) / 1e6
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True) / 1e6

print("Total Parameters: ".ljust(30), f"{total_params:.2f}M")
print("Total Trainable Parameters: ".ljust(30), f"{total_trainable_params:.2f}M")

# input = torch.randn(8, 3, 64, 64).to("cuda")
# output = model(input).to("cuda")
# print(input.shape, output.shape)