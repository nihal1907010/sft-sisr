import torch
from thop import profile
import csv
import os

def profile_sr_model(model, scale=4, img_size=64, device="cuda", csv_path="sr_profile.csv"):
    """
    Profile a single-image SR model and save results to CSV.

    Args:
        model: torch.nn.Module (your SR model instance)
        scale: upscale factor (x2, x3, x4, etc.)
        img_size: LR input size (HxW)
        device: "cuda" or "cpu"
        csv_path: file to save results
    """
    model = model.to(device).eval()

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dummy LR input (1 image, 3 channels, HxW)
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Compute MACs and FLOPs
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    macs_g = macs / 1e9
    flops_g = 2 * macs_g  # FLOPs ≈ 2 × MACs

    # Collect results
    results = {
        "img_size": img_size,
        "scale": scale,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "params_M": total_params / 1e6,
        "params_trainable_M": trainable_params / 1e6,
        "macs_G": macs_g,
        "flops_G": flops_g,
        "device": device,
    }

    # Print summary
    print("\n=== Model Profiling (SR) ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")
    print("============================\n")

    # Save to CSV (append if file exists)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    print(f"Saved results to {csv_path}")
    return results


# --------------------------------------------------------------------
# Example usage:
from basicsr.archs.edsr_arch import EDSR
# main
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=192,
#              num_block=6,
#              upscale=4,
#              num_heads=6,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# only spatial path
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# only frequency path
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# without domain swap
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# without cross attention
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# without spatial encoder
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# without frequency encoder
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# additional convolutional layers
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# wide
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=192,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# deep
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=8,
#              upscale=4,
#              num_heads=8,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# conv
# model = EDSR(num_in_ch=3,
#              num_out_ch=3,
#              num_feat=64,
#              num_block=4,
#              upscale=4,
#              num_heads=4,
#              hw=8,
#              ww=8,
#              img_range=255.,
#              rgb_mean=[0.4488, 0.4371, 0.4040])

# main-2
model = EDSR(num_in_ch=3,
             num_out_ch=3,
             num_feat=96,
             num_block=10,
             upscale=4,
             num_heads=8,
             hw=8,
             ww=8,
             img_range=255.,
             rgb_mean=[0.4488, 0.4371, 0.4040])

profile_sr_model(model, scale=4, img_size=64, csv_path="stats/main-2.csv")
