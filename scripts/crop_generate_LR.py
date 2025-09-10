import os
from pathlib import Path
import random
import torch.nn.functional as F
import torchvision.io as io

# --- CONFIG ---
input_dir = "datasets/sr"       # folder containing your input images
output_dir = "datasets/sr_down" # folder to save results
crop_size = 1024            # HR crop size (square)
scales = [2, 3, 4]          # downsampling factors
extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
random_crop = False         # if False -> center crop, if True -> random crop
# ---------------

Path(output_dir).mkdir(parents=True, exist_ok=True)

def list_images(root, exts):
    exts = {e.lower() for e in exts}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]

def crop_image(img, size, random_crop=False):
    """Crop a square patch (CxHxW tensor)."""
    _, H, W = img.shape
    if H < size or W < size:
        raise ValueError(f"Image too small for {size}x{size} crop: {H}x{W}")

    if random_crop:
        top = random.randint(0, H - size)
        left = random.randint(0, W - size)
    else:
        top = (H - size) // 2
        left = (W - size) // 2

    return img[:, top:top+size, left:left+size]

images = list_images(input_dir, extensions)

for img_path in images:
    try:
        img = io.read_image(str(img_path))  # CxHxW, uint8

        # normalize to RGB
        if img.shape[0] == 4:  # drop alpha
            img = img[:3]
        elif img.shape[0] == 1:  # grayscale -> 3ch
            img = img.repeat(3, 1, 1)

        # --- STEP 1: Crop HR 1024x1024 ---
        HR = crop_image(img, crop_size, random_crop)
        HR_f = HR.float().unsqueeze(0) / 255.0  # 1xCxHxW

        # Save HR
        hr_folder = Path(output_dir) / "HR"
        hr_folder.mkdir(parents=True, exist_ok=True)
        save_name = Path(img_path).with_suffix(".png").name
        hr_path = hr_folder / save_name
        io.write_png(HR, str(hr_path))

        # --- STEP 2: Generate LRs from HR crop ---
        _, _, H, W = HR_f.shape
        for s in scales:
            nh, nw = H // s, W // s
            LR = F.interpolate(
                HR_f, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True
            )
            LR_out = (LR.clamp(0, 1).squeeze(0) * 255.0).byte()

            out_folder = Path(output_dir) / f"x{s}"
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = out_folder / save_name
            io.write_png(LR_out, str(out_path))

        print(f"Processed: {img_path}")

    except Exception as e:
        print(f"Failed {img_path}: {e}")

print("Done.")
