import os
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import torchvision.io as io

# --- CONFIG ---
input_dir = "datasets/sr"       # folder containing your input images
output_dir = "datasets/sr_down" # folder to save results
scales = [2, 3, 4]         # downsampling factors
extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
# ---------------

Path(output_dir).mkdir(parents=True, exist_ok=True)

def list_images(root, exts):
    exts = {e.lower() for e in exts}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]

def safe_new_size(h, w, s):
    return max(h // s, 1), max(w // s, 1)

images = list_images(input_dir, extensions)

for img_path in images:
    try:
        img = io.read_image(str(img_path))  # CxHxW, uint8
        # normalize to RGB
        if img.shape[0] == 4:  # drop alpha
            img = img[:3]
        elif img.shape[0] == 1:  # grayscale -> 3ch
            img = img.repeat(3, 1, 1)

        C, H, W = img.shape
        img_f = img.float().unsqueeze(0) / 255.0  # 1xCxHxW

        for s in scales:
            nh, nw = safe_new_size(H, W, s)
            down = F.interpolate(
                img_f, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True
            )
            out = (down.clamp(0, 1).squeeze(0) * 255.0).byte()

            out_folder = Path(output_dir) / f"x{s}"
            out_folder.mkdir(parents=True, exist_ok=True)

            save_path = out_folder / Path(img_path).with_suffix(".png").name
            io.write_png(out, str(save_path))

        print(f"Processed: {img_path}")

    except Exception as e:
        print(f"Failed {img_path}: {e}")

print("Done.")
