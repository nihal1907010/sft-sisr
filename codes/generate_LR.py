# downscale_folder.py
# Batch downscale images by a fixed factor using a chosen resampling kernel.
# No command-line arguments; edit the CONFIG block below.

import os
from typing import Tuple
from PIL import Image, ImageOps

# === CONFIG (edit me) =========================================================
INPUT_DIR  = "datasets/Test/Manga109/HR"
OUTPUT_DIR = "data/Manga109/LRx4_bilinear"
SCALE      = 4                 # Allowed: 2, 3, 4
KERNEL     = "bilinear"        # 'bilinear','bicubic','lanczos','nearest','box','hamming'
OVERWRITE  = False             # If False, skip files that already exist
# ============================================================================

assert SCALE in {2, 3, 4}, "SCALE must be one of 2, 3, or 4."

# Pillow 9/10 compatibility for resampling enums
def _resampling_attr(name: str):
    try:
        return getattr(Image.Resampling, name)
    except AttributeError:
        return getattr(Image, name)  # older Pillow fallback

RESAMPLE_MAP = {
    "nearest": _resampling_attr("NEAREST"),
    "bilinear": _resampling_attr("BILINEAR"),
    "bicubic": _resampling_attr("BICUBIC"),
    "lanczos": getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "ANTIALIAS", Image.BICUBIC)),
    "box": getattr(getattr(Image, "Resampling", Image), "BOX", _resampling_attr("BILINEAR")),
    "hamming": getattr(getattr(Image, "Resampling", Image), "HAMMING", _resampling_attr("BILINEAR")),
}

def get_resample(name: str):
    name = name.lower()
    if name not in RESAMPLE_MAP:
        raise ValueError(f"Unknown kernel '{name}'. Choose from: {', '.join(RESAMPLE_MAP.keys())}")
    return RESAMPLE_MAP[name]

def new_size(orig: Tuple[int, int], scale: int) -> Tuple[int, int]:
    w, h = orig
    nw = max(1, int(round(w / scale)))
    nh = max(1, int(round(h / scale)))
    return nw, nh

def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def save_params_for_ext(ext: str):
    ext = ext.lower()
    if ext in {".jpg", ".jpeg"}:
        return {"quality": 95, "optimize": True, "subsampling": 2}
    return {}

def ensure_mode_for_ext(img: Image.Image, ext: str) -> Image.Image:
    ext = ext.lower()
    # Apply EXIF orientation and pick a safe mode for the target format.
    img = ImageOps.exif_transpose(img)
    if ext in {".jpg", ".jpeg"} and img.mode not in {"RGB", "L"}:
        return img.convert("RGB")
    return img

def main():
    resample = get_resample(KERNEL)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for root, _, files in os.walk(INPUT_DIR):
        rel = os.path.relpath(root, INPUT_DIR)
        out_dir = os.path.join(OUTPUT_DIR, rel) if rel != "." else OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        for fname in files:
            in_path = os.path.join(root, fname)
            if not is_image(in_path):
                continue

            name, ext = os.path.splitext(fname)
            out_name = f"{name}_x{SCALE}_{KERNEL.lower()}{ext.lower()}"
            out_path = os.path.join(out_dir, out_name)

            if not OVERWRITE and os.path.exists(out_path):
                print(f"Skip (exists): {out_path}")
                continue

            try:
                with Image.open(in_path) as im:
                    im = ensure_mode_for_ext(im, ext)
                    target = new_size(im.size, SCALE)
                    down = im.resize(target, resample=resample)
                    down.save(out_path, **save_params_for_ext(ext))
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed: {in_path} -> {e}")

if __name__ == "__main__":
    main()
