# upscale_folder.py
# Batch upscale images by a fixed factor using a chosen resampling kernel.
# No command-line arguments; edit the CONFIG block below.

import os
from typing import Tuple
from PIL import Image, ImageOps

# === CONFIG (edit me) =========================================================
INPUT_DIR  = "data/Urban100/LRx4_bilinear"
OUTPUT_DIR = "data/Urban100/upscaled_bilinear_lanczos"
SCALE      = 4                 # Common: 2, 3, 4 (any int >= 2 is fine)
KERNEL     = "lanczos"         # 'lanczos','bicubic','bilinear','nearest','box','hamming'
OVERWRITE  = False             # If False, skip files that already exist
# ============================================================================

assert isinstance(SCALE, int) and SCALE >= 2, "SCALE must be an integer >= 2."

# Pillow 9/10 compatibility for resampling enums
def _resampling_attr(name: str):
    try:
        return getattr(Image.Resampling, name)
    except AttributeError:
        return getattr(Image, name)  # older Pillow fallback

RESAMPLE_MAP = {
    "nearest":  _resampling_attr("NEAREST"),
    "bilinear": _resampling_attr("BILINEAR"),
    "bicubic":  _resampling_attr("BICUBIC"),
    "lanczos":  getattr(getattr(Image, "Resampling", Image), "LANCZOS",
                        getattr(Image, "ANTIALIAS", _resampling_attr("BICUBIC"))),
    "box":      getattr(getattr(Image, "Resampling", Image), "BOX", _resampling_attr("BILINEAR")),
    "hamming":  getattr(getattr(Image, "Resampling", Image), "HAMMING", _resampling_attr("BILINEAR")),
}

def get_resample(name: str):
    name = name.lower()
    if name not in RESAMPLE_MAP:
        raise ValueError(f"Unknown kernel '{name}'. Choose from: {', '.join(RESAMPLE_MAP.keys())}")
    return RESAMPLE_MAP[name]

def new_size_up(orig: Tuple[int, int], scale: int) -> Tuple[int, int]:
    w, h = orig
    return max(1, w * scale), max(1, h * scale)

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
    img = ImageOps.exif_transpose(img)  # correct orientation
    # JPEG doesn't support alpha; convert to a safe mode
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
            out_name = f"{name}_upx{SCALE}_{KERNEL.lower()}{ext.lower()}"
            out_path = os.path.join(out_dir, out_name)

            if not OVERWRITE and os.path.exists(out_path):
                print(f"Skip (exists): {out_path}")
                continue

            try:
                with Image.open(in_path) as im:
                    im = ensure_mode_for_ext(im, ext)
                    target = new_size_up(im.size, SCALE)
                    up = im.resize(target, resample=resample)
                    up.save(out_path, **save_params_for_ext(ext))
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed: {in_path} -> {e}")

if __name__ == "__main__":
    main()
