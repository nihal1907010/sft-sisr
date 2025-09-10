#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple

from PIL import Image

# Optional: NumPy/torch aren't required for this script,
# but we seed them anyway in case you later extend it.
try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Deterministic HR/LR paired random crops for SISR (default x4)."
    )
    p.add_argument("--hr_dir", type=str, default="datasets/Valid/DIV2K/HR", help="Folder with HR images.")
    p.add_argument("--lr_dir", type=str, default="datasets/Valid/DIV2K/LRx4", help="Folder with LR images.")
    p.add_argument("--out_hr_dir", type=str, default="datasets/Valid/DIV2K/HR_randomcrop", help="Output HR dir.")
    p.add_argument("--out_lr_dir", type=str, default="datasets/Valid/DIV2K/LRx4_randomcrop", help="Output LR dir.")
    p.add_argument("--scale", type=int, default=4, help="Downscale factor.")
    p.add_argument("--hr_patch", type=int, default=1000, help="HR patch size.")
    p.add_argument("--num_patches_per_image", type=int, default=1,
                   help="Patches to draw per image pair.")
    p.add_argument("--ext", type=str, default="png", help="Output extension (png/jpg/...).")
    p.add_argument("--seed", type=int, default=10, help="Seed for reproducibility.")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    p.add_argument("--allowed_exts", type=str, default="png,jpg,jpeg,webp,bmp",
                   help="Comma-separated allowed input extensions.")
    return p.parse_args()


def set_all_seeds(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        # Not strictly necessary here, but helps if you add torch ops later.
        try:
            torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except Exception:
            pass


def list_images(root: Path, exts: List[str], recursive: bool) -> List[Path]:
    patterns = [f"**/*.{e}" if recursive else f"*.{e}" for e in exts]
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    # Sort for deterministic traversal across runs/platforms
    return sorted(out, key=lambda p: str(p).lower())


def normalize_stem(stem: str, scale: int) -> str:
    s = stem.lower()
    tokens = [
        f"x{scale}", f"{scale}x",
        f"srf_{scale}", f"_srf_{scale}",
        "_lr", "-lr", ".lr",
        "_hr", "-hr", ".hr",
        f"_x{scale}", f"-x{scale}", f".x{scale}",
        f"_{scale}x", f"-{scale}x", f".{scale}x",
    ]
    for t in tokens:
        s = s.replace(t, "")
    s = re.sub(r"[_\-.]+", "_", s).strip("_- .")
    return s


def build_index(paths: List[Path], scale: int) -> Dict[str, Path]:
    # paths already sorted => first occurrence is deterministic (lexicographically smallest)
    idx: Dict[str, Path] = {}
    for p in paths:
        key = normalize_stem(p.stem, scale)
        idx.setdefault(key, p)
    return idx


def verify_sizes(hr_img: Image.Image, lr_img: Image.Image, scale: int) -> bool:
    hr_w, hr_h = hr_img.size
    lr_w, lr_h = lr_img.size
    return (hr_w == lr_w * scale) and (hr_h == lr_h * scale)


def sample_top_left(hr_w: int, hr_h: int, hr_patch: int, scale: int, rng: random.Random) -> Tuple[int, int]:
    max_x = hr_w - hr_patch
    max_y = hr_h - hr_patch
    if max_x < 0 or max_y < 0:
        return -1, -1
    # enforce grid alignment so LR coords are integers
    xs = list(range(0, max_x + 1, scale))
    ys = list(range(0, max_y + 1, scale))
    # rng is seeded => deterministic picks
    x = rng.choice(xs) if xs else 0
    y = rng.choice(ys) if ys else 0
    return x, y


def crop_pair(
    hr_img: Image.Image, lr_img: Image.Image, hr_xy: Tuple[int, int], hr_patch: int, scale: int
) -> Tuple[Image.Image, Image.Image]:
    x_hr, y_hr = hr_xy
    hr_crop = hr_img.crop((x_hr, y_hr, x_hr + hr_patch, y_hr + hr_patch))
    lr_patch = hr_patch // scale
    x_lr, y_lr = x_hr // scale, y_hr // scale
    lr_crop = lr_img.crop((x_lr, y_lr, x_lr + lr_patch, y_lr + lr_patch))
    return hr_crop, lr_crop


def main():
    args = parse_args()
    assert args.hr_patch % args.scale == 0, "hr_patch must be divisible by scale."

    set_all_seeds(args.seed)
    rng = random.Random(args.seed)  # local RNG (also deterministic)

    hr_dir = Path(args.hr_dir)
    lr_dir = Path(args.lr_dir)
    out_hr = Path(args.out_hr_dir); out_hr.mkdir(parents=True, exist_ok=True)
    out_lr = Path(args.out_lr_dir); out_lr.mkdir(parents=True, exist_ok=True)

    exts = [e.strip().lower() for e in args.allowed_exts.split(",") if e.strip()]

    hr_list = list_images(hr_dir, exts, args.recursive)
    lr_list = list_images(lr_dir, exts, args.recursive)

    if not hr_list:
        print(f"[WARN] No HR images found in {hr_dir}")
    if not lr_list:
        print(f"[WARN] No LR images found in {lr_dir}")

    hr_index = build_index(hr_list, args.scale)
    lr_index = build_index(lr_list, args.scale)

    # Deterministic key order
    keys = sorted(set(hr_index.keys()) & set(lr_index.keys()))

    print(f"[INFO] Seed={args.seed}")
    print(f"[INFO] Found {len(keys)} matched HR/LR pairs.")

    saved = 0
    skipped_small = 0
    skipped_size_mismatch = 0
    lr_patch = args.hr_patch // args.scale

    for key in keys:
        hr_path = hr_index[key]
        lr_path = lr_index[key]

        try:
            # PIL is deterministic for these ops; convert to RGB for consistency
            hr_img = Image.open(hr_path).convert("RGB")
            lr_img = Image.open(lr_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open pair ({hr_path.name}, {lr_path.name}): {e}")
            continue

        if not verify_sizes(hr_img, lr_img, args.scale):
            print(f"[WARN] Size mismatch (expect HR==LR*{args.scale}): {hr_path.name}, {lr_path.name}")
            skipped_size_mismatch += 1
            continue

        hr_w, hr_h = hr_img.size
        if hr_w < args.hr_patch or hr_h < args.hr_patch:
            print(f"[WARN] HR too small for {args.hr_patch}x{args.hr_patch}: {hr_path.name} ({hr_w}x{hr_h})")
            skipped_small += 1
            continue

        for i in range(args.num_patches_per_image):
            x_hr, y_hr = sample_top_left(hr_w, hr_h, args.hr_patch, args.scale, rng)
            if x_hr < 0:
                continue
            hr_crop, lr_crop = crop_pair(hr_img, lr_img, (x_hr, y_hr), args.hr_patch, args.scale)

            # Use HR stem to keep names aligned; deterministic due to sorted traversal
            stem = hr_path.stem
            fname = f"{stem}_{i:04d}.{args.ext}"
            hr_crop.save(out_hr / fname)
            lr_crop.save(out_lr / fname)
            saved += 1

    print(f"[DONE] Saved {saved} paired patches "
          f"(HR {args.hr_patch}x{args.hr_patch}, LR {lr_patch}x{lr_patch}, scale {args.scale}).")
    if skipped_small:
        print(f"[INFO] Skipped {skipped_small} images due to insufficient HR size.")
    if skipped_size_mismatch:
        print(f"[INFO] Skipped {skipped_size_mismatch} pairs due to HR/LR size mismatch.")


if __name__ == "__main__":
    main()
