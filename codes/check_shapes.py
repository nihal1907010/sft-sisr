# check_lr_hr_recursive.py
# Recursively verify that HR images are exactly SCALE times larger than LR images.
# Matches strictly by filename (same name in both LR and HR).
# Edit the CONFIG section for your folders and scale.

import os
from typing import Dict, Tuple, Optional
from PIL import Image, ImageOps

# === CONFIG (edit me) =========================================================
LR_DIR     = "data/B100/LRx4_bilinear"
HR_DIR     = "data/B100/HR"
SCALE      = 4                 # Allowed values: 2, 3, 4 (or any integer >= 2)
# ============================================================================

assert isinstance(SCALE, int) and SCALE >= 2, "SCALE must be an integer >= 2."
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def walk_images(folder: str):
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if os.path.isfile(p) and is_image(p):
                yield p

def safe_size(path: str) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # handle orientation
            return im.size  # (width, height)
    except Exception as e:
        print(f"[ERR] Failed to read image size: {path} -> {e}")
        return None

def index_by_filename(folder: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    collisions = []
    for p in walk_images(folder):
        fname = os.path.basename(p)
        if fname in idx:
            collisions.append((fname, idx[fname], p))
        else:
            idx[fname] = p
    if collisions:
        print("\n[WARN] Duplicate filenames detected (using first occurrence):")
        for fname, p1, p2 in collisions:
            print(f"  '{fname}':\n    {p1}\n    {p2}")
        print()
    return idx

def main():
    hr_index = index_by_filename(HR_DIR)

    total = 0
    matched = 0
    missing_in_hr = []
    unreadable = []
    mismatches = []

    for lr_path in walk_images(LR_DIR):
        total += 1
        fname = os.path.basename(lr_path)
        hr_path = hr_index.get(fname)

        if not hr_path:
            missing_in_hr.append(lr_path)
            continue

        matched += 1
        lr_wh = safe_size(lr_path)
        hr_wh = safe_size(hr_path)

        if lr_wh is None or hr_wh is None:
            unreadable.append((lr_path, hr_path))
            continue

        lr_w, lr_h = lr_wh
        hr_w, hr_h = hr_wh
        exp_w, exp_h = lr_w * SCALE, lr_h * SCALE

        if (hr_w, hr_h) != (exp_w, exp_h):
            mismatches.append({
                "file": fname,
                "lr": (lr_w, lr_h, lr_path),
                "hr": (hr_w, hr_h, hr_path),
                "expected_hr": (exp_w, exp_h),
            })

    # ---- Reporting ----
    print("\n=== LR vs HR Scale Check (recursive, same filename) ===")
    print(f"SCALE expected:    {SCALE}x (HR should be LR * {SCALE})")
    print(f"LR images scanned: {total}")
    print(f"Pairs matched:     {matched}")

    if missing_in_hr:
        print(f"\n[Missing in HR] ({len(missing_in_hr)})")
        for p in missing_in_hr:
            print("  " + p)

    if unreadable:
        print(f"\n[Unreadable images] ({len(unreadable)})")
        for lr_p, hr_p in unreadable:
            print(f"  LR: {lr_p}")
            print(f"  HR: {hr_p}")

    if mismatches:
        print(f"\n[MISMATCHED sizes] ({len(mismatches)})")
        for m in mismatches:
            (lr_w, lr_h, lr_p) = m["lr"]
            (hr_w, hr_h, hr_p) = m["hr"]
            (exp_w, exp_h) = m["expected_hr"]
            print(f"  '{m['file']}':")
            print(f"    LR: {lr_w}x{lr_h}  ->  expected HR: {exp_w}x{exp_h}")
            print(f"    HR: {hr_w}x{hr_h}  @  {hr_p}")

    if not missing_in_hr and not unreadable and not mismatches:
        print("\nAll checked pairs match the expected scale. ✅")

if __name__ == "__main__":
    main()
