import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def list_images(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def read_image_rgb(path: Path, max_size: int | None) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if max_size is not None:
        h, w = img.shape[:2]
        m = max(h, w)
        if m > max_size:
            scale = max_size / m
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def partial_sum(path: Path, max_size: int | None) -> tuple[np.ndarray, int] | None:
    try:
        img = read_image_rgb(path, max_size)
        if img is None:
            return None
        s = img.reshape(-1, 3).sum(axis=0, dtype=np.float64)
        n = img.shape[0] * img.shape[1]
        return s, n
    except:
        return None

def compute_mean_rgb(folder: Path, recursive: bool, workers: int, max_size: int | None):
    paths = list_images(folder, recursive)
    if not paths:
        raise SystemExit("No images found.")

    total_sum = np.zeros(3, dtype=np.float64)
    total_px = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(partial_sum, p, max_size) for p in paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="img"):
            res = f.result()
            if res is None:
                continue
            s, n = res
            total_sum += s
            total_px += n

    if total_px == 0:
        raise SystemExit("All images failed to load.")

    mean_rgb_255 = total_sum / total_px
    mean_rgb_norm = mean_rgb_255 / 255.0

    return mean_rgb_255.tolist(), mean_rgb_norm.tolist(), len(paths), total_px

def main():
    ap = argparse.ArgumentParser(description="Compute mean RGB values for a folder of images.")
    ap.add_argument("folder", type=str, help="Path to the image folder")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="Number of parallel workers (default: CPU cores)")
    ap.add_argument("--max-size", type=int, default=None,
                    help="Optional longest-side size to downscale images before computing mean (faster)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    mean_rgb_255, mean_rgb_norm, num_imgs, total_px = compute_mean_rgb(
        folder, args.recursive, args.workers, args.max_size
    )

    print(f"Images processed: {num_imgs}")
    print(f"Total pixels (after optional downscale): {total_px:,}")
    print("Mean RGB (0–255):", [round(x, 6) for x in mean_rgb_255])
    print("Mean RGB (0–1):", [round(x, 6) for x in mean_rgb_norm])

if __name__ == "__main__":
    main()
