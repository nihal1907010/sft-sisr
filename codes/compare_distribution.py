#!/usr/bin/env python3
"""
Compare Fourier energy distributions across three folders of images.

Usage:
  python fourier_compare.py /path/to/folderA /path/to/folderB /path/to/folderC /path/to/output

Notes:
- Images are matched by filename. Only names present in all three folders are processed.
- Each triplet is resized to the smallest HxW among the three before FFT, so frequencies align.
- Output plots are saved as <name>_fourier_energy.png in the output folder.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Utilities ----------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path):
    return {p.name: p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()}


def load_gray(path: Path) -> np.ndarray:
    # Convert to grayscale float32 in range [0,1]
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def resize_to(shape_hw, arr: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    if arr.shape == (h, w):
        return arr
    return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0


def radial_energy_distribution(img: np.ndarray, nbins: int = 300):
    """
    Compute radially averaged Fourier energy distribution (power spectrum).
    Returns (bin_centers_freq, normalized_energy_per_bin).
    Frequency units are cycles per pixel (cpp).
    """
    h, w = img.shape

    # 2D FFT and power spectrum
    F = np.fft.fft2(img)
    P = np.abs(F) ** 2  # power
    P = np.fft.fftshift(P)

    # Frequency grids (cycles per pixel)
    fy = np.fft.fftfreq(h, d=1.0)
    fx = np.fft.fftfreq(w, d=1.0)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    FR = np.sqrt(FX ** 2 + FY ** 2)  # radial frequency magnitude

    # Radial binning
    fr_flat = FR.ravel()
    p_flat = P.ravel()

    # Bins from 0 to max radial freq present
    fmax = fr_flat.max()
    bins = np.linspace(0, fmax, nbins + 1)
    bin_idx = np.digitize(fr_flat, bins) - 1
    # clip to valid range
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    # Sum energy per bin and count per bin
    energy_per_bin = np.bincount(bin_idx, weights=p_flat, minlength=nbins).astype(np.float64)
    count_per_bin = np.bincount(bin_idx, minlength=nbins).astype(np.float64)
    # Avoid divide-by-zero; use mean energy per radius ring
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_energy = np.where(count_per_bin > 0, energy_per_bin / count_per_bin, 0.0)

    # Normalize to unit area (so curves are comparable)
    area = mean_energy.sum()
    if area > 0:
        mean_energy /= area

    # Bin centers
    f_centers = 0.5 * (bins[:-1] + bins[1:])

    return f_centers, mean_energy


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def common_filenames(a: dict, b: dict, c: dict):
    return sorted(set(a.keys()) & set(b.keys()) & set(c.keys()))


def smallest_shape(arrays):
    """Return (h, w) of the smallest area image among list."""
    shapes = [(arr.shape[0], arr.shape[1]) for arr in arrays]
    # sort by area, then by h, then by w
    h, w = sorted(shapes, key=lambda s: (s[0] * s[1], s[0], s[1]))[0]
    return h, w


# ---------- Main processing ----------

def process_triplets(dirA: Path, dirB: Path, dirC: Path, out_dir: Path,
                     nbins: int = 300, y_log: bool = True):
    imgsA = list_images(dirA)
    imgsB = list_images(dirB)
    imgsC = list_images(dirC)

    names = common_filenames(imgsA, imgsB, imgsC)
    if not names:
        print("No common image filenames found across the three folders.")
        return

    ensure_outdir(out_dir)

    print(f"Found {len(names)} common images. Processing...")

    for name in names:
        try:
            arrA = load_gray(imgsA[name])
            arrB = load_gray(imgsB[name])
            arrC = load_gray(imgsC[name])

            # Resize each to the smallest shape among the three
            h_min, w_min = smallest_shape([arrA, arrB, arrC])
            arrA = resize_to((h_min, w_min), arrA)
            arrB = resize_to((h_min, w_min), arrB)
            arrC = resize_to((h_min, w_min), arrC)

            fA, eA = radial_energy_distribution(arrA, nbins=nbins)
            fB, eB = radial_energy_distribution(arrB, nbins=nbins)
            fC, eC = radial_energy_distribution(arrC, nbins=nbins)

            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(fA, eA, label=dirA.name)
            plt.plot(fB, eB, label=dirB.name)
            plt.plot(fC, eC, label=dirC.name)
            plt.xlabel("Spatial frequency (cycles/pixel)")
            plt.ylabel("Normalized energy (radial mean)")
            plt.title(f"Fourier Energy Distribution: {name}")
            if y_log:
                plt.yscale("log")
            plt.xlim(0, min(fA.max(), fB.max(), fC.max()))
            plt.legend()
            plt.tight_layout()

            out_name = Path(name).stem + "_fourier_energy.png"
            plt.savefig(out_dir / out_name, dpi=150)
            plt.close()

        except Exception as e:
            print(f"[WARN] Skipped {name} due to error: {e}")

    print(f"Done. Saved plots to: {out_dir}")


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)

    dirA = Path(sys.argv[1]).expanduser().resolve()
    dirB = Path(sys.argv[2]).expanduser().resolve()
    dirC = Path(sys.argv[3]).expanduser().resolve()
    out_dir = Path(sys.argv[4]).expanduser().resolve()

    for d in [dirA, dirB, dirC]:
        if not d.exists() or not d.is_dir():
            print(f"Input folder does not exist or is not a directory: {d}")
            sys.exit(2)

    process_triplets(dirA, dirB, dirC, out_dir, nbins=300, y_log=True)


if __name__ == "__main__":
    main()
