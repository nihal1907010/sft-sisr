# #!/usr/bin/env python3
# """
# Make "Normalized Power" plots like the screenshot.

# Usage:
#   python make_normalized_power_plots.py /path/HR /path/Classical /path/Deep /path/output

# Notes:
# - Folders must contain images with the SAME filenames.
# - Each triplet is resized to the smallest HxW before FFT so the spectra align.
# - Output: <name>_normalized_power.png in the output folder.
# """

# import os
# import sys
# from pathlib import Path
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch

# IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# def list_images(folder: Path):
#     return {p.name: p for p in folder.iterdir()
#             if p.suffix.lower() in IMG_EXTS and p.is_file()}

# def load_gray(path: Path) -> np.ndarray:
#     im = Image.open(path).convert("L")
#     arr = np.asarray(im, dtype=np.float32) / 255.0
#     return arr

# def resize_to(shape_hw, arr: np.ndarray) -> np.ndarray:
#     h, w = shape_hw
#     if arr.shape == (h, w):
#         return arr
#     return np.asarray(
#         Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
#         dtype=np.float32
#     ) / 255.0

# def smallest_shape(arrays):
#     shapes = [(a.shape[0], a.shape[1]) for a in arrays]
#     return sorted(shapes, key=lambda s: (s[0]*s[1], s[0], s[1]))[0]

# def radial_energy_distribution(img: np.ndarray, nbins: int = 300):
#     """
#     Return (radius_index 0..nbins-1, normalized mean power per radius bin).
#     We bin by radius in the FFT magnitude (fftshifted), then normalize to unit area.
#     """
#     h, w = img.shape

#     # 2D FFT power spectrum, DC at center
#     F = np.fft.fft2(img)
#     P = np.abs(F)**2
#     P = np.fft.fftshift(P)

#     # radius per pixel (in pixels of the FFT grid, not cycles/pixel)
#     yy, xx = np.indices((h, w))
#     cy, cx = (h-1)/2.0, (w-1)/2.0
#     r = np.sqrt((yy - cy)**2 + (xx - cx)**2)

#     # max possible radius (corner)
#     rmax = r.max()

#     # bin edges from 0..rmax mapped to nbins
#     bins = np.linspace(0, rmax, nbins + 1)
#     ridx = np.digitize(r.ravel(), bins) - 1
#     ridx = np.clip(ridx, 0, nbins - 1)

#     # mean power per radius ring
#     power_sum = np.bincount(ridx, weights=P.ravel(), minlength=nbins).astype(np.float64)
#     count = np.bincount(ridx, minlength=nbins).astype(np.float64)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         mean_power = np.where(count > 0, power_sum / count, 0.0)

#     # normalize to unit area so curves are directly comparable
#     s = mean_power.sum()
#     if s > 0:
#         mean_power /= s

#     # X axis is "Frequency radius" in bin indices (0..nbins-1)
#     radius_idx = np.arange(nbins, dtype=np.float64)
#     return radius_idx, mean_power

# def ensure_outdir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)

# def main():
#     if len(sys.argv) != 5:
#         print(__doc__)
#         sys.exit(1)

#     dir_hr = Path(sys.argv[1]).expanduser().resolve()
#     dir_classical = Path(sys.argv[2]).expanduser().resolve()
#     dir_deep = Path(sys.argv[3]).expanduser().resolve()
#     out_dir = Path(sys.argv[4]).expanduser().resolve()

#     for d in [dir_hr, dir_classical, dir_deep]:
#         if not d.exists() or not d.is_dir():
#             print(f"Input folder does not exist or is not a directory: {d}")
#             sys.exit(2)

#     imgs_hr = list_images(dir_hr)
#     imgs_cl = list_images(dir_classical)
#     imgs_dp = list_images(dir_deep)

#     names = sorted(set(imgs_hr) & set(imgs_cl) & set(imgs_dp))
#     if not names:
#         print("No common filenames found across the three folders.")
#         sys.exit(0)

#     ensure_outdir(out_dir)
#     print(f"Found {len(names)} images. Writing plots to: {out_dir}")

#     # aesthetic settings: line styles to match the screenshot
#     line_styles = {
#         "HR": dict(ls="-", lw=2.0),
#         "Classical": dict(ls="--", lw=2.0),
#         "Deep": dict(ls=":", lw=2.5),
#     }

#     nbins = 300  # x-axis goes 0..300 like the figure
#     for name in names:
#         # load + resize to smallest among the three
#         A = load_gray(imgs_hr[name])
#         B = load_gray(imgs_cl[name])
#         C = load_gray(imgs_dp[name])
#         hmin, wmin = smallest_shape([A, B, C])
#         A = resize_to((hmin, wmin), A)
#         B = resize_to((hmin, wmin), B)
#         C = resize_to((hmin, wmin), C)

#         # spectra
#         xA, yA = radial_energy_distribution(A, nbins=nbins)
#         xB, yB = radial_energy_distribution(B, nbins=nbins)
#         xC, yC = radial_energy_distribution(C, nbins=nbins)

#         # figure
#         fig, ax = plt.subplots(figsize=(9, 7))

#         # shaded frequency bands (as fractions of radius range)
#         xmax = nbins - 1
#         lo_end = int(0.10 * xmax)
#         mid_end = int(0.40 * xmax)

#         ax.axvspan(0, lo_end, alpha=0.10)
#         ax.axvspan(lo_end, mid_end, alpha=0.08)
#         ax.axvspan(mid_end, xmax, alpha=0.06)

#         # curves (HR solid, Classical dashed, Deep dotted)
#         ax.plot(xA, yA, label="HR", **line_styles["HR"])
#         ax.plot(xB, yB, label="Classical", **line_styles["Classical"])
#         ax.plot(xC, yC, label="Deep", **line_styles["Deep"])

#         # axes/labels
#         ax.set_yscale("log")
#         ax.set_xlim(0, xmax)
#         ax.set_xlabel("Frequency radius")
#         ax.set_ylabel("Normalized power")

#         # title: "Normalized Power — <stem>"
#         stem = Path(name).stem
#         ax.set_title(f"Normalized Power — {stem}")

#         # legend: include band labels + line labels
#         band_handles = [
#             Patch(alpha=0.10, label="Low (0–10%)"),
#             Patch(alpha=0.08, label="Mid (10–40%)"),
#             Patch(alpha=0.06, label="High (40–100%)"),
#         ]
#         leg1 = ax.legend(handles=band_handles, loc="upper right")
#         ax.add_artist(leg1)
#         ax.legend(loc="upper right", framealpha=1.0, bbox_to_anchor=(1.0, 0.86))

#         ax.grid(True, which="both", ls=":", alpha=0.4)
#         fig.tight_layout()

#         out_path = out_dir / f"{stem}_normalized_power.png"
#         fig.savefig(out_path, dpi=160)
#         plt.close(fig)

#     print("Done.")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Make "Normalized Power" plots with colored frequency bands and larger fonts.

Usage:
  python make_normalized_power_plots.py /path/HR /path/Classical /path/Deep /path/output

Notes:
- Folders must contain images with the SAME filenames.
- Each triplet is resized to the smallest HxW before FFT so the spectra align.
- Output: <name>_normalized_power.png in the output folder.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# Config
# ----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Font sizes (bigger, cleaner)
plt.rcParams.update({
    "figure.titlesize": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# Line styles to match your screenshot’s intent
LINE_STYLES = {
    "HR":         dict(ls="-",  lw=2.4),
    "Classical":  dict(ls="--", lw=2.4),
    "Deep":       dict(ls=":",  lw=2.8),
}

# Colored frequency bands (light colors)
BAND_FRAC_EDGES = (0.00, 0.10, 0.40, 1.00)  # low: 0–10%, mid: 10–40%, high: 40–100%
BAND_COLORS = {
    "Low (0–10%)":   ("#a6cee3", 0.25),  # light blue
    "Mid (10–40%)":  ("#b2df8a", 0.18),  # light green
    "High (40–100%)":("#fdbf6f", 0.12),  # light orange
}

# ----------------------------
# Helpers
# ----------------------------
def list_images(folder: Path):
    return {p.name: p for p in folder.iterdir()
            if p.suffix.lower() in IMG_EXTS and p.is_file()}

def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def resize_to(shape_hw, arr: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    if arr.shape == (h, w):
        return arr
    return np.asarray(
        Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
        dtype=np.float32
    ) / 255.0

def smallest_shape(arrays):
    shapes = [(a.shape[0], a.shape[1]) for a in arrays]
    return sorted(shapes, key=lambda s: (s[0]*s[1], s[0], s[1]))[0]

def radial_energy_distribution(img: np.ndarray, nbins: int = 300):
    """
    Return (radius_index 0..nbins-1, normalized mean power per radius bin).
    We bin by radius in the FFT magnitude (fftshifted), then normalize to unit area.
    """
    h, w = img.shape

    # 2D FFT power spectrum, DC at center
    F = np.fft.fft2(img)
    P = np.abs(F)**2
    P = np.fft.fftshift(P)

    # radius per pixel (in pixels of the FFT grid, not cycles/pixel)
    yy, xx = np.indices((h, w))
    cy, cx = (h-1)/2.0, (w-1)/2.0
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    # max possible radius (corner)
    rmax = r.max()

    # bin edges from 0..rmax mapped to nbins
    bins = np.linspace(0, rmax, nbins + 1)
    ridx = np.digitize(r.ravel(), bins) - 1
    ridx = np.clip(ridx, 0, nbins - 1)

    # mean power per radius ring
    power_sum = np.bincount(ridx, weights=P.ravel(), minlength=nbins).astype(np.float64)
    count = np.bincount(ridx, minlength=nbins).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_power = np.where(count > 0, power_sum / count, 0.0)

    # normalize to unit area so curves are directly comparable
    s = mean_power.sum()
    if s > 0:
        mean_power /= s

    radius_idx = np.arange(nbins, dtype=np.float64)
    return radius_idx, mean_power

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)

    dir_hr = Path(sys.argv[1]).expanduser().resolve()
    dir_classical = Path(sys.argv[2]).expanduser().resolve()
    dir_deep = Path(sys.argv[3]).expanduser().resolve()
    out_dir = Path(sys.argv[4]).expanduser().resolve()

    for d in [dir_hr, dir_classical, dir_deep]:
        if not d.exists() or not d.is_dir():
            print(f"Input folder does not exist or is not a directory: {d}")
            sys.exit(2)

    imgs_hr = list_images(dir_hr)
    imgs_cl = list_images(dir_classical)
    imgs_dp = list_images(dir_deep)

    names = sorted(set(imgs_hr) & set(imgs_cl) & set(imgs_dp))
    if not names:
        print("No common filenames found across the three folders.")
        sys.exit(0)

    ensure_outdir(out_dir)
    print(f"Found {len(names)} images. Writing plots to: {out_dir}")

    nbins = 300  # x-axis is 0..299 (bin indices)
    for name in names:
        # load + resize to smallest among the three
        A = load_gray(imgs_hr[name])
        B = load_gray(imgs_cl[name])
        C = load_gray(imgs_dp[name])
        hmin, wmin = smallest_shape([A, B, C])
        A = resize_to((hmin, wmin), A)
        B = resize_to((hmin, wmin), B)
        C = resize_to((hmin, wmin), C)

        # spectra
        xA, yA = radial_energy_distribution(A, nbins=nbins)
        xB, yB = radial_energy_distribution(B, nbins=nbins)
        xC, yC = radial_energy_distribution(C, nbins=nbins)

        # figure
        fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=140)

        # shaded frequency bands (as fractions of radius range)
        xmax = nbins - 1
        f0, f1, f2, f3 = BAND_FRAC_EDGES
        lo_end  = int(f1 * xmax)
        mid_end = int(f2 * xmax)

        # draw colored spans
        (c_low, a_low)   = BAND_COLORS["Low (0–10%)"]
        (c_mid, a_mid)   = BAND_COLORS["Mid (10–40%)"]
        (c_high, a_high) = BAND_COLORS["High (40–100%)"]
        ax.axvspan(0, lo_end,  facecolor=c_low,  alpha=a_low,  linewidth=0)
        ax.axvspan(lo_end, mid_end, facecolor=c_mid,  alpha=a_mid,  linewidth=0)
        ax.axvspan(mid_end, xmax,  facecolor=c_high, alpha=a_high, linewidth=0)

        # curves (HR solid, Classical dashed, Deep dotted)
        ax.plot(xA, yA, label="HR", **LINE_STYLES["HR"])
        ax.plot(xB, yB, label="Classical", **LINE_STYLES["Classical"])
        ax.plot(xC, yC, label="Deep", **LINE_STYLES["Deep"])

        # axes/labels
        ax.set_yscale("log")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Frequency radius (bin index)")
        ax.set_ylabel("Normalized power (radial mean)")

        # title: "Normalized Power — <stem>"
        stem = Path(name).stem
        ax.set_title(f"Normalized Power — {stem}")

        # legends: one for bands, one for lines
        band_handles = [
            Patch(facecolor=c_low,  alpha=a_low,  label="Low (0–10%)"),
            Patch(facecolor=c_mid,  alpha=a_mid,  label="Mid (10–40%)"),
            Patch(facecolor=c_high, alpha=a_high, label="High (40–100%)"),
        ]
        leg_bands = ax.legend(handles=band_handles, loc="upper right", framealpha=1.0)
        ax.add_artist(leg_bands)
        ax.legend(title="Methods", loc="upper right", framealpha=1.0, bbox_to_anchor=(1.0, 0.84))

        ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()

        out_path = out_dir / f"{stem}_normalized_power.png"
        fig.savefig(out_path)
        plt.close(fig)

    print("Done.")

if __name__ == "__main__":
    main()
