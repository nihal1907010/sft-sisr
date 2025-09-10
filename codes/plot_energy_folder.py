import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PIL import Image

# ========== User settings ==========
GT_DIR        = Path("data/Manga109/HR")                         # ground truth images
CLASSICAL_DIR = Path("data/Manga109/upscaled_bilinear_lanczos")  # classical upscaled images
DEEP_DIR      = Path("data/Manga109/upscaled_bilinear_model")    # deep-learning upscaled images
OUT_DIR       = Path("data/Manga109/comparison")                 # where plots will be saved

NUM_BINS = 1000
FONTSIZE = 16
USE_LOG_Y = True
NORMALIZE_POWER = True
SHADE_REGIONS = True
# ===================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def to_gray(arr):
    if arr.ndim == 3:
        return 0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]
    return arr

def radial_power_profile(img_array: np.ndarray, num_bins: int = 1000):
    img = np.asarray(to_gray(img_array), dtype=np.float32)
    H, W = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    P = (np.abs(F) ** 2).astype(np.float64)

    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_int = r.astype(np.int32)
    r_max = int(np.floor(r.max()))
    radial_sum = np.bincount(r_int.ravel(), weights=P.ravel(), minlength=r_max + 1)
    radial_count = np.bincount(r_int.ravel(), minlength=r_max + 1)
    radial_mean = radial_sum / np.maximum(radial_count, 1)

    r_src = np.arange(len(radial_mean), dtype=np.float64)
    r_tgt = np.linspace(0, r_src[-1], num_bins, dtype=np.float64)
    power = np.interp(r_tgt, r_src, radial_mean)

    if NORMALIZE_POWER and power.max() > 0:
        power = power / power.max()
    return r_tgt, power

def load_image(path: Path):
    return np.array(Image.open(path))

def stemmap(folder: Path):
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS}

# ---- gather common names across all three folders ----
gt_map  = stemmap(GT_DIR)
cl_map  = stemmap(CLASSICAL_DIR)
dp_map  = stemmap(DEEP_DIR)
common  = sorted(set(gt_map) & set(cl_map) & set(dp_map))

if not common:
    print("No matching image stems across all three folders.")
    raise SystemExit

print(f"Found {len(common)} matching images.")

# ---- accumulate for averages ----
per_image = []
global_max_r = np.inf

for name in common:
    img_gt  = load_image(gt_map[name])
    img_cl  = load_image(cl_map[name])
    img_dp  = load_image(dp_map[name])

    r_gt, p_gt = radial_power_profile(img_gt, NUM_BINS)
    r_cl, p_cl = radial_power_profile(img_cl, NUM_BINS)
    r_dp, p_dp = radial_power_profile(img_dp, NUM_BINS)

    per_max_r = min(r_gt.max(), r_cl.max(), r_dp.max())
    global_max_r = min(global_max_r, per_max_r)

    per_image.append((name, (r_gt, p_gt), (r_cl, p_cl), (r_dp, p_dp), per_max_r))

# global common radius grid
r_common = np.linspace(0, global_max_r, NUM_BINS, dtype=np.float64)
sum_gt = np.zeros_like(r_common)
sum_cl = np.zeros_like(r_common)
sum_dp = np.zeros_like(r_common)
count  = 0

# ---- per-image plots ----
for name, (r_gt, p_gt), (r_cl, p_cl), (r_dp, p_dp), per_max_r in per_image:
    p_gt_i = np.interp(r_common, r_gt, p_gt)
    p_cl_i = np.interp(r_common, r_cl, p_cl)
    p_dp_i = np.interp(r_common, r_dp, p_dp)

    sum_gt += p_gt_i
    sum_cl += p_cl_i
    sum_dp += p_dp_i
    count  += 1

    plt.figure(figsize=(10, 6))
    max_r = per_max_r

    if SHADE_REGIONS:
        low_end = 0.10 * max_r
        mid_end = 0.40 * max_r
        plt.axvspan(0, low_end,      color="red",    alpha=0.12, label="Low freq")
        plt.axvspan(low_end, mid_end, color="green",  alpha=0.12, label="Mid freq")
        plt.axvspan(mid_end, max_r,   color="orange", alpha=0.12, label="High freq")

    mask_gt = r_gt <= max_r
    mask_cl = r_cl <= max_r
    mask_dp = r_dp <= max_r

    plt.plot(r_gt[mask_gt], p_gt[mask_gt], label="GT",        lw=2)
    plt.plot(r_cl[mask_cl], p_cl[mask_cl], label="Classical", lw=2, ls="--")
    plt.plot(r_dp[mask_dp], p_dp[mask_dp], label="Model",      lw=2, ls=":")

    if USE_LOG_Y:
        plt.yscale("log")

    plt.xlabel("Fourier frequency radius (pixels)", fontsize=FONTSIZE)
    plt.ylabel("Normalized power" if NORMALIZE_POWER else "Power", fontsize=FONTSIZE)
    plt.title(f"Radial Power Spectrum Comparison — {name}", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE-2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    base = OUT_DIR / f"{name}_power_compare"
    plt.savefig(base.with_suffix(".png"), dpi=150)
    plt.savefig(base.with_suffix(".pdf"))
    plt.close()

# ---- average plot & CSV ----
avg_gt = sum_gt / max(count, 1)
avg_cl = sum_cl / max(count, 1)
avg_dp = sum_dp / max(count, 1)

# Save the numeric data used for the average plot
avg_df = pd.DataFrame({
    "freq_radius": r_common,
    "gt_avg":      avg_gt,
    "classical_avg": avg_cl,
    "deep_avg":      avg_dp,
})
avg_df.to_csv(OUT_DIR / "AVERAGE_power_compare.csv", index=False)

# Plot and save the average curves
plt.figure(figsize=(10, 6))
max_r = r_common.max()

if SHADE_REGIONS:
    low_end = 0.10 * max_r
    mid_end = 0.40 * max_r
    plt.axvspan(0, low_end,      color="red",    alpha=0.12, label="Low freq")
    plt.axvspan(low_end, mid_end, color="green",  alpha=0.12, label="Mid freq")
    plt.axvspan(mid_end, max_r,   color="orange", alpha=0.12, label="High freq")

plt.plot(r_common, avg_gt, label="GT (avg)",        lw=2)
plt.plot(r_common, avg_cl, label="Classical (avg)", lw=2, ls="--")
plt.plot(r_common, avg_dp, label="Model (avg)",      lw=2, ls=":")

if USE_LOG_Y:
    plt.yscale("log")

plt.xlabel("Fourier frequency radius (pixels)", fontsize=FONTSIZE)
plt.ylabel("Normalized power" if NORMALIZE_POWER else "Power", fontsize=FONTSIZE)
plt.title("Radial Power Spectrum Comparison — AVERAGE", fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.legend(fontsize=FONTSIZE-2)
plt.grid(True, alpha=0.3)
plt.tight_layout()

avg_base = OUT_DIR / "AVERAGE_power_compare"
plt.savefig(avg_base.with_suffix(".png"), dpi=150)
plt.savefig(avg_base.with_suffix(".pdf"))
plt.close()

print(f"Saved per-image plots (PNG+PDF), average plot (PNG+PDF), and average CSV to: {OUT_DIR.resolve()}")
