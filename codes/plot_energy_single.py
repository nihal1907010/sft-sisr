import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ========== User settings ==========
IMG_PATH = Path("data/Urban100/HR/img001.png")  # <- replace with your image path
NUM_BINS = 1000
USE_LOG_Y = True
NORMALIZE_POWER = True
FONTSIZE = 16
# ===================================

def radial_power_profile(img_array: np.ndarray, num_bins: int = 1000):
    if img_array.ndim == 3:
        img_array = 0.2989 * img_array[..., 0] + 0.5870 * img_array[..., 1] + 0.1140 * img_array[..., 2]
    img = np.asarray(img_array, dtype=np.float32)

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
    with np.errstate(invalid='ignore', divide='ignore'):
        radial_mean = radial_sum / np.maximum(radial_count, 1)

    r_source = np.arange(len(radial_mean), dtype=np.float64)
    r_target = np.linspace(0, r_source[-1], num_bins, dtype=np.float64)
    power_interp = np.interp(r_target, r_source, radial_mean)

    if NORMALIZE_POWER and power_interp.max() > 0:
        power_interp = power_interp / power_interp.max()

    return r_target, power_interp

# ---- Load image & compute profile ----
img = np.array(Image.open(IMG_PATH))
radii, norm_power = radial_power_profile(img, NUM_BINS)

# ---- Plot ----
plt.figure(figsize=(10, 6))
plt.plot(radii, norm_power, linewidth=1.6, color="blue", label="Power spectrum")

if USE_LOG_Y:
    plt.yscale("log")

# Define regions as fractions of max radius
max_r = radii.max()
low_end   = 0.1 * max_r
mid_end   = 0.4 * max_r

# Shaded regions
plt.axvspan(0, low_end, color="red", alpha=0.15, label="Low freq")
plt.axvspan(low_end, mid_end, color="green", alpha=0.15, label="Mid freq")
plt.axvspan(mid_end, max_r, color="orange", alpha=0.15, label="High freq")

# Labels
plt.xlabel("Fourier frequency radius (pixels)", fontsize=FONTSIZE)
plt.ylabel("Normalized power" if NORMALIZE_POWER else "Power", fontsize=FONTSIZE)
plt.title(f"Radial Power Spectrum ({IMG_PATH.name})", fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.legend(fontsize=FONTSIZE-2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
