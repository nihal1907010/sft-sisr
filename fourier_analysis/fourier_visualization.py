import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# =========================
# User settings (edit me)
# =========================
INPUT_DIR   = Path("datasets/Urban100_Lanczos_Upsampled")  # folder of images
OUTPUT_DIR  = Path("datasets/Urban100_Classical_Energy_Distribution")         # where to save results
PATTERN     = "*"                                # e.g. "*.png" or "*.jpg"
SAVE_CSV    = True                               # also save raw curves as CSV
SHOW_CUMUL  = True                               # add cumulative energy curve
LOG_Y       = True                               # log y-axis for power
# =========================

def load_grayscale(path: Path) -> np.ndarray:
    """Load an image as float32 grayscale in [0,1]."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("RGB")
    img = ImageOps.grayscale(img)
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

def radial_power_spectrum(img: np.ndarray):
    """
    Compute (freq_radius, radial_mean_power, cumulative_energy_fraction).
    """
    H, W = img.shape
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    power = (np.abs(Fshift) ** 2).astype(np.float64)

    cy, cx = H // 2, W // 2
    Y, X = np.indices((H, W))
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    R_int = R.astype(np.int64)

    radial_sum = np.bincount(R_int.ravel(), weights=power.ravel())
    radial_count = np.bincount(R_int.ravel())
    radial_mean = np.zeros_like(radial_sum, dtype=np.float64)
    mask = radial_count > 0
    radial_mean[mask] = radial_sum[mask] / radial_count[mask]

    total_energy = power.sum()
    cumulative = np.cumsum(radial_sum)
    cum_frac = cumulative / (total_energy + 1e-12)

    freqs = np.arange(len(radial_mean))
    return freqs, radial_mean, cum_frac

def make_plot(img: np.ndarray, freqs, radial_mean, cum_frac, title: str,
              show_cumulative: bool = True, log_y: bool = True):
    """Create the 3-panel figure and return it."""
    F = np.fft.fftshift(np.fft.fft2(img))
    mag_vis = np.log1p(np.abs(F))

    fig = plt.figure(figsize=(16, 4.8))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Original")
    ax1.imshow(img, cmap="gray")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("FFT Magnitude (log)")
    ax2.imshow(mag_vis, cmap="gray")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Frequency Energy Distribution")
    ax3.plot(freqs, radial_mean, label="Radial mean power")
    if log_y:
        ax3.set_yscale("log")
    ax3.set_xlabel("Frequency radius (pixels)")
    ax3.set_ylabel("Power")
    ax3.grid(True, alpha=0.3)

    if show_cumulative:
        ax4 = ax3.twinx()
        ax4.plot(freqs, cum_frac, linestyle="--", label="Cumulative energy")
        ax4.set_ylabel("Cumulative energy")
        ax4.set_ylim(0, 1.0)
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    return fig

def save_curves_csv(csv_path: Path, freqs, radial_mean, cum_frac):
    data = np.column_stack([freqs, radial_mean, cum_frac])
    header = "freq_radius,radial_mean_power,cumulative_energy_fraction"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.10g")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    images = sorted([p for p in INPUT_DIR.rglob(PATTERN) if p.suffix.lower() in exts])

    if not images:
        print(f"No images found in {INPUT_DIR} with pattern '{PATTERN}'.")
        return

    for i, img_path in enumerate(images, 1):
        try:
            img = load_grayscale(img_path)
            freqs, radial_mean, cum_frac = radial_power_spectrum(img)

            # Mirror subfolder structure inside OUTPUT_DIR
            rel = img_path.relative_to(INPUT_DIR)
            out_dir_for_img = OUTPUT_DIR / rel.parent
            out_dir_for_img.mkdir(parents=True, exist_ok=True)

            base = out_dir_for_img / rel.stem
            fig = make_plot(
                img, freqs, radial_mean, cum_frac,
                title=str(rel),
                show_cumulative=SHOW_CUMUL,
                log_y=LOG_Y
            )
            fig.savefig(base.with_suffix(".png"), dpi=160, bbox_inches="tight")
            plt.close(fig)

            if SAVE_CSV:
                save_curves_csv(base.with_suffix(".csv"), freqs, radial_mean, cum_frac)

            print(f"[{i}/{len(images)}] Saved: {base.with_suffix('.png').relative_to(OUTPUT_DIR)}"
                  + (" (+ CSV)" if SAVE_CSV else ""))

        except Exception as e:
            print(f"Failed on {img_path}: {e}")

if __name__ == "__main__":
    main()
