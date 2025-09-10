import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========== User settings ==========
GT_CSV_DIR = Path("datasets/Urban100_HR_Energy_Distribution")        # ground truth CSVs
UPSCALE_CSV_DIR = Path("datasets/Urban100_Lanczos_Upsampled")   # upscaled CSVs
OUT_DIR = Path("datasets/Classical_Energy_Comparison")            # results go here
SAVE_OVERLAY_PLOTS = True
# ===================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect CSVs
gt_csvs = {f.stem: f for f in GT_CSV_DIR.glob("*.csv")}
up_csvs = {f.stem.replace("_main", ""): f for f in UPSCALE_CSV_DIR.glob("*.csv")}
common = sorted(set(gt_csvs) & set(up_csvs))

results = []

for name in common:
    gt = pd.read_csv(gt_csvs[name])
    up = pd.read_csv(up_csvs[name])

    # align by min length (just in case dimensions differ)
    n = min(len(gt), len(up))
    freqs = gt["freq_radius"].values[:n]
    gt_power = gt["radial_mean_power"].values[:n]
    up_power = up["radial_mean_power"].values[:n]
    gt_cum = gt["cumulative_energy_fraction"].values[:n]
    up_cum = up["cumulative_energy_fraction"].values[:n]

    # normalize power for fair comparison (optional, depends on use case)
    gt_power /= gt_power.sum()
    up_power /= up_power.sum()

    # simple errors
    mse_power = np.mean((gt_power - up_power)**2)
    mse_cum = np.mean((gt_cum - up_cum)**2)

    # band-specific errors
    max_r = freqs.max()
    def band_mse(frac_low, frac_high, arr1, arr2):
        mask = (freqs >= frac_low * max_r) & (freqs < frac_high * max_r)
        if mask.sum() == 0:
            return np.nan
        return np.mean((arr1[mask] - arr2[mask])**2)

    mse_low  = band_mse(0.0, 0.1, gt_power, up_power)
    mse_mid  = band_mse(0.1, 0.4, gt_power, up_power)
    mse_high = band_mse(0.4, 1.0, gt_power, up_power)

    results.append({
        "image": name,
        "mse_power": mse_power,
        "mse_cum": mse_cum,
        "mse_low": mse_low,
        "mse_mid": mse_mid,
        "mse_high": mse_high
    })

    # Overlay plots
    if SAVE_OVERLAY_PLOTS:
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.title(f"Power Spectrum: {name}")
        plt.plot(freqs, gt_power, label="GT", lw=2)
        plt.plot(freqs, up_power, label="Upscaled", lw=2, ls="--")
        plt.yscale("log")
        plt.xlabel("Frequency radius")
        plt.ylabel("Normalized power")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1,2,2)
        plt.title("Cumulative Energy")
        plt.plot(freqs, gt_cum, label="GT", lw=2)
        plt.plot(freqs, up_cum, label="Upscaled", lw=2, ls="--")
        plt.xlabel("Frequency radius")
        plt.ylabel("Cumulative energy fraction")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{name}_compare.png", dpi=150)
        plt.close()

# Save summary
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "frequency_comparison_summary.csv", index=False)
print("Saved comparison results to", OUT_DIR)
