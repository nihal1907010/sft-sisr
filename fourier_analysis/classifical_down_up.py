import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# User settings (edit me)
# =========================
HR_CSV_DIR        = Path("datasets/Urban100_HR_Energy_Distribution")        # ground truth CSVs
CLASSIC_CSV_DIR   = Path("datasets/Urban100_Classical_Energy_Distribution")   # classical upscaled CSVs
DEEP_CSV_DIR      = Path("datasets/Urban100_Upscaled_Energy_Distribution")      # deep-learning upscaled CSVs
OUT_DIR           = Path("outputs/freq_compare_all")     # where to save results

# If filenames differ (e.g., "img001_main.csv"), strip these suffixes before matching
CLASSIC_STRIP_SUFFIX = ""         # e.g., "_lanczos"
DEEP_STRIP_SUFFIX    = "_main"    # set "" if none

MAKE_PER_IMAGE_PLOTS = True
# =========================


# ---------- Helpers ----------
def load_curve_csv(path: Path):
    """Return (freqs, power, cumulative) as numpy arrays using only the three required columns."""
    df = pd.read_csv(path, usecols=["freq_radius", "radial_mean_power", "cumulative_energy_fraction"])
    f = df["freq_radius"].to_numpy()
    p = df["radial_mean_power"].to_numpy(dtype=float)
    c = df["cumulative_energy_fraction"].to_numpy(dtype=float)
    return f, p, c

def normalize_power(p):
    s = np.sum(p)
    if s <= 0 or not np.isfinite(s):
        return np.zeros_like(p)
    return p / s

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def cosine_sim(a, b):
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

def band_mask(freqs, lo_frac, hi_frac):
    if len(freqs) == 0:
        return np.zeros(0, dtype=bool)
    fmax = max(freqs.max(), 1.0)
    return (freqs >= lo_frac * fmax) & (freqs < hi_frac * fmax)

def k_at_cum(cum, freqs, target=0.5):
    """Radius where cumulative crosses target (linear interp)."""
    if len(cum) == 0:
        return np.nan
    idx = np.searchsorted(cum, target, side="left")
    if idx == 0:
        return float(freqs[0])
    if idx >= len(cum):
        return float(freqs[-1])
    x0, x1 = cum[idx-1], cum[idx]
    f0, f1 = freqs[idx-1], freqs[idx]
    if x1 == x0:
        return float(f1)
    t = (target - x0) / (x1 - x0)
    return float(f0 + t * (f1 - f0))

def collect(folder: Path, strip_suffix=""):
    """Map base stem (after stripping suffix) -> path."""
    mapping = {}
    for p in folder.glob("*.csv"):  # make this .rglob("*.csv") if nested
        stem = p.stem
        if strip_suffix and stem.endswith(strip_suffix):
            stem = stem[: -len(strip_suffix)]
        mapping[stem] = p
    return mapping

def shaded_bands(ax, freqs):
    """Shade low/mid/high bands for readability."""
    if len(freqs) == 0:
        return
    fmax = freqs.max()
    ax.axvspan(0.0, 0.10 * fmax, alpha=0.08, label="Low (0–10%)")
    ax.axvspan(0.10 * fmax, 0.40 * fmax, alpha=0.06, label="Mid (10–40%)")
    ax.axvspan(0.40 * fmax, 1.00 * fmax, alpha=0.04, label="High (40–100%)")

def make_overlay_plot(name, freqs, p_hr_n, p_classic_n, p_deep_n, c_hr, c_classic, c_deep, out_path):
    plt.figure(figsize=(14, 6))

    # Power (normalized) with log y
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(f"Normalized Power — {name}")
    shaded_bands(ax1, freqs)
    ax1.plot(freqs, p_hr_n, label="HR", linewidth=2)
    ax1.plot(freqs, p_classic_n, label="Classical", linestyle="--", linewidth=2)
    ax1.plot(freqs, p_deep_n, label="Deep", linestyle=":", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Frequency radius")
    ax1.set_ylabel("Normalized power")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Cumulative energy
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Cumulative Energy")
    shaded_bands(ax2, freqs)
    ax2.plot(freqs, c_hr, label="HR", linewidth=2)
    ax2.plot(freqs, c_classic, label="Classical", linestyle="--", linewidth=2)
    ax2.plot(freqs, c_deep, label="Deep", linestyle=":", linewidth=2)
    ax2.set_xlabel("Frequency radius")
    ax2.set_ylabel("Cumulative energy fraction")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------- Main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs_dir = OUT_DIR / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Collect & match
    hr_map      = collect(HR_CSV_DIR, strip_suffix="")
    classic_map = collect(CLASSIC_CSV_DIR, strip_suffix=CLASSIC_STRIP_SUFFIX)
    deep_map    = collect(DEEP_CSV_DIR, strip_suffix=DEEP_STRIP_SUFFIX)

    common = sorted(set(hr_map) & set(classic_map) & set(deep_map))
    if not common:
        print("No matched CSV stems across HR, Classical, and Deep folders.")
        return

    rows = []

    for i, stem in enumerate(common, 1):
        try:
            f_hr, p_hr, c_hr = load_curve_csv(hr_map[stem])
            f_c , p_c , c_c  = load_curve_csv(classic_map[stem])
            f_d , p_d , c_d  = load_curve_csv(deep_map[stem])

            # Align by shortest length (assume same freq grid by index)
            n = min(len(f_hr), len(f_c), len(f_d))
            f = f_hr[:n]
            p_hr, p_c, p_d = p_hr[:n], p_c[:n], p_d[:n]
            c_hr, c_c, c_d = c_hr[:n], c_c[:n], c_d[:n]

            # Normalize power (shape comparison)
            p_hr_n = normalize_power(p_hr)
            p_c_n  = normalize_power(p_c)
            p_d_n  = normalize_power(p_d)

            # Global metrics
            power_mse_classic = mse(p_hr_n, p_c_n)
            power_mse_deep    = mse(p_hr_n, p_d_n)
            cos_classic       = cosine_sim(p_hr_n, p_c_n)
            cos_deep          = cosine_sim(p_hr_n, p_d_n)
            cum_mse_classic   = mse(c_hr, c_c)
            cum_mse_deep      = mse(c_hr, c_d)

            # Banded MSEs (on normalized power)
            mask_low  = band_mask(f, 0.00, 0.10)
            mask_mid  = band_mask(f, 0.10, 0.40)
            mask_high = band_mask(f, 0.40, 1.00)
            def banded_mse(a, b, m):
                if m.sum() == 0: return np.nan
                return float(np.mean((a[m] - b[m])**2))
            low_mse_c  = banded_mse(p_hr_n, p_c_n, mask_low)
            mid_mse_c  = banded_mse(p_hr_n, p_c_n, mask_mid)
            high_mse_c = banded_mse(p_hr_n, p_c_n, mask_high)
            low_mse_d  = banded_mse(p_hr_n, p_d_n, mask_low)
            mid_mse_d  = banded_mse(p_hr_n, p_d_n, mask_mid)
            high_mse_d = banded_mse(p_hr_n, p_d_n, mask_high)

            # k50 / k80 (radius at 50% / 80% cumulative energy)
            k50_hr = k_at_cum(c_hr, f, 0.50)
            k80_hr = k_at_cum(c_hr, f, 0.80)
            k50_c  = k_at_cum(c_c , f, 0.50)
            k80_c  = k_at_cum(c_c , f, 0.80)
            k50_d  = k_at_cum(c_d , f, 0.50)
            k80_d  = k_at_cum(c_d , f, 0.80)
            dk50_c = k50_c - k50_hr
            dk80_c = k80_c - k80_hr
            dk50_d = k50_d - k50_hr
            dk80_d = k80_d - k80_hr

            # Per-image plot
            if MAKE_PER_IMAGE_PLOTS:
                make_overlay_plot(
                    stem, f, p_hr_n, p_c_n, p_d_n, c_hr, c_c, c_d,
                    figs_dir / f"{stem}_overlay.png"
                )

            # Append metrics row
            rows.append({
                "image": stem,
                # overall power shape similarity
                "power_mse_classic": power_mse_classic,
                "power_mse_deep": power_mse_deep,
                "cosine_power_classic": cos_classic,
                "cosine_power_deep": cos_deep,
                # cumulative agreement
                "cum_mse_classic": cum_mse_classic,
                "cum_mse_deep": cum_mse_deep,
                # band MSEs
                "low_mse_classic": low_mse_c,
                "mid_mse_classic": mid_mse_c,
                "high_mse_classic": high_mse_c,
                "low_mse_deep": low_mse_d,
                "mid_mse_deep": mid_mse_d,
                "high_mse_deep": high_mse_d,
                # k50 / k80 deltas
                "dk50_classic": dk50_c, "dk80_classic": dk80_c,
                "dk50_deep": dk50_d,   "dk80_deep": dk80_d,
            })

            print(f"[{i}/{len(common)}] {stem}: power MSE (C/D) = {power_mse_classic:.3e}/{power_mse_deep:.3e} | "
                  f"cum MSE (C/D) = {cum_mse_classic:.3e}/{cum_mse_deep:.3e}")

        except Exception as e:
            print(f"Failed on {stem}: {e}")

    # Save per-image metrics CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_image_metrics.csv", index=False)

    # Dataset-level summary
    def safe_mean(col):
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    summary = {
        # Lower is better for MSE; higher is better for cosine
        "mean_power_mse_classic": safe_mean("power_mse_classic"),
        "mean_power_mse_deep": safe_mean("power_mse_deep"),
        "mean_cosine_power_classic": safe_mean("cosine_power_classic"),
        "mean_cosine_power_deep": safe_mean("cosine_power_deep"),
        "mean_cum_mse_classic": safe_mean("cum_mse_classic"),
        "mean_cum_mse_deep": safe_mean("cum_mse_deep"),
        "mean_low_mse_classic": safe_mean("low_mse_classic"),
        "mean_mid_mse_classic": safe_mean("mid_mse_classic"),
        "mean_high_mse_classic": safe_mean("high_mse_classic"),
        "mean_low_mse_deep": safe_mean("low_mse_deep"),
        "mean_mid_mse_deep": safe_mean("mid_mse_deep"),
        "mean_high_mse_deep": safe_mean("high_mse_deep"),
        "n_images": len(df),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "dataset_summary.csv", index=False)

    # -------- Aggregate visualizations --------
    # Bar chart: mean MSE (power & cumulative)
    plt.figure(figsize=(8,5))
    x = np.arange(2)
    bar1 = [summary["mean_power_mse_deep"], summary["mean_power_mse_classic"]]
    bar2 = [summary["mean_cum_mse_deep"],   summary["mean_cum_mse_classic"]]
    plt.bar(x - 0.15, bar1, width=0.3, label="Power MSE")
    plt.bar(x + 0.15, bar2, width=0.3, label="Cumulative MSE")
    plt.xticks(x, ["Deep", "Classic"])
    plt.title("Dataset-level Means (lower is better)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_bar_mse.png", dpi=160)
    plt.close()

    # Boxplots for banded high-frequency MSEs
    plt.figure(figsize=(7,5))
    data = [df["high_mse_deep"].dropna(), df["high_mse_classic"].dropna()]
    plt.boxplot(data, labels=["Deep", "Classic"])
    plt.title("High-frequency MSE (normalized power)")
    plt.ylabel("MSE")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplot_high_band_mse.png", dpi=160)
    plt.close()

    print(f"Done. Per-image plots -> {OUT_DIR/'figs'}, metrics -> {OUT_DIR}.")


if __name__ == "__main__":
    main()
