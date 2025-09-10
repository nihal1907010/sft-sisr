# freq_compare_pipeline.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# =========================
# User settings (edit me)
# =========================
HR_DIR         = Path("datasets/Test/Manga109/HR")      # original HR images
DEEP_DIR       = Path("results/3.main/visualization/Manga109")             # deep-learning upscaled images
CLASSIC_IMG_DIR= Path("results/bilinear_lanczos/visualization/Manga109")        # where to save classical upscaled images

# Scale and resampling methods for the classical baseline
SCALE            = 4
DOWNSAMPLE_METHOD= "bilinear"   # options: "nearest","bilinear","bicubic","lanczos","box","hamming"
UPSAMPLE_METHOD  = "lanczos"    # options: "nearest","bilinear","bicubic","lanczos","box","hamming"

# Results (energy CSVs, comparisons, plots)
RESULTS_ROOT   = Path("outputs/Manga109_freq_results")           # all CSVs and comparison plots
PATTERN        = "*"                                    # e.g., "*.png" or "*.jpg"
DEEP_STRIP_SUFFIX = "_main"                                  # set to "_main" etc if the deep filenames have a suffix
CROP_TO_MULT_OF_SCALE = True                            # center-crop HR so W,H divisible by SCALE
MAKE_PER_IMAGE_PLOTS  = True
# =========================


# ---------- Utilities ----------
def pil_resample(name: str):
    nm = name.lower()
    if nm == "nearest": return Image.NEAREST
    if nm == "bilinear": return Image.BILINEAR
    if nm == "bicubic": return Image.BICUBIC
    if nm == "lanczos": return Image.LANCZOS
    if nm == "box": return Image.BOX
    if nm == "hamming": return Image.HAMMING
    raise ValueError(f"Unknown resample: {name}")

def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode == "P":
        img = img.convert("RGB")
    return img

def to_luma01(img: Image.Image) -> np.ndarray:
    """Return Y (luma) in [0,1] as float32."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    if img.mode == "L":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    ycbcr = img.convert("YCbCr")
    y = np.asarray(ycbcr, dtype=np.float32)[..., 0] / 255.0
    return y

def ensure_divisible(img: Image.Image, s: int) -> Image.Image:
    if not CROP_TO_MULT_OF_SCALE: return img
    w, h = img.size
    nw, nh = (w // s) * s, (h // s) * s
    if nw == w and nh == h: return img
    left = (w - nw) // 2
    top  = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))

def classic_down_up(hr_img: Image.Image, scale=4, down="bilinear", up="lanczos") -> Image.Image:
    """Downscale HR by SCALE with 'down' method, then upscale back with 'up' method."""
    hr_img = ensure_divisible(hr_img, scale)
    w, h = hr_img.size
    down_im = hr_img.resize((w // scale, h // scale), resample=pil_resample(down))
    up_im   = down_im.resize((w, h), resample=pil_resample(up))
    return up_im

def radial_power_spectrum(y: np.ndarray):
    """Return (freqs, radial_mean_power, cumulative_energy_fraction)."""
    H, W = y.shape
    F = np.fft.fftshift(np.fft.fft2(y))
    power = (np.abs(F) ** 2).astype(np.float64)

    cy, cx = H // 2, W // 2
    Y, X = np.indices((H, W))
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.int64)

    radial_sum = np.bincount(R.ravel(), weights=power.ravel())
    radial_cnt = np.bincount(R.ravel())
    radial_mean = np.zeros_like(radial_sum, dtype=np.float64)
    m = radial_cnt > 0
    radial_mean[m] = radial_sum[m] / radial_cnt[m]

    total = power.sum()
    cumulative = np.cumsum(radial_sum)
    cum_frac = cumulative / (total + 1e-12)

    freqs = np.arange(len(radial_mean))
    return freqs, radial_mean, cum_frac

def save_energy_csv(csv_path: Path, freqs, radial_mean, cum_frac):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([freqs, radial_mean, cum_frac])
    header = "freq_radius,radial_mean_power,cumulative_energy_fraction"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.10g")

def normalize_power(p):
    s = p.sum()
    return p / (s + 1e-12)

def band_mask(freqs, lo_frac, hi_frac):
    if len(freqs) == 0: return np.zeros(0, dtype=bool)
    fmax = max(float(freqs.max()), 1.0)
    return (freqs >= lo_frac * fmax) & (freqs < hi_frac * fmax)

def k_at_cum(cum, freqs, target=0.5):
    if len(cum) == 0: return np.nan
    idx = np.searchsorted(cum, target, side="left")
    if idx == 0: return float(freqs[0])
    if idx >= len(cum): return float(freqs[-1])
    x0, x1 = cum[idx-1], cum[idx]
    f0, f1 = freqs[idx-1], freqs[idx]
    if x1 == x0: return float(f1)
    t = (target - x0) / (x1 - x0)
    return float(f0 + t * (f1 - f0))

def mse(a, b): return float(np.mean((a - b) ** 2))
def cosine_sim(a, b):
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

def shaded_bands(ax, freqs):
    if len(freqs) == 0: return
    fmax = freqs.max()
    ax.axvspan(0.0, 0.10 * fmax, alpha=0.08, label="Low (0–10%)")
    ax.axvspan(0.10 * fmax, 0.40 * fmax, alpha=0.06, label="Mid (10–40%)")
    ax.axvspan(0.40 * fmax, 1.00 * fmax, alpha=0.04, label="High (40–100%)")

def make_overlay_plot(name, freqs, p_hr_n, p_classic_n, p_deep_n, c_hr, c_classic, c_deep, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))

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


# ---------- Pipeline ----------
def main():
    # Prepare dirs
    CLASSIC_IMG_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "csv" / "hr").mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "csv" / "deep").mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "csv" / "classic").mkdir(parents=True, exist_ok=True)
    figs_dir = RESULTS_ROOT / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Gather HR images
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")
    hr_list = sorted([p for p in HR_DIR.rglob(PATTERN) if p.suffix.lower() in exts])

    # Map deep files by stem (with optional suffix strip)
    deep_map = {}
    for p in DEEP_DIR.rglob(PATTERN):
        if p.suffix.lower() not in exts: continue
        st = p.stem
        if DEEP_STRIP_SUFFIX and st.endswith(DEEP_STRIP_SUFFIX):
            st = st[: -len(DEEP_STRIP_SUFFIX)]
        deep_map[st] = p

    rows = []

    for i, hr_path in enumerate(hr_list, 1):
        try:
            name = hr_path.stem
            if name not in deep_map:
                print(f"[Skip] No deep match for {name}")
                continue

            # --- Create classical baseline image ---
            hr_img = load_image(hr_path)
            hr_img = ensure_divisible(hr_img, SCALE)
            classic_img = classic_down_up(hr_img, SCALE, DOWNSAMPLE_METHOD, UPSAMPLE_METHOD)

            # Save classical image (mirror structure)
            rel = hr_path.relative_to(HR_DIR)
            out_img_path = CLASSIC_IMG_DIR / rel
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            classic_img.save(out_img_path)

            # --- Compute energy CSVs for HR, Classic, Deep (from images) ---
            deep_img = load_image(deep_map[name])
            # Resize deep to HR size if needed
            if deep_img.size != hr_img.size:
                deep_img = deep_img.resize(hr_img.size, resample=Image.LANCZOS)

            for set_name, img in [("hr", hr_img), ("classic", classic_img), ("deep", deep_img)]:
                y = to_luma01(img)
                freqs, radial, cum = radial_power_spectrum(y)
                csv_path = RESULTS_ROOT / "csv" / set_name / f"{name}.csv"
                save_energy_csv(csv_path, freqs, radial, cum)

            # --- Load back curves & compare (per-image) ---
            def load_curve(dir_name):
                df = pd.read_csv(RESULTS_ROOT / "csv" / dir_name / f"{name}.csv")
                return (df["freq_radius"].to_numpy(),
                        df["radial_mean_power"].to_numpy(dtype=float),
                        df["cumulative_energy_fraction"].to_numpy(dtype=float))

            f_hr, p_hr, c_hr = load_curve("hr")
            f_cl, p_cl, c_cl = load_curve("classic")
            f_dp, p_dp, c_dp = load_curve("deep")

            n = min(len(f_hr), len(f_cl), len(f_dp))
            f = f_hr[:n]
            p_hr, p_cl, p_dp = p_hr[:n], p_cl[:n], p_dp[:n]
            c_hr, c_cl, c_dp = c_hr[:n], c_cl[:n], c_dp[:n]

            p_hr_n = normalize_power(p_hr)
            p_cl_n = normalize_power(p_cl)
            p_dp_n = normalize_power(p_dp)

            # Metrics
            m_power_cl = mse(p_hr_n, p_cl_n)
            m_power_dp = mse(p_hr_n, p_dp_n)
            cos_cl = cosine_sim(p_hr_n, p_cl_n)
            cos_dp = cosine_sim(p_hr_n, p_dp_n)
            m_cum_cl = mse(c_hr, c_cl)
            m_cum_dp = mse(c_hr, c_dp)

            low = band_mask(f, 0.0, 0.10); mid = band_mask(f, 0.10, 0.40); high = band_mask(f, 0.40, 1.0)
            def bm(a,b,m): return float(np.mean((a[m]-b[m])**2)) if m.sum() else np.nan
            low_cl, mid_cl, high_cl = bm(p_hr_n,p_cl_n,low), bm(p_hr_n,p_cl_n,mid), bm(p_hr_n,p_cl_n,high)
            low_dp, mid_dp, high_dp = bm(p_hr_n,p_dp_n,low), bm(p_hr_n,p_dp_n,mid), bm(p_hr_n,p_dp_n,high)

            k50_hr = k_at_cum(c_hr, f, 0.50); k80_hr = k_at_cum(c_hr, f, 0.80)
            k50_cl = k_at_cum(c_cl, f, 0.50); k80_cl = k_at_cum(c_cl, f, 0.80)
            k50_dp = k_at_cum(c_dp, f, 0.50); k80_dp = k_at_cum(c_dp, f, 0.80)
            dk50_cl, dk80_cl = k50_cl - k50_hr, k80_cl - k80_hr
            dk50_dp, dk80_dp = k50_dp - k50_hr, k80_dp - k80_hr

            # Per-image overlay plot
            if MAKE_PER_IMAGE_PLOTS:
                make_overlay_plot(name, f, p_hr_n, p_cl_n, p_dp_n, c_hr, c_cl, c_dp,
                                  figs_dir / f"{name}_overlay.png")

            rows.append({
                "image": name,
                "power_mse_classic": m_power_cl, "power_mse_deep": m_power_dp,
                "cosine_power_classic": cos_cl,  "cosine_power_deep": cos_dp,
                "cum_mse_classic": m_cum_cl,     "cum_mse_deep": m_cum_dp,
                "low_mse_classic": low_cl, "mid_mse_classic": mid_cl, "high_mse_classic": high_cl,
                "low_mse_deep": low_dp, "mid_mse_deep": mid_dp, "high_mse_deep": high_dp,
                "dk50_classic": dk50_cl, "dk80_classic": dk80_cl,
                "dk50_deep": dk50_dp,   "dk80_deep": dk80_dp,
            })

            print(f"[{i}/{len(hr_list)}] {name}: Power MSE C/D = {m_power_cl:.3e}/{m_power_dp:.3e}  |  "
                  f"Cum MSE C/D = {m_cum_cl:.3e}/{m_cum_dp:.3e}")

        except Exception as e:
            print(f"Failed on {hr_path}: {e}")

    # Save per-image metrics
    df = pd.DataFrame(rows)
    (RESULTS_ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_ROOT / "metrics" / "per_image_metrics.csv", index=False)

    # Dataset-level summary
    def smean(col): return float(pd.to_numeric(df[col], errors="coerce").mean())
    summary = {
        "mean_power_mse_classic": smean("power_mse_classic"),
        "mean_power_mse_deep": smean("power_mse_deep"),
        "mean_cosine_power_classic": smean("cosine_power_classic"),
        "mean_cosine_power_deep": smean("cosine_power_deep"),
        "mean_cum_mse_classic": smean("cum_mse_classic"),
        "mean_cum_mse_deep": smean("cum_mse_deep"),
        "mean_low_mse_classic": smean("low_mse_classic"),
        "mean_mid_mse_classic": smean("mid_mse_classic"),
        "mean_high_mse_classic": smean("high_mse_classic"),
        "mean_low_mse_deep": smean("low_mse_deep"),
        "mean_mid_mse_deep": smean("mid_mse_deep"),
        "mean_high_mse_deep": smean("high_mse_deep"),
        "mean_dk50_classic": smean("dk50_classic"),
        "mean_dk80_classic": smean("dk80_classic"),
        "mean_dk50_deep": smean("dk50_deep"),
        "mean_dk80_deep": smean("dk80_deep"),
        "n_images": len(df)
    }
    pd.DataFrame([summary]).to_csv(RESULTS_ROOT / "metrics" / "dataset_summary.csv", index=False)

    # Aggregate visualizations
    # Mean MSEs
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
    plt.savefig(RESULTS_ROOT / "metrics" / "summary_bar_mse.png", dpi=160)
    plt.close()

    # High-frequency MSE boxplot
    plt.figure(figsize=(7,5))
    data = [df["high_mse_deep"].dropna(), df["high_mse_classic"].dropna()]
    plt.boxplot(data, labels=["Deep", "Classic"])
    plt.title("High-frequency MSE (normalized power)")
    plt.ylabel("MSE")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "metrics" / "boxplot_high_band_mse.png", dpi=160)
    plt.close()

    print(f"Done.\n- Classical images -> {CLASSIC_IMG_DIR}\n- Energy CSVs -> {RESULTS_ROOT/'csv'}\n"
          f"- Per-image plots -> {RESULTS_ROOT/'figs'}\n- Metrics -> {RESULTS_ROOT/'metrics'}")

if __name__ == "__main__":
    main()
