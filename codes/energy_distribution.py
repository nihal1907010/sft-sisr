import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from datetime import datetime

# -------------------------
# CONFIG (edit these paths)
# -------------------------
GT_DIR = "data/Urban100/HR"
CLASSICAL_DIR = "data/Urban100/upscaled_bilinear_lanczos"
DEEP_DIR = "data/Urban100/upscaled_bilinear_model"

# Output root (a single folder containing plots/ and csv/)
OUTPUT_ROOT = r"./fourier_energy_reports/Urban100"

# Number of frequency bins
NUM_BINS = 1000

# Whether to downscale very large images for speed (None = no cap)
MAX_SIDE = None  # e.g., 1024

# -------------------------
# Helpers
# -------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def to_grayscale_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to float32 grayscale array in [0,1].
    Uses ITU-R BT.601 luma if RGB.
    """
    if img.mode == "L":
        arr = np.asarray(img, dtype=np.float32)
    else:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        # Luma conversion
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    # Normalize to [0,1] if not already
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

def maybe_downscale(img: Image.Image, max_side=None) -> Image.Image:
    if max_side is None:
        return img
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    scale = max_side / s
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.BICUBIC)

def radial_fourier_energy(image_array: np.ndarray, num_bins=1000):
    """
    Compute azimuthally averaged (radial) power spectral density (Fourier energy) as a 1D profile.
    Returns:
      freq_bin_centers: array in [0,1], normalized radial frequency (0 = DC, 1 = Nyquist)
      energy_norm: normalized energy per bin (sums to 1 across all bins)
    """
    # Remove NaNs/Infs if any
    img = np.nan_to_num(image_array, copy=False)

    # 2D FFT (centered) and power spectrum
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2  # power

    h, w = img.shape
    # Frequency coordinates (cycles per pixel)
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)  # radial frequency magnitude (0 to ~0.5*sqrt(2))

    # Normalize radius by the maximum Nyquist in 1D (0.5), not by diagonal
    # We want 0..1 where 1 corresponds to Nyquist (0.5 cycles/pixel).
    # Convert absolute freq magnitude to "normalized radial frequency"
    Rn = R / 0.5  # so 0.5 cycles/pixel -> 1.0

    # Clip to [0,1] range for binning (anything above Nyquist maps to <= sqrt(2) in theory;
    # we cap at 1.0 to only consider up to Nyquist in the radial sense).
    Rn = np.clip(Rn, 0.0, 1.0)

    # Bin edges and centers
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Flatten arrays
    r_flat = Rn.ravel()
    p_flat = P.ravel()

    # Sum power per bin
    bin_indices = np.floor(r_flat * num_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    energy_bins = np.bincount(bin_indices, weights=p_flat, minlength=num_bins)

    # Normalize energy to unit area (sum=1)
    total = energy_bins.sum()
    if total > 0:
        energy_norm = energy_bins / total
    else:
        energy_norm = energy_bins  # all zeros edge-case

    return centers, energy_norm

def write_csv(path, headers, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def safe_stem(filename):
    name, _ = os.path.splitext(os.path.basename(filename))
    return name

def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, fn)
        for fn in sorted(os.listdir(folder))
        if os.path.splitext(fn.lower())[1] in exts
    ]

# -------------------------
# Main processing
# -------------------------

def process_group(name, folder, num_bins=NUM_BINS, max_side=MAX_SIDE, csv_root="", plots_root=""):
    """
    Processes a single group (GT, classical, deep).
    Returns dict with per-image spectra and average/mean/std arrays.
    """
    images = list_images(folder)
    spectra = []
    freq_centers = None

    # Output CSV folder for per-image spectra
    img_csv_dir = ensure_dir(os.path.join(csv_root, "spectra", name))

    for path in images:
        try:
            img = Image.open(path)
            img = maybe_downscale(img, max_side=max_side)
            arr = to_grayscale_array(img)
            centers, energy = radial_fourier_energy(arr, num_bins=num_bins)
            if freq_centers is None:
                freq_centers = centers
            else:
                # Sanity: all centers should match
                if len(freq_centers) != len(centers) or not np.allclose(freq_centers, centers):
                    # Re-bin to match baseline centers (should not happen with fixed num_bins)
                    pass
            spectra.append(energy)

            # Save per-image CSV
            csv_path = os.path.join(img_csv_dir, f"{safe_stem(path)}_spectrum.csv")
            rows = list(zip(freq_centers, energy))
            write_csv(csv_path, headers=["frequency_bin_normalized_0_to_1", "normalized_energy"], rows=rows)
        except Exception as e:
            print(f"[{name}] Skipped {path} due to error: {e}")

    if len(spectra) == 0:
        # Return empty structure if no images
        return {
            "name": name,
            "freq": np.linspace(0, 1, num_bins, endpoint=False) + (0.5/num_bins),
            "spectra": [],
            "mean": None,
            "std": None,
            "count": 0
        }

    spectra_arr = np.stack(spectra, axis=0)  # [N, B]
    mean_spec = spectra_arr.mean(axis=0)
    std_spec = spectra_arr.std(axis=0)

    # Save average spectrum CSV
    avg_csv_dir = ensure_dir(os.path.join(csv_root, "averages"))
    avg_csv_path = os.path.join(avg_csv_dir, f"{name}_average_spectrum.csv")
    rows = list(zip(freq_centers, mean_spec, std_spec))
    write_csv(avg_csv_path, headers=["frequency_bin_normalized_0_to_1", "mean_normalized_energy", "std_normalized_energy"], rows=rows)

    return {
        "name": name,
        "freq": freq_centers,
        "spectra": spectra_arr,
        "mean": mean_spec,
        "std": std_spec,
        "count": spectra_arr.shape[0]
    }

def main():
    # Prepare output structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(OUTPUT_ROOT)
    plots_root = ensure_dir(os.path.join(root, "plots"))
    csv_root = ensure_dir(os.path.join(root, "csv"))

    # Process all groups
    groups = [
        ("ground_truth", GT_DIR),
        ("classical", CLASSICAL_DIR),
        ("deep", DEEP_DIR),
    ]

    results = []
    for name, folder in groups:
        print(f"Processing {name} from: {folder}")
        res = process_group(name, folder, num_bins=NUM_BINS, max_side=MAX_SIDE, csv_root=csv_root, plots_root=plots_root)
        results.append(res)

    # Combined averages CSV
    comb_csv_dir = ensure_dir(os.path.join(csv_root, "averages"))
    comb_csv_path = os.path.join(comb_csv_dir, "combined_average_spectra.csv")

    # Prepare combined CSV rows with columns: freq, mean_gt, std_gt, mean_classical, std_classical, mean_deep, std_deep
    freq = results[0]["freq"] if results and results[0]["freq"] is not None else np.linspace(0, 1, NUM_BINS, endpoint=False) + (0.5/NUM_BINS)
    name_to_res = {r["name"]: r for r in results}

    def get_mean_std(n):
        r = name_to_res.get(n, {})
        return (r.get("mean", None), r.get("std", None))

    m_gt, s_gt = get_mean_std("ground_truth")
    m_cl, s_cl = get_mean_std("classical")
    m_dp, s_dp = get_mean_std("deep")

    # Align any None with zeros for CSV consistency
    def nz(x):
        return np.zeros_like(freq) if x is None else x

    rows = list(zip(
        freq,
        nz(m_gt), nz(s_gt),
        nz(m_cl), nz(s_cl),
        nz(m_dp), nz(s_dp)
    ))
    write_csv(
        comb_csv_path,
        headers=[
            "frequency_bin_normalized_0_to_1",
            "gt_mean", "gt_std",
            "classical_mean", "classical_std",
            "deep_mean", "deep_std"
        ],
        rows=rows
    )

    # -------------------------
    # PLOTS
    # -------------------------

    # 1) Per-image spectra overlay (light alpha)
    plt.figure(figsize=(10, 6))
    colors = {
        "ground_truth": None,  # let matplotlib choose
        "classical": None,
        "deep": None
    }
    for res in results:
        if res["spectra"] is None or len(res["spectra"]) == 0:
            continue
        # Use the first plotted line's color for consistency within group
        group_color = None
        for i, spec in enumerate(res["spectra"]):
            line, = plt.plot(res["freq"], spec, alpha=0.15, linewidth=0.8, label=res["name"] if i == 0 else None)
            if group_color is None:
                group_color = line.get_color()
        colors[res["name"]] = group_color

    plt.title("Per-image Fourier Energy Distribution (Normalized)")
    plt.xlabel("Normalized Radial Frequency (0 = DC, 1 = Nyquist)")
    plt.ylabel("Normalized Energy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    per_image_plot_path = os.path.join(plots_root, "per_image_spectra.png")
    plt.tight_layout()
    plt.savefig(per_image_plot_path, dpi=180)
    plt.close()

    # 2) Group averages with ±1σ shading
    plt.figure(figsize=(10, 6))
    for res in results:
        if res["mean"] is None:
            continue
        c = colors.get(res["name"], None)
        plt.plot(res["freq"], res["mean"], linewidth=2.0, label=f"{res['name']} (n={res['count']})", color=c)
        if res["std"] is not None:
            lo = np.clip(res["mean"] - res["std"], 0.0, None)
            hi = res["mean"] + res["std"]
            plt.fill_between(res["freq"], lo, hi, alpha=0.2, label=None, color=c)

    plt.title("Average Fourier Energy Distribution ±1σ (Normalized)")
    plt.xlabel("Normalized Radial Frequency (0 = DC, 1 = Nyquist)")
    plt.ylabel("Normalized Energy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    avg_plot_path = os.path.join(plots_root, "group_averages.png")
    plt.tight_layout()
    plt.savefig(avg_plot_path, dpi=180)
    plt.close()

    print("\nDone.")
    print(f"CSV saved under: {csv_root}")
    print(f"Plots saved under: {plots_root}")

if __name__ == "__main__":
    main()
