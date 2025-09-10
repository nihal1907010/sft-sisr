import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

# -----------------------------
# Config
# -----------------------------
FOLDERS = {
    "highres": "datasets/Test/Urban100/HR",
    "classical": "datasets/Urban100_Lanczos_Upsampled",
    "deeplearn": "results/3.main/visualization/Urban100",
}
OUTPUT_DIR = Path("outputs")
PER_IMAGE_DIR = OUTPUT_DIR / "per_image"
SUMMARY_DIR = OUTPUT_DIR / "summary"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# -----------------------------
# Helpers
# -----------------------------

def load_grayscale(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32)
    if arr.max() > 0:
        arr /= 255.0
    return arr

def pad_or_crop_to_match(a: np.ndarray, ref_shape) -> np.ndarray:
    h, w = ref_shape
    ah, aw = a.shape
    if (ah, aw) == (h, w):
        return a
    out = a
    if ah < h:
        pad_top = (h - ah) // 2
        pad_bot = h - ah - pad_top
        out = np.pad(out, ((pad_top, pad_bot), (0, 0)), mode="reflect")
    elif ah > h:
        cut_top = (ah - h) // 2
        out = out[cut_top:cut_top+h, :]
    ah2, aw2 = out.shape
    if aw2 < w:
        pad_left = (w - aw2) // 2
        pad_right = w - aw2 - pad_left
        out = np.pad(out, ((0, 0), (pad_left, pad_right)), mode="reflect")
    elif aw2 > w:
        cut_left = (aw2 - w) // 2
        out = out[:, cut_left:cut_left+w]
    return out

def radial_energy_bins(img: np.ndarray, n_bins=3) -> np.ndarray:
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    power = np.abs(Fshift) ** 2
    H, W = img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H))
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    FX, FY = np.meshgrid(fx, fy)
    radii = np.sqrt(FX**2 + FY**2)
    rmax = radii.max()
    edges = np.linspace(0.0, rmax, n_bins+1)
    total_energy = power.sum() + 1e-12
    band_energy = []
    for i in range(n_bins):
        mask = (radii >= edges[i]) & (radii < edges[i+1])
        band_energy.append(power[mask].sum() / total_energy)
    band_energy = np.array(band_energy)
    return band_energy / (band_energy.sum() + 1e-12)

def list_images(folder: Path, strip_main_suffix=False):
    mapping = {}
    if not folder.exists():
        return mapping
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            stem = p.stem
            if strip_main_suffix and stem.endswith("_main"):
                stem = stem[:-5]
            mapping[stem] = p
    return mapping

def ensure_dirs():
    PER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    highres_map   = list_images(Path(FOLDERS["highres"]))
    classical_map = list_images(Path(FOLDERS["classical"]))
    deeplearn_map = list_images(Path(FOLDERS["deeplearn"]), strip_main_suffix=True)

    common_stems = sorted(set(highres_map) & set(classical_map) & set(deeplearn_map))
    if not common_stems:
        print("No common stems, check filenames!")
        return

    per_image_csv = PER_IMAGE_DIR / "per_image_metrics.csv"
    with open(per_image_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image_id",
                         "highres_low","highres_med","highres_high",
                         "classical_low","classical_med","classical_high",
                         "deeplearn_low","deeplearn_med","deeplearn_high"])

        all_hr, all_cl, all_dl = [], [], []

        for stem in common_stems:
            im_hr = load_grayscale(highres_map[stem])
            im_cl = load_grayscale(classical_map[stem])
            im_dl = load_grayscale(deeplearn_map[stem])
            im_cl = pad_or_crop_to_match(im_cl, im_hr.shape)
            im_dl = pad_or_crop_to_match(im_dl, im_hr.shape)

            hr_bands = radial_energy_bins(im_hr, 3)
            cl_bands = radial_energy_bins(im_cl, 3)
            dl_bands = radial_energy_bins(im_dl, 3)

            writer.writerow([stem, *hr_bands, *cl_bands, *dl_bands])

            # Scatter plot
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(*hr_bands, c="b", marker="o", s=80, label="High-Res")
            ax.scatter(*cl_bands, c="g", marker="^", s=80, label="Classical")
            ax.scatter(*dl_bands, c="r", marker="s", s=80, label="Deep Learning")

            ax.set_xlabel("Low frequency energy")
            ax.set_ylabel("Medium frequency energy")
            ax.set_zlabel("High frequency energy")
            ax.set_title(f"Frequency Energy Scatter: {stem}")
            ax.legend()

            plt.tight_layout()
            plt.savefig(PER_IMAGE_DIR / f"{stem}_scatter.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            all_hr.append(hr_bands)
            all_cl.append(cl_bands)
            all_dl.append(dl_bands)

    # Summary (mean points)
    mean_hr = np.mean(all_hr, axis=0)
    mean_cl = np.mean(all_cl, axis=0)
    mean_dl = np.mean(all_dl, axis=0)

    with open(SUMMARY_DIR / "summary_metrics.csv", "w", newline="") as fsum:
        writer = csv.writer(fsum)
        writer.writerow(["metric","low","medium","high"])
        writer.writerow(["avg_highres_energy", *mean_hr])
        writer.writerow(["avg_classical_energy", *mean_cl])
        writer.writerow(["avg_deeplearn_energy", *mean_dl])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*mean_hr, c="b", marker="o", s=100, label="High-Res (avg)")
    ax.scatter(*mean_cl, c="g", marker="^", s=100, label="Classical (avg)")
    ax.scatter(*mean_dl, c="r", marker="s", s=100, label="Deep Learning (avg)")

    ax.set_xlabel("Low frequency energy")
    ax.set_ylabel("Medium frequency energy")
    ax.set_zlabel("High frequency energy")
    ax.set_title("Average Frequency Energy Scatter (All Images)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(SUMMARY_DIR / "summary_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Done. Scatter plots saved!")

if __name__ == "__main__":
    main()
