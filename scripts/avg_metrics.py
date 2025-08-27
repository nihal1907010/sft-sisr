import pandas as pd
from pathlib import Path
import re

# === SET THIS TO YOUR ROOT FOLDER ===
ROOT = Path("results/results")  # contains subfolders like "10.conv", "11.wide", ...

OUT_CSV = ROOT / "model_metrics_wide_cleaned.csv"

def clean_model_name(name: str) -> str:
    """Remove leading 'digits.' prefix from a model folder name."""
    return re.sub(r"^\d+\.", "", name)

def clean_dataset_label(stem: str) -> str:
    """Remove 'metrics' (and surrounding underscores/dashes) from a filename stem."""
    # Examples: 'B100_metrics' -> 'B100', 'Urban100-metrics' -> 'Urban100'
    s = re.sub(r"[_-]*metrics[_-]*", "", stem, flags=re.IGNORECASE)
    return s or stem  # fallback if everything stripped

def find_metric_cols(df):
    """Find psnr/ssim column names (case-insensitive, prefix match)."""
    lower = {c.lower(): c for c in df.columns}
    psnr = lower.get("psnr") or next((c for c in df.columns if c.lower().startswith("psnr")), None)
    ssim = lower.get("ssim") or next((c for c in df.columns if c.lower().startswith("ssim")), None)
    return psnr, ssim

# --- gather model folders ---
model_dirs = [d for d in sorted(ROOT.iterdir()) if d.is_dir()]
if not model_dirs:
    raise SystemExit(f"No model folders found under {ROOT}")

# --- infer dataset order & labels from the first model folder (by sorted filename) ---
first_csvs = sorted(model_dirs[0].glob("*.csv"))
if not first_csvs:
    raise SystemExit(f"No dataset CSVs found inside {model_dirs[0]}")
dataset_labels = [clean_dataset_label(p.stem) for p in first_csvs]

def csvs_for_model(model_dir: Path):
    """Return sorted list of CSVs for a model (aligned by index with first_csvs)."""
    return sorted(model_dir.glob("*.csv"))

rows = []
for model_dir in model_dirs:
    model_name = clean_model_name(model_dir.name)
    csvs = csvs_for_model(model_dir)

    row = {"model_variant": model_name}
    for i, ds_label in enumerate(dataset_labels):
        # default to NaN if missing
        psnr_key = f"{ds_label}_psnr"
        ssim_key = f"{ds_label}_ssim"
        row[psnr_key] = pd.NA
        row[ssim_key] = pd.NA

        if i >= len(csvs):
            print(f"[WARN] {model_dir.name}: missing dataset #{i+1} ({ds_label})")
            continue

        csv_path = csvs[i]
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] skipping {csv_path}: {e}")
            continue

        psnr_col, ssim_col = find_metric_cols(df)
        if psnr_col is None or ssim_col is None:
            print(f"[WARN] {csv_path.name}: missing psnr/ssim columns; columns={list(df.columns)}")
            continue

        psnr = pd.to_numeric(df[psnr_col], errors="coerce").mean()
        ssim = pd.to_numeric(df[ssim_col], errors="coerce").mean()
        row[psnr_key] = psnr
        row[ssim_key] = ssim

    rows.append(row)

# Build DataFrame with stable column order
cols = ["model_variant"]
for ds in dataset_labels:
    cols += [f"{ds}_psnr", f"{ds}_ssim"]

out = pd.DataFrame(rows)
# Ensure all expected columns exist even if some models missed datasets
for c in cols:
    if c not in out.columns:
        out[c] = pd.NA
out = out[cols]

out.to_csv(OUT_CSV, index=False)
print(f"✅ Wrote: {OUT_CSV}")
