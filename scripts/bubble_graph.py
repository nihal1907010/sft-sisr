# bubble_pdfs_xlsx_variants_fanout_labels.py
# Requirements: pandas, matplotlib, torch, openpyxl

import re
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
XLSX_PATH = Path("model_full_stats.xlsx")   # your Excel file
SHEET_NAME = 0                               # or a sheet name string
DATASET_COL_RANGE = (1, 11)                  # B..K (df.iloc[:, 1:11])
SAVE_DIR = Path("./bubble_outputs")
SAVE_PDF_DIR = SAVE_DIR / "per_dataset_pdf"
SAVE_PDF_DIR.mkdir(parents=True, exist_ok=True)

# All PDFs will have the same physical size (equal width/height across outputs)
FIGSIZE = (12, 7)   # wide to reduce collisions
DPI = 300

# Bubble radius mapping (points). Matplotlib 's' is area in points^2 -> s = radius^2
MIN_RADIUS_PT = 6.0
MAX_RADIUS_PT = 22.0

# Label layout (in points)
LABEL_STYLE     = "right"   # "right", "angled", or "vertical"
LABEL_OFFSET_PT = 50        # distance from bubble to label anchor (along leader)
MIN_SEP_PT      = 30        # minimum vertical separation between labels (in points)
ANGLE_DEG       = 30        # fan-out angle (used by "right" and "angled")

# -----------------------------
# Helpers
# -----------------------------
def find_params_col(df):
    for c in df.columns:
        norm = str(c).strip().lower().replace(" ", "").replace("(", "_").replace(")", "")
        if norm in ("params_m", "parameters_m", "paramsm"):
            return c
    for c in df.columns:
        lc = str(c).strip().lower()
        if "params" in lc and ("_m" in lc or "(m" in lc or "million" in lc or "millions" in lc):
            return c
    raise ValueError("No parameters column found. Expect something like 'params_M' (millions).")

def scale_radius_from_params(params: np.ndarray,
                             rmin=MIN_RADIUS_PT, rmax=MAX_RADIUS_PT) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return np.full_like(p, (rmin + rmax) / 2.0, dtype=float)
    pmin, pmax = np.min(p[finite]), np.max(p[finite])
    if np.isclose(pmin, pmax):
        return np.full_like(p, (rmin + rmax) / 2.0, dtype=float)
    r = rmin + (p - pmin) * (rmax - rmin) / (pmax - pmin)
    r[~finite] = (rmin + rmax) / 2.0
    return r

def get_variant_col(df):
    for c in df.columns:
        if str(c).strip().lower() == "model_variant":
            return c
    raise ValueError("No column named 'model_variant' found in the sheet.")

def collect_dataset_pairs(df, start_idx=1, end_idx=11):
    wide_cols = list(df.columns[start_idx:end_idx])
    if not wide_cols:
        raise ValueError("No dataset columns found in B–K.")
    pair_map = defaultdict(lambda: {"psnr": None, "ssim": None})
    pat = re.compile(r"^(?P<ds>.+)_(?P<m>psnr|ssim)$", re.IGNORECASE)
    for c in wide_cols:
        m = pat.match(str(c))
        if m:
            ds = m.group("ds")
            metric = m.group("m").lower()
            pair_map[ds][metric] = c
    pairs = [(ds, v["psnr"], v["ssim"]) for ds, v in pair_map.items() if v["psnr"] and v["ssim"]]
    if not pairs:
        raise ValueError("No valid <dataset>_psnr / <dataset>_ssim pairs found in B–K.")
    return sorted(pairs, key=lambda t: str(t[0]).lower())

# -----------------------------
# Color Mapping
# -----------------------------
def make_color_map(variants):
    """Assign a unique color to each variant using a colormap."""
    cmap = plt.cm.get_cmap("tab20", len(variants))
    return {v: cmap(i) for i, v in enumerate(variants)}

# -----------------------------
# Labeling functions (modified to take colors)
# -----------------------------
def label_fanout_right_no_overlap(ax, xs, ys, labels, colors,
                                  offset_pt=LABEL_OFFSET_PT, min_sep_pt=MIN_SEP_PT, angle_deg=ANGLE_DEG):
    fig = ax.figure
    fig.canvas.draw()
    to_disp = ax.transData.transform
    to_data = ax.transData.inverted().transform
    ax_bbox_disp = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    y_bottom = ax_bbox_disp.y0

    pt_to_px = fig.dpi / 72.0
    offset_px = offset_pt * pt_to_px
    min_sep_px = min_sep_pt * pt_to_px

    pts_disp = to_disp(np.column_stack([xs, ys]))
    xd, yd = pts_disp[:, 0], pts_disp[:, 1]

    ang = np.deg2rad(abs(angle_deg))
    dx = np.cos(ang) * offset_px
    dy = np.sin(ang) * offset_px

    x_tgt = xd + dx
    y_tgt = yd - dy

    order = np.argsort(y_tgt)[::-1]  # top to bottom
    y_final = np.copy(y_tgt)
    placed = []
    for i in order:
        y_cand = y_tgt[i]
        for y_prev in placed:
            if y_cand > y_prev - min_sep_px:
                y_cand = y_prev - min_sep_px
        y_cand = max(y_cand, y_bottom + 2.0)
        y_final[i] = y_cand
        placed.append(y_cand)

    for i, (name, color) in enumerate(zip(labels, colors)):
        xlab, ylab = to_data((x_tgt[i], y_final[i]))
        x0, y0 = to_data((xd[i], yd[i]))
        ax.plot([x0, xlab], [y0, ylab], lw=0.9, alpha=0.85, color=color, clip_on=False)
        ax.text(xlab, ylab, str(name), fontsize=16, ha="left", va="center",
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
                clip_on=False)

    # Expand xlim so labels aren’t clipped
    x0, x1 = ax.get_xlim()
    extra = (dx / fig.dpi) * (x1 - x0) * 0.1
    ax.set_xlim(x0, x1 + extra)

# -----------------------------
# Load Excel & reshape
# -----------------------------
df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
if df.empty:
    raise ValueError("The Excel sheet appears to be empty.")

variant_col = get_variant_col(df)
params_col  = find_params_col(df)
ds_pairs    = collect_dataset_pairs(df, *DATASET_COL_RANGE)

records = []
for _, row in df.iterrows():
    variant  = row[variant_col]
    params_m = row.get(params_col, np.nan)
    for ds, psnr_c, ssim_c in ds_pairs:
        psnr_v = row.get(psnr_c, np.nan)
        ssim_v = row.get(ssim_c, np.nan)
        if pd.isna(psnr_v) or pd.isna(ssim_v):
            continue
        records.append(
            {
                "variant": variant,
                "dataset": ds,
                "psnr": float(psnr_v),
                "ssim": float(ssim_v),
                "params_M": float(params_m) if pd.notna(params_m) else np.nan,
            }
        )

long_df = pd.DataFrame.from_records(records)
if long_df.empty:
    raise ValueError("No valid PSNR/SSIM pairs found after reshaping.")

# Make color map across all variants
variant_list = long_df["variant"].unique().tolist()
color_map = make_color_map(variant_list)

# -----------------------------
# Save individual PDFs per dataset
# -----------------------------
for ds in sorted(long_df["dataset"].unique(), key=lambda s: str(s).lower()):
    sub = long_df[long_df["dataset"] == ds].copy()

    xs = sub["psnr"].to_numpy(dtype=float)
    ys = sub["ssim"].to_numpy(dtype=float)
    params_m = sub["params_M"].to_numpy(dtype=float)
    variants = sub["variant"].astype(str).tolist()

    radii = scale_radius_from_params(params_m, rmin=MIN_RADIUS_PT, rmax=MAX_RADIUS_PT)
    sizes_pts2 = np.square(radii)

    colors = [color_map[v] for v in variants]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.scatter(xs, ys, s=sizes_pts2 if sizes_pts2.size else 300.0,
               alpha=0.65, linewidths=0.5, edgecolors="black", c=colors)

    ax.set_title(str(ds), fontsize=16, fontweight="bold")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("SSIM")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if xs.size and ys.size:
        xr = xs.max() - xs.min() if xs.size > 1 else 1.0
        yr = ys.max() - ys.min() if ys.size > 1 else 0.1
        ax.set_xlim(xs.min() - 0.05 * xr, xs.max() + 0.10 * xr)
        ax.set_ylim(ys.min() - 0.05 * yr, ys.max() + 0.12 * yr)

    # ----- Labels -----
    if LABEL_STYLE == "right":
        label_fanout_right_no_overlap(ax, xs, ys, variants, colors,
                                      offset_pt=LABEL_OFFSET_PT,
                                      min_sep_pt=MIN_SEP_PT,
                                      angle_deg=ANGLE_DEG)
    else:
        # fall back to black labels for other styles
        label_fanout_right_no_overlap(ax, xs, ys, variants, ["black"]*len(variants),
                                      offset_pt=LABEL_OFFSET_PT,
                                      min_sep_pt=MIN_SEP_PT,
                                      angle_deg=ANGLE_DEG)

    out_path = SAVE_PDF_DIR / f"{str(ds).replace('/', '_')}.pdf"
    fig.savefig(out_path, format="pdf", dpi=DPI)
    plt.close(fig)

print(f"Saved individual dataset PDFs in: {SAVE_PDF_DIR.resolve()}")
