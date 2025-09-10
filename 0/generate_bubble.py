import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager, rcParams, font_manager as fm
from matplotlib.font_manager import FontProperties
from pathlib import Path
import os

# =========================
# CONFIG — EDIT AS NEEDED
# =========================
EXCEL_PATH = Path("model_full_stats.xlsx")
SHEET_NAME = 0
OUTPUT_PDF = Path("model_quality_vs_size.pdf")  # <-- save path

COL_X = "Average PSNR"
COL_Y = "Average SSIM"
COL_SIZE = "Parameters (Millions)"
COL_LABEL = "Model Variants"

TITLE = "Model Quality vs. Size"
FIGSIZE = (10, 7)

# ------- FONT: FORCE EXACT TIMES (NO FALLBACKS) -------
# Set this to the exact Times font file on your system.
# Examples:
#   Windows: r"C:\Windows\Fonts\times.ttf"  or "times.ttf" (Times New Roman Regular)
#   macOS:   "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
#   Linux:   path to downloaded "Times New Roman.ttf" (after installing mscorefonts or manual copy)
FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"   # <-- CHANGE THIS

# Sizes
BASE_FONTSIZE = 14
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
ANNOTATION_FONTSIZE = 14

# Bubble styling
BUBBLE_ALPHA = 0.65
BUBBLE_EDGE_COLOR = "black"
BUBBLE_EDGE_WIDTH = 0.6
BUBBLE_MIN_AREA = 60
BUBBLE_MAX_AREA = 2000

# Grid
SHOW_GRID = True
GRID_STYLE = dict(linestyle="--", linewidth=0.6, alpha=0.4)

# Leader / label styling
LEADER1_LEN_PT = 18         # base length of the -45° segment (points)
LEADER2_LEN_PT = 26         # length of the horizontal segment (points)
LEADER_LW = 1.5             # thickness of connecting leader lines
LEADER_COLOR = "gray"
LABEL_PAD_PT = 4            # gap from end of horizontal line to label
LABEL_V_OFFSET_PT = 0       # vertical tweak of label from the end of horizontal line
LABEL_BOX = dict(boxstyle="round,pad=0.25", facecolor="white",
                 edgecolor="none", alpha=0.85)  # no outline on label box

# Auto-extension of the -45° segment to avoid overlapping labels
AUTO_EXTEND = True
EXTEND_STEP_PT = 6          # increase -45° length by this many points when overlapping
MAX_EXTRA_PT = 48           # cap on extra length beyond LEADER1_LEN_PT
MAX_REPEL_ITERS = 30        # safety limit for iterations
# =========================


def _check_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")


def _scale_to_area(values, min_area=BUBBLE_MIN_AREA, max_area=BUBBLE_MAX_AREA):
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return np.full_like(v, (min_area + max_area) / 2.0, dtype=float)
    return np.interp(v, (vmin, vmax), (min_area, max_area))


def _pt_to_px(fig, pt):
    return pt * fig.dpi / 72.0


def _offset_data(ax, x, y, dx_pt, dy_pt):
    """Return data coords after applying a point offset (dx_pt, dy_pt) from (x, y)."""
    fig = ax.figure
    xpix, ypix = ax.transData.transform((x, y))
    xpix += _pt_to_px(fig, dx_pt)
    ypix += _pt_to_px(fig, dy_pt)
    return ax.transData.inverted().transform((xpix, ypix))


def _compute_elbow_and_label(ax, x, y, len1_pt, len2_pt):
    """Given a point (x,y) and leader lengths in points, return elbow, end-of-horizontal, and label coords."""
    c = np.sqrt(2) / 2.0
    dx1_pt = len1_pt * c
    dy1_pt = -len1_pt * c

    ex, ey = _offset_data(ax, x, y, dx1_pt, dy1_pt)
    hx, hy = _offset_data(ax, ex, ey, len2_pt, 0)
    lx, ly = _offset_data(ax, hx, hy, LABEL_PAD_PT, LABEL_V_OFFSET_PT)
    return (ex, ey), (hx, hy), (lx, ly)


def _draw_two_segment(ax, x, y, elbow, horiz_end, lw, color):
    """Draw two leader segments and return the Line2D artists."""
    (ex, ey), (hx, hy) = elbow, horiz_end
    l1 = Line2D([x, ex], [y, ey], lw=lw, color=color, solid_capstyle="round")
    l2 = Line2D([ex, hx], [ey, hy], lw=lw, color=color, solid_capstyle="round")
    ax.add_line(l1)
    ax.add_line(l2)
    return l1, l2


def _force_times_font():
    """Register and force Matplotlib to use exactly the TTF at FONT_PATH, no fallbacks."""
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(
            f"Times font file not found at:\n{FONT_PATH}\n"
            "Please set FONT_PATH to the exact Times New Roman .ttf on your system."
        )

    # Register the font file
    font_manager.fontManager.addfont(FONT_PATH)

    # Get the internal font name from the file (e.g., 'Times New Roman')
    fp = FontProperties(fname=FONT_PATH)
    font_name = fp.get_name()

    # Set Matplotlib to ONLY use this font
    rcParams["font.family"] = font_name
    rcParams["font.serif"] = [font_name]
    rcParams["pdf.fonttype"] = 42   # embed TrueType in PDF
    rcParams["ps.fonttype"] = 42

    # Optional sanity check: ensure the name is known to font manager
    available = {f.name for f in fm.fontManager.ttflist}
    if font_name not in available:
        # Sometimes ttflist isn't rebuilt immediately; this still usually works.
        # Print a gentle warning; we already set by name via rcParams.
        print(f"Warning: '{font_name}' not listed in ttflist yet, but it will still be used.")

    return font_name


def main():
    font_name = _force_times_font()

    mpl.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    })

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    _check_columns(df, [COL_X, COL_Y, COL_SIZE, COL_LABEL])
    df = df[[COL_X, COL_Y, COL_SIZE, COL_LABEL]].dropna(subset=[COL_X, COL_Y, COL_SIZE])

    areas = _scale_to_area(df[COL_SIZE].values)
    cmap = plt.cm.get_cmap("tab20", max(3, len(df)))
    colors = [cmap(i % cmap.N) for i in range(len(df))]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.scatter(df[COL_X], df[COL_Y], s=areas, c=colors,
               alpha=BUBBLE_ALPHA, edgecolor=BUBBLE_EDGE_COLOR, linewidth=BUBBLE_EDGE_WIDTH)

    ax.set_title(TITLE)
    ax.set_xlabel(COL_X)
    ax.set_ylabel(COL_Y)
    if SHOW_GRID:
        ax.grid(True, **GRID_STYLE)

    # --- Legend sorted by parameter size ---
    xs = df[COL_X].to_numpy(float)
    ys = df[COL_Y].to_numpy(float)
    labs = df[COL_LABEL].astype(str).to_list()
    variant_sizes = df[COL_SIZE].to_numpy(float)

    sort_idx = np.argsort(variant_sizes)

    handles = [
        plt.scatter([], [], s=areas[i],
                    edgecolor=BUBBLE_EDGE_COLOR, facecolor=colors[i],
                    alpha=BUBBLE_ALPHA, linewidth=BUBBLE_EDGE_WIDTH)
        for i in sort_idx
    ]
    labels = [f"{labs[i]} — {variant_sizes[i]:.1f}M" for i in sort_idx]

    ax.legend(
        handles, labels,
        title="Params (Millions) by Variant",
        scatterpoints=1, frameon=True, loc="best",
        fontsize=TICK_FONTSIZE-1, title_fontsize=TICK_FONTSIZE
    )

    # --- Create leaders + labels with optional auto-extension ---
    extra = np.zeros(len(xs), dtype=float)
    line_pairs = [None] * len(xs)
    text_art = [None] * len(xs)

    def place_all():
        for i in range(len(xs)):
            if line_pairs[i] is not None:
                for ln in line_pairs[i]:
                    ln.remove()
            if text_art[i] is not None:
                text_art[i].remove()

        for i, (x, y, lab) in enumerate(zip(xs, ys, labs)):
            elbow, horiz_end, label_pos = _compute_elbow_and_label(
                ax, x, y, LEADER1_LEN_PT + extra[i], LEADER2_LEN_PT
            )
            line_pairs[i] = _draw_two_segment(ax, x, y, elbow, horiz_end,
                                              LEADER_LW, LEADER_COLOR)
            lx, ly = label_pos
            text_art[i] = ax.text(lx, ly, lab, fontsize=ANNOTATION_FONTSIZE,
                                  bbox=LABEL_BOX, ha="left", va="center")

    def any_overlaps_and_mark():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbs = [t.get_window_extent(renderer=renderer).expanded(1.03, 1.15) for t in text_art]
        needs = np.zeros(len(bbs), dtype=bool)
        overlap_found = False
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                if bbs[i].overlaps(bbs[j]):
                    overlap_found = True
                    if xs[i] <= xs[j]:
                        needs[i] = True
                    else:
                        needs[j] = True
        return overlap_found, needs

    place_all()

    if AUTO_EXTEND:
        for _ in range(MAX_REPEL_ITERS):
            overlap_found, needs = any_overlaps_and_mark()
            if not overlap_found:
                break
            for i, n in enumerate(needs):
                if n and extra[i] < MAX_EXTRA_PT:
                    extra[i] += EXTEND_STEP_PT
            place_all()

    # Save to PDF before showing (embed TrueType thanks to pdf.fonttype=42)
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    print(f"Using font: {font_name}")
    print(f"Plot saved to {OUTPUT_PDF.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
