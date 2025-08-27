def main():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # === Config ===
    excel_path = Path("model_full_stats.xlsx")  # adjust if needed
    sheet = "Sheet1"
    figsize = (15, 10)
    title_fs = 35
    label_fs = 35
    tick_fs = 30
    note_fs = 11      # per-point label font size
    dpi = 300

    # Which datasets to plot (PSNR columns)
    datasets = [
        ("Set5_psnr", "Set5"),
        ("Set14_psnr", "Set14"),
        ("B100_psnr", "B100"),
        ("Urban100_psnr", "Urban100"),
        ("Manga109_psnr", "Manga109"),
    ]

    # === Load data ===
    df = pd.read_excel(excel_path, sheet_name=sheet)

    # Basic validation
    req = ["model_variant", "params_M", "macs_G"] + [c for c, _ in datasets]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    # Normalize bubble sizes from MACs (G) → points^2 range
    macs = df["macs_G"].astype(float).to_numpy()
    macs_min, macs_max = float(np.min(macs)), float(np.max(macs))
    # avoid div-by-zero if all MACs are identical
    macs_norm = (macs - macs_min) / (macs_max - macs_min + 1e-9)
    # map to a visible range
    sizes_pts2 = 200 + 1800 * macs_norm
    size_map = dict(zip(df.index, sizes_pts2))

    # Helper to add a size legend with representative MACs
    def add_size_legend(ax, values):
        # choose quartiles as size legend markers
        q = np.quantile(values, [0.25, 0.5, 0.9])
        for qv in q:
            s = 200 + 1800 * ((qv - macs_min) / (macs_max - macs_min + 1e-9))
            ax.scatter([], [], s=s, alpha=0.7, label=f"{qv:.2f} G MACs")
        ax.legend(
            title="MACs (G)",
            scatterpoints=1,
            fontsize=12,
            title_fontsize=14,
            frameon=True,
            loc="lower right"
        )

    # === Make one figure per dataset ===
    for col, name in datasets:
        fig, ax = plt.subplots(figsize=figsize)

        x = df["params_M"].astype(float).to_numpy()
        y = df[col].astype(float).to_numpy()
        s = np.array([size_map[i] for i in df.index])

        # scatter
        ax.scatter(x, y, s=s, alpha=0.7, marker='o', edgecolors='white', linewidths=1.0)

        # annotate each point
        for xi, yi, label in zip(x, y, df["model_variant"]):
            ax.annotate(str(label), (xi, yi), xytext=(6, 6),
                        textcoords="offset points", fontsize=note_fs)

        # axes, title, grid
        ax.set_xlabel("Parameters (M)", fontsize=label_fs)
        ax.set_ylabel("PSNR (dB)", fontsize=label_fs)
        ax.set_title(f"{name}: PSNR vs Parameters (bubble size = MACs in G)", fontsize=title_fs)

        for t in ax.get_xticklabels():
            t.set_fontsize(tick_fs)
        for t in ax.get_yticklabels():
            t.set_fontsize(tick_fs)

        ax.grid(visible=True, linestyle='-.', linewidth=0.5)

        # size legend
        add_size_legend(ax, macs)

        # Save (before show)
        out = Path(f"{name}_psnr_bubble.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)  # close to keep memory clean

    print("Done. Saved:",
          [f"{name}_psnr_bubble.png" for _, name in datasets])


if __name__ == "__main__":
    main()
