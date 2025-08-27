import os
import pandas as pd

parent_dir = "results"

# 1) Discover folders first (so the new output dir isn't included)
folders = sorted([f for f in os.listdir(parent_dir)
                  if os.path.isdir(os.path.join(parent_dir, f)) and not f.startswith(".")])

if not folders:
    raise RuntimeError(f"No folders found in {parent_dir}")

# 2) Figure out the csv filenames from the first real folder
sample_folder = os.path.join(parent_dir, folders[0])
csv_files = sorted([f for f in os.listdir(sample_folder) if f.lower().endswith(".csv")])
if not csv_files:
    raise RuntimeError(f"No CSV files found in {sample_folder}")

# 3) NOW create the output dir
output_dir = os.path.join(parent_dir, "merged_by_metric")
os.makedirs(output_dir, exist_ok=True)

def type_from_folder(folder_name: str) -> str:
    parts = folder_name.split(".", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else folder_name

for csv_name in csv_files:
    merged_psnr = None
    merged_ssim = None

    for folder in folders:
        folder_path = os.path.join(parent_dir, folder)

        # Skip the output_dir if someone reuses the script
        if os.path.abspath(folder_path) == os.path.abspath(output_dir):
            continue

        file_path = os.path.join(folder_path, csv_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected file missing: {file_path}")

        df = pd.read_csv(file_path)
        required_cols = {"img_name", "psnr", "ssim"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{file_path} must contain columns {required_cols}, found {set(df.columns)}")

        tname = type_from_folder(folder)

        df_psnr = df[["img_name", "psnr"]].copy()
        df_psnr["psnr"] = pd.to_numeric(df_psnr["psnr"], errors="coerce")
        df_psnr.rename(columns={"psnr": tname}, inplace=True)

        df_ssim = df[["img_name", "ssim"]].copy()
        df_ssim["ssim"] = pd.to_numeric(df_ssim["ssim"], errors="coerce")
        df_ssim.rename(columns={"ssim": tname}, inplace=True)

        merged_psnr = df_psnr if merged_psnr is None else pd.merge(merged_psnr, df_psnr, on="img_name", how="outer")
        merged_ssim = df_ssim if merged_ssim is None else pd.merge(merged_ssim, df_ssim, on="img_name", how="outer")

    merged_psnr = merged_psnr.sort_values("img_name").reset_index(drop=True)
    merged_ssim = merged_ssim.sort_values("img_name").reset_index(drop=True)

    base, _ = os.path.splitext(csv_name)
    merged_psnr.to_csv(os.path.join(output_dir, f"{base}_psnr.csv"), index=False)
    merged_ssim.to_csv(os.path.join(output_dir, f"{base}_ssim.csv"), index=False)
    print("Saved:", os.path.join(output_dir, f"{base}_psnr.csv"))
    print("Saved:", os.path.join(output_dir, f"{base}_ssim.csv"))
