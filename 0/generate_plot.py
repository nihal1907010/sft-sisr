import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.size'] = 20

# Formatter function for x-axis (iterations in K)
def k_formatter(x, pos):
    return f"{int(x/1000)}k"

formatter = FuncFormatter(k_formatter)

file_names = [
    "main", "only_spatial_path", "only_frequency_path", "without_domain_swap",
    "without_cross_attention", "no_spatial_encoder", "without_frequency_encoder",
    "conv", "wide", "deep"
]

for file_name in file_names:

    # ---- LOSS ----
    data = pd.read_csv(f"0/train_valid_loss/{file_name}.csv")
    plt.figure(figsize=(10, 8))
    plt.plot(data["Step"], data["Train_Loss"], marker=".", linestyle="-", label="Train Loss")
    plt.plot(data["Step"], data["Valid_Loss"], marker=".", linestyle="--", label="Valid Loss")
    plt.title("Train and Validation Loss over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.ylim(0, 0.1)
    plt.gca().xaxis.set_major_formatter(formatter)  # Apply K scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"0/train_valid_loss/{file_name}_loss.pdf")

    # ---- PSNR ----
    data = pd.read_csv(f"0/train_valid_psnr/{file_name}.csv")
    plt.figure(figsize=(10, 8))
    plt.plot(data["Step"], data["Train_PSNR"], marker=".", linestyle="-", label="Train PSNR")
    plt.plot(data["Step"], data["Valid_PSNR"], marker=".", linestyle="--", label="Valid PSNR")
    plt.title("Train and Validation PSNR over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR")
    plt.ylim(12, 33)
    plt.gca().xaxis.set_major_formatter(formatter)  # Apply K scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"0/train_valid_psnr/{file_name}_psnr.pdf")

    # ---- SSIM ----
    data = pd.read_csv(f"0/train_valid_ssim/{file_name}.csv")
    plt.figure(figsize=(10, 8))
    plt.plot(data["Step"], data["Train_SSIM"], marker=".", linestyle="-", label="Train SSIM")
    plt.plot(data["Step"], data["Valid_SSIM"], marker=".", linestyle="--", label="Valid SSIM")
    plt.title("Train and Validation SSIM over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("SSIM")
    plt.ylim(0.4, 1)
    plt.gca().xaxis.set_major_formatter(formatter)  # Apply K scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"0/train_valid_ssim/{file_name}_ssim.pdf")
