import pandas as pd

train_folder_names = ["3.main", "4.only_spatial_path", "5.only_frequency_path", "6.without_domain_swap", "7.without_cross_attention", "8.no_spatial_encoder", "9.without_frequency_encoder", "10.conv", "11.wide", "12.deep"]

valid_file_names = ["base", "only_spatial_path", "only_frequency_path", "without_domain_swap", "without_cross_attention", "without_spatial_encoder", "without_frequency_encoder", "conv", "base-wide", "base-deep"]

idx = 2

for i in range(10):
    if i == 7:
        idx = 3

    train_file = f"0/train_data/{train_folder_names[i]}/{train_folder_names[i][idx:]}_LOSS.csv"
    valid_file = f"0/valid_data/L1_Y/valid_{valid_file_names[i]}.csv"

    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)

    train_data.drop(columns=["Wall time"], inplace=True)
    valid_data.drop(columns=["Wall time"], inplace=True)

    train_data.rename(columns={"Value": "Train_Loss"}, inplace=True)
    valid_data.rename(columns={"Value": "Valid_Loss"}, inplace=True)

    final_data = pd.merge(train_data, valid_data, on="Step", how="inner")

    final_data.to_csv(f"0/train_valid_loss/{train_folder_names[i][idx:]}.csv", index=False)

    train_file = f"0/train_data/{train_folder_names[i]}/{train_folder_names[i][idx:]}_PSNR.csv"
    valid_file = f"0/valid_data/PSNR_Y/valid_{valid_file_names[i]}.csv"

    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)

    train_data.drop(columns=["Wall time"], inplace=True)
    valid_data.drop(columns=["Wall time"], inplace=True)

    train_data.rename(columns={"Value": "Train_PSNR"}, inplace=True)
    valid_data.rename(columns={"Value": "Valid_PSNR"}, inplace=True)

    final_data = pd.merge(train_data, valid_data, on="Step", how="inner")

    final_data.to_csv(f"0/train_valid_psnr/{train_folder_names[i][idx:]}.csv", index=False)

    train_file = f"0/train_data/{train_folder_names[i]}/{train_folder_names[i][idx:]}_SSIM.csv"
    valid_file = f"0/valid_data/SSIM_Y/valid_{valid_file_names[i]}.csv"

    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)

    train_data.drop(columns=["Wall time"], inplace=True)
    valid_data.drop(columns=["Wall time"], inplace=True)

    train_data.rename(columns={"Value": "Train_SSIM"}, inplace=True)
    valid_data.rename(columns={"Value": "Valid_SSIM"}, inplace=True)

    final_data = pd.merge(train_data, valid_data, on="Step", how="inner")

    final_data.to_csv(f"0/train_valid_ssim/{train_folder_names[i][idx:]}.csv", index=False)

