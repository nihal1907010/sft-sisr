import os
import argparse
from PIL import Image

def center_crop_patch(hr_path, lr_path, out_hr_path, out_lr_path, size=256):
    """Center-crop 256x256 from HR and corresponding 64x64 from LR."""
    hr_img = Image.open(hr_path)
    lr_img = Image.open(lr_path)

    # HR size
    hr_w, hr_h = hr_img.size
    x = (hr_w - size) // 2
    y = (hr_h - size) // 2

    # Crop HR
    hr_patch = hr_img.crop((x, y, x + size, y + size))

    # For LR: scale coordinates since HR is 4x LR
    lr_size = size // 4
    lr_x, lr_y = x // 4, y // 4
    lr_patch = lr_img.crop((lr_x, lr_y, lr_x + lr_size, lr_y + lr_size))

    hr_patch.save(out_hr_path)
    lr_patch.save(out_lr_path)


def process_folders(hr_dir, lr_dir, out_hr_dir, out_lr_dir, size=256):
    os.makedirs(out_hr_dir, exist_ok=True)
    os.makedirs(out_lr_dir, exist_ok=True)

    hr_files = sorted(os.listdir(hr_dir))
    lr_files = sorted(os.listdir(lr_dir))

    for hr_file, lr_file in zip(hr_files, lr_files):
        hr_path = os.path.join(hr_dir, hr_file)
        lr_path = os.path.join(lr_dir, lr_file)

        out_hr_path = os.path.join(out_hr_dir, f"patch_{hr_file}")
        out_lr_path = os.path.join(out_lr_dir, f"patch_{lr_file}")

        center_crop_patch(hr_path, lr_path, out_hr_path, out_lr_path, size)
        print(f"Saved patches: {out_hr_path}, {out_lr_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, default="datasets/Valid/DIV2K/HR", help="Path to HR image folder")
    parser.add_argument("--lr_dir", type=str, default="datasets/Valid/DIV2K/LRx4", help="Path to LR image folder")
    parser.add_argument("--out_hr_dir", type=str, default="datasets/Valid/DIV2K/HR_center_crop", help="Output folder for HR patches")
    parser.add_argument("--out_lr_dir", type=str, default="datasets/Valid/DIV2K/LRx4_center_crop", help="Output folder for LR patches")
    parser.add_argument("--size", type=int, default=256, help="HR patch size")
    args = parser.parse_args()

    process_folders(args.hr_dir, args.lr_dir, args.out_hr_dir, args.out_lr_dir, args.size)


if __name__ == "__main__":
    main()
