import cv2
import numpy as np

def compute_losses(img1_path, img2_path):
    # Read images
    img1 = cv2.imread(img1_path).astype(np.float32)
    img2 = cv2.imread(img2_path).astype(np.float32)

    # Ensure same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # Compute L1 loss (mean absolute difference)
    l1_loss = np.mean(np.abs(img1 - img2))

    # Compute L2 loss (mean squared difference)
    l2_loss = np.mean((img1 - img2) ** 2)

    return l1_loss, l2_loss


if __name__ == "__main__":
    # Example usage
    camera_image = "datasets/defense/3/img002.png"
    software_upscaled = "datasets/defense/3/software_up.jpg"
    thesis_upscaled = "datasets/defense/3/thesis_up.webp"

    l1, l2 = compute_losses(camera_image, software_upscaled)
    print("Software")
    print(f"L1 Loss (MAE): {l1}")
    print(f"L2 Loss (MSE): {l2}")
    l1, l2 = compute_losses(camera_image, thesis_upscaled)
    print("Thesis")
    print(f"L1 Loss (MAE): {l1}")
    print(f"L2 Loss (MSE): {l2}")
