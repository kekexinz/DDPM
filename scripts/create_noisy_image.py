import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

def create_noisy_images(clean_image, noise_levels, normalize=True):
    """
    Generate a series of noisy images with increasing Gaussian noise levels.
    
    Args:
        clean_image (torch.Tensor): The clean input image (C, H, W) or (1, H, W).
        noise_levels (list): List of noise standard deviations (sigma) for Gaussian noise.
        normalize (bool): Whether to clip and normalize the noisy images to [-1, 1].
    
    Returns:
        torch.Tensor: Tensor of shape (len(noise_levels), C, H, W) containing noisy images.
    """
    noisy_images = []

    for sigma in noise_levels:
        # Generate Gaussian noise
        noise = torch.randn_like(clean_image) * sigma

        # Add noise to the clean image
        noisy_image = clean_image + noise

        # Normalize to [-1, 1] or [0, 1] if needed
        if normalize:
            noisy_image = torch.clamp(noisy_image, -1, 1)

        noisy_images.append(noisy_image)

    return torch.stack(noisy_images)

# Load a clean MNIST digit
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# MNIST dataset
mnist_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Extract a clean image of digit "3"
digit_3_image = None
for image, label in mnist_dataset:
    if label == 3:
        digit_3_image = image  # Shape: (1, 28, 28)
        break

if digit_3_image is None:
    raise ValueError("Digit '3' not found in the dataset!")

# Create noisy versions of the clean image
noise_levels = [1.0, 2.0,5.0,10.0]  # Increasing noise standard deviations
noisy_images = create_noisy_images(digit_3_image, noise_levels)

# Visualize the noisy images
plt.figure(figsize=(10, 5))
for i, noisy_image in enumerate(noisy_images):
    plt.subplot(1, len(noise_levels), i + 1)
    plt.imshow(noisy_image.squeeze().cpu().numpy(), cmap="gray")
    plt.title(f"σ={noise_levels[i]}")
    plt.axis("off")
plt.show()