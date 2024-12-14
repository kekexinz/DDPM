import argparse
import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
from ddpm import script_utils
import imageio
import torchvision.transforms as transforms
import os

def save_diffusion_sequence_as_gif(diffusion_sequence, save_path, gif_name="diffusion_process.gif"):
    """
    Save a diffusion sequence as a GIF.
    
    Args:
        diffusion_sequence (list): List of tensors representing the diffusion sequence.
        save_path (str): Directory where the GIF will be saved.
        gif_name (str): Name of the output GIF file.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Transform to convert tensor to PIL image
    to_pil = transforms.ToPILImage()

    # List to store frames for the GIF
    frames = []

    for i, tensor_image in enumerate(diffusion_sequence):
        # Normalize the tensor to [0, 1] range for visualization
        normalized_image = ((tensor_image + 1) / 2).clip(0, 1)

        # Convert each batch sample to an image and save frames
        for j in range(normalized_image.size(0)):  # Iterate over batch
            pil_image = to_pil(normalized_image[j])  # Convert to PIL image
            frames.append(pil_image)
    print(i)
    # Save the frames as a GIF
    gif_path = os.path.join(save_path, gif_name)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    print(f"GIF saved to {gif_path}")

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


def main():
    # Load noisy image
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
    plt.imshow(noisy_images[1].squeeze().numpy(), cmap='gray')
    plt.savefig("results/noisy_image.png")



    args = create_argparser().parse_args()
    device = args.device
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path))
        print("Model loaded")

        if args.use_labels:
            for label in range(10):
                y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                samples = diffusion.sample(args.num_images // 10, device, y=y)

                for image_id in range(len(samples)):
                    image = ((samples[image_id] + 1) / 2).clip(0, 1)
                    torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
        else:
            print("Sampling images")
            noisy_image = noisy_images[1].unsqueeze(0)
            samples = diffusion.sample_diffusion_sequence(args.num_images, device, noisy_image=noisy_image)
            save_diffusion_sequence_as_gif(samples, args.save_dir)
            #print(samples[0].shape)
            #for image_id in range(len(samples)):
            #    image = ((samples[image_id] + 1) / 2).clip(0, 1)
            #    torchvision.utils.save_image(image, f"{args.save_dir}/test_{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1, device=device)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()