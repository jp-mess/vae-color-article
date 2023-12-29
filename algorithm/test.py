from PIL import Image
import requests
import torch
import numpy as np
from torchvision.transforms import CenterCrop, Resize
from torchvision.transforms import functional as TF
from sklearn.neural_network import MLPRegressor
from diffusers.models import AutoencoderKL
import models
import copy
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# VAE input dimension (can't be too big for the model this file uses)
device = 'cpu'
VAE_INPUT_DIM = 224

# Function to load an image from a URL
def load_image(url):
    return Image.open(requests.get(url, stream=True).raw)

# Function to process and encode the image
def process_and_encode(image, transforms, vae):
    processed_image = transforms(image)
    processed_image = TF.to_tensor(processed_image).unsqueeze(0)
    with torch.no_grad():
        encoded = vae.encode(processed_image).latent_dist
    mean = encoded.mean
    logvar = encoded.logvar
    return torch.cat((mean,logvar),dim=1)

# Function to apply regression model
def apply_regression(distorted, fixed, original_image):
    clf = MLPRegressor(activation='relu', hidden_layer_sizes=(32), alpha=0.001, random_state=20)
    clf.fit(distorted, fixed)

    height, width = original_image.size
    original_pixels = np.array(original_image) / 255
    original_pixels = original_pixels.reshape(width*height, 3)
    painted_pixels = clf.predict(original_pixels)
    painted_image = painted_pixels.reshape((width, height, 3))
    painted_image = np.clip(painted_image, 0, 1)
    painted_image = np.uint8(painted_image * 255)
    return painted_image

# Function to display images
def display_images(images, titles, save_path=None):
    fig, axes = plt.subplots(1, len(images))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

def balance_latent_representation(encoded, latent_balancer):
    with torch.no_grad():
        adjusted_encoded = latent_balancer(encoded)
    return adjusted_encoded

def decode_latent(encoded, vae):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    sampled_mean = DiagonalGaussianDistribution(encoded).sample()
    decoded = vae.decode(sampled_mean).sample.squeeze(0)
    return decoded.permute((1,2,0)).detach().numpy()

def color_mapping(distorted, fixed, original_image):
    """
    Learn a color mapping from a distorted image to a fixed image using MLPRegressor.

    Args:
    - distorted (numpy.ndarray): The distorted image array of shape (VAE_INPUT_DIM, VAE_INPUT_DIM, 3).
    - fixed (numpy.ndarray): The fixed image array of shape (VAE_INPUT_DIM, VAE_INPUT_DIM, 3).
    - original_image (PIL.Image): The original image to apply the learned color mapping.

    Returns:
    - numpy.ndarray: The original image with the learned color mapping applied.
    """

    # Reshape images for MLPRegressor
    distorted_flat = distorted.reshape(VAE_INPUT_DIM*VAE_INPUT_DIM, 3)
    fixed_flat = fixed.reshape(VAE_INPUT_DIM*VAE_INPUT_DIM, 3)

    # Initialize and fit the MLPRegressor
    clf = MLPRegressor(activation='relu', hidden_layer_sizes=(32), alpha=0.001, random_state=20)
    clf.fit(distorted_flat, fixed_flat)

    # Reshape the images back to their original shape
    distorted = distorted.reshape((VAE_INPUT_DIM, VAE_INPUT_DIM, 3))
    fixed = fixed.reshape((VAE_INPUT_DIM, VAE_INPUT_DIM, 3))

    # Prepare the original image for color mapping
    height, width = original_image.size
    original_image_array = np.array(original_image) / 255
    original_pixels = original_image_array.reshape(width*height, 3)

    # Apply the learned color mapping
    painted_pixels = clf.predict(original_pixels)
    painted_pixels = np.array(painted_pixels).clip(0, 1)
    painted_image = np.uint8(painted_pixels.reshape((width, height, 3)) * 255)

    return painted_image

def recursive_color_correction(original_image, iterations, transforms, vae, latent_balancer):
    """
    Apply iterative color correction on an image.

    Args:
    - original_image (PIL.Image): The original image to process.
    - iterations (int): Number of times to apply the color correction iteratively.
    - transforms (torchvision.transforms): Image transformations to apply.
    - vae (Model): The Variational Autoencoder model for encoding.
    - latent_balancer (Model): The balancing model to adjust the VAE's output.

    Returns:
    - PIL.Image: The final color-corrected image after specified iterations.
    """

    adjusted_image = original_image
    for _ in range(iterations):
        # Process and encode the image
        encoded = process_and_encode(adjusted_image, transforms, vae)

        # Fix the colors in the encoding
        fixed_encoded = balance_latent_representation(copy.deepcopy(encoded), latent_balancer)

        fixed_subimage = decode_latent(fixed_encoded, vae)
        distorted_subimage = decode_latent(encoded, vae)


        # Apply color mapping
        painted_image_array = color_mapping(distorted_subimage, fixed_subimage, adjusted_image)

        # Update the adjusted_image for the next iteration
        adjusted_image = Image.fromarray(painted_image_array)

    return adjusted_image

def get_adjusted_image_path(original_image_path):
    """
    Generate the file path for the adjusted image.

    Args:
    - original_image_path (str): File path of the original image.

    Returns:
    - str: File path for the adjusted image.
    """

    # Extract the directory and filename from the original image path
    original_dir, filename = os.path.split(original_image_path)

    # Define the new directory for the adjusted image
    adjusted_dir = os.path.join(original_dir, 'adjusted')

    # Create the adjusted directory if it does not exist
    if not os.path.exists(adjusted_dir):
        os.makedirs(adjusted_dir)

    # Construct the path for the adjusted image
    adjusted_image_path = os.path.join(adjusted_dir, filename)

    return adjusted_image_path

def load_image(image_path):
    return Image.open(image_path)

def display_before_after(before_path, after_path, dpi=100):
    """
    Display a comparison of two images: 'before' and 'after',
    arranged based on their aspect ratio, and save in 'plots' directory.

    Args:
    - before_path (str): File path of the 'before' image.
    - after_path (str): File path of the 'after' image.
    - dpi (int, optional): DPI for saving the image. Defaults to 100.
    """

    # Load images
    original_image = Image.open(before_path)
    painted_image = Image.open(after_path)

    # Determine layout based on image aspect ratio
    width, height = original_image.size
    if width > height:
        # Landscape orientation: stack images vertically
        fig, axes = plt.subplots(2, 1)
    else:
        # Portrait orientation: side by side images
        fig, axes = plt.subplots(1, 2)

    # Display 'before' and 'after' images
    axes = axes.ravel()  # Flatten axes array for uniform handling
    axes[0].imshow(original_image)
    axes[0].set_title('Before')
    axes[0].axis('off')

    axes[1].imshow(painted_image)
    axes[1].set_title('After')
    axes[1].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Prepare the directory for saving the plot
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Determine the file name for the plot
    stem = Path(before_path).stem
    plot_file_path = os.path.join(plots_dir, stem + '.png')

    # Save the figure
    plt.savefig(plot_file_path, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()

    return plot_file_path



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Color Correction")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--iterations", type=int, default=1, help="Number of recursive iterations")
    args = parser.parse_args()

    # Load the image
    original_image = load_image(args.image)

    # Define transformations and model
    transforms = torch.nn.Sequential(
        CenterCrop((min(original_image.size), min(original_image.size))),
        Resize((VAE_INPUT_DIM, VAE_INPUT_DIM)),
    )
    
    # Someone else trained this (see: StableDiffusion)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # I trained this (network in models.py)
    latent_balancer = models.FCN()
    balance_network = "checkpoints/latent_balancer.pth"
    latent_balancer.load_state_dict(torch.load(balance_network))
    latent_balancer.to(device)

    # Perform the color / lighting correction
    adjusted_image = recursive_color_correction(original_image, args.iterations, transforms, vae, latent_balancer)

    # Save outputs
    adjusted_image_path = get_adjusted_image_path(args.image)
    adjusted_image.save(adjusted_image_path)
    
    plot_path = display_before_after(args.image, adjusted_image_path)
    print("Plot saved at:", plot_path)


if __name__ == "__main__":
    main()
