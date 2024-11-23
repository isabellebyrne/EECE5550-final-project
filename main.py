from model import FeatureExtractorCNN
import torch
import torch.nn as nn
import torch.optim as optim
from similarityMatrix import similarityMatrix, visualizeMatrix
import numpy as np

#temorary solution for the problem with the KMP duplicate library
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_loop_images(batch_size=128, channels=3, height=32, width=32):
    """
    Generates a tensor of 32x32 images that simulate a loop structure.

    Parameters:
    - batch_size: int, the number of images to generate (default: 128).
    - channels: int, the number of channels in each image (default: 3).
    - height: int, the height of each image (default: 32).
    - width: int, the width of each image (default: 32).

    Returns:
    - images: torch.Tensor, a tensor of shape (batch_size, channels, height, width).
    """
    images = torch.zeros((batch_size, channels, height, width))
    angles = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
    
    for i, theta in enumerate(angles):
        # Create a base grid for the image
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xv, yv = np.meshgrid(x, y)
        
        circular_pattern = np.sin(2 * np.pi * (xv * np.cos(theta) + yv * np.sin(theta)))
        normalized_pattern = (circular_pattern - circular_pattern.min()) / (circular_pattern.max() - circular_pattern.min())
        
        for c in range(channels):
            images[i, c] = torch.tensor(normalized_pattern, dtype=torch.float32)

    # Add random noise to simulate natural variability
    alpha = 0.05
    images += torch.randn_like(images) * alpha

    return images


if __name__ == "__main__":
    dummy_input = torch.randn(128, 3, 32, 32) # Batch of 128 images with feature vectors of size 128
    dummy_loop_matrix = generate_loop_images() # Generate a loop matrix for visualization
    print(dummy_input.shape)
    print(dummy_loop_matrix.shape)
    model = FeatureExtractorCNN()
    for input in [dummy_input, dummy_loop_matrix]:
        features = model(input).detach().numpy()
        s_matrix = similarityMatrix(features)
        visualizeMatrix(s_matrix)
