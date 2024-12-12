import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

#temorary solution for the problem with the KMP duplicate library
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_similarity_matrix(feature_vectors):
    """
    Compute the similarity matrix for visual loop closure detection.

    Args:
        feature_vectors (np.ndarray): CNN feature vectors of shape (n_images, 500).

    Returns:
        np.ndarray: Similarity matrix of shape (n_images, n_images).
    """
    feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)
    print("Feature Vectors Shape:", feature_vectors.shape)

    normalized_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    n_images = normalized_vectors.shape[0]
    print('n_images:', n_images)
    
    distances = np.zeros((n_images, n_images))
    for i in range(n_images):
        for j in range(n_images):
            distances[i, j] = np.linalg.norm(normalized_vectors[i] - normalized_vectors[j])

    max_distance = np.max(distances)
    similarity_matrix = 1 - (distances / max_distance)
    return similarity_matrix


def dummy_transform_images_to_feature_vectors(images):
    feature_vectors = []
    random_L_kernel = np.random.rand(1, 221)
    random_R_kernel = np.random.rand(221, 500)
    for image in images:
        image_np = image.numpy()
        feature_vector = random_L_kernel @ image_np @ random_R_kernel
        feature_vectors.append(feature_vector)
    return np.array(feature_vectors)


def generate_loop_images(batch_size=128, channels=1, height=200, width=32):
    """
    Generates a tensor of nxn images that simulate a loop structure.

    Parameters:
    - batch_size: int, the number of images to generate (default: 128).
    - channels: int, the number of channels in each image (default: 1).
    - height: int, the height of each image (default: 200).
    - width: int, the width of each image (default: 200).

    Returns:
    - images: torch.Tensor, a tensor of shape (batch_size, channels, height, width).
    """
    images = torch.zeros((batch_size, channels, height, width))
    angles = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
    for i, theta in enumerate(angles):
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xv, yv = np.meshgrid(x, y) # base grid
        circular_pattern = np.sin(2 * np.pi * (xv * np.cos(theta) + yv * np.sin(theta)))
        normalized_pattern = (circular_pattern - circular_pattern.min()) / (circular_pattern.max() - circular_pattern.min())
        for c in range(channels):
            images[i, c] = torch.tensor(normalized_pattern, dtype=torch.float32)
    alpha = 0.05
    images += torch.randn_like(images) * alpha # gausian noise
    return images


def display_matrix(matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='hot')
    plt.title("Similarity Matrix Heatmap")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.legend(loc='best')
    plt.show()


def demo_similarity_matrix():
    images = generate_loop_images(height=221, width=221, batch_size=200)
    feature_vectors = dummy_transform_images_to_feature_vectors(images)
    similarity_matrix = compute_similarity_matrix(feature_vectors)
    print("Similarity Matrix:")
    print(similarity_matrix)
    display_matrix(similarity_matrix)


if __name__ == "__main__":
    demo_similarity_matrix()