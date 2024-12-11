import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import os
import random
import cv2

class CustomPairwiseDataset(Dataset):
    def __init__(self, img_directory, matrix_file):
        self.img_directory = img_directory
        self.matrix = np.loadtxt(matrix_file, delimiter=',')
        self.image_files = sorted(os.listdir(img_directory))
        self.matrix_shape = self.matrix.shape
        self.pairs = self._create_pairs()
        self._sample_pairs()

        # Focal length:
        fc = [ 367.481519978327754 , 366.991059667167065 ]
        # Principal point:
        cc = [ 328.535778962615268 , 233.779960757465176 ]
        # Distortion coefficients:
        kc = [ -0.293510277812333,
                0.065334967950619,
            -0.000117308680498,
                0.000304779905426,
                0.000000000000000 ]
        
        # Camera matrix
        self.camera_matrix = np.array([
            [fc[0], 0, cc[0]],
            [0, fc[1], cc[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.im_width  = 640
        self.im_height = 480

        # Distortion coefficients
        self.dist_coeffs = np.array(kc, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)
    
    def _create_pairs(self):
        pairs = []
        n = self.matrix.shape[0]
        for i in range(n):
            for j in range(i):
                if i==j:
                    continue
                label = self.matrix[i, j]
                pairs.append((i, j, label))
        return pairs
    
    def _sample_pairs(self):
        #very large number of 0s compared to 1s in pair labels
        #sample pairs with label 0 such that they are equal to number of
        #pairs with label 1

        # Separate pairs by label
        label_1_pairs = [(r, c, l) for (r, c, l) in self.pairs if l == 1]
        label_0_pairs = [(r, c, l) for (r, c, l) in self.pairs if l == 0]

        # Sample label 0 pairs
        n_label_0 = len(label_0_pairs)
        n_label_1 = len(label_1_pairs)
        print("n_0:", n_label_0, "n_1:", n_label_1)

        random.seed(42)
        sampled_label_0_pairs = random.sample(label_0_pairs, n_label_1)
        self.pairs = sampled_label_0_pairs + label_1_pairs
        random.shuffle(self.pairs)
        print("Total pairs after sampling:", len(self.pairs))

    def _transform(self, img_x):
        #normalize image x
        transform = transforms.Compose([
            transforms.ToTensor(),
            #using mean and standard deviation of imagenet dataset
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize((221, 221), antialias=True)
        ])
        return transform(img_x)

    def __getitem__(self, index):

        img1_idx, img2_idx, label = self.pairs[index]
        img1_path = os.path.join(self.img_directory, self.image_files[img1_idx])
        img2_path = os.path.join(self.img_directory, self.image_files[img2_idx])
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        undistorted_img1 = cv2.undistort(img1, self.camera_matrix, self.dist_coeffs)
        undistorted_img2 = cv2.undistort(img2, self.camera_matrix, self.dist_coeffs)

        transformed_img1 = self._transform(undistorted_img1)
        transformed_img2 = self._transform(undistorted_img2)
        
        return transformed_img1, transformed_img2, torch.tensor(label, dtype=torch.float32)
    
def reverse_transform(tensor):
    """
    Reverse the preprocessing transformations: Normalize -> Convert to PIL Image
    """
    # Imagenet mean and std for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Undo normalization
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    
    # Clip values to [0, 1] (in case normalization introduces out-of-bounds values)
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to NumPy array
    img = tensor.permute(1, 2, 0).numpy()  # C x H x W -> H x W x C for Matplotlib
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255] for visualization
    
    return img

def plot_dataloader_samples(dataloader, num_samples=5, figsize=(10, 10)):
    # Get one batch of data
    img1_batch, img2_batch, labels_batch = next(iter(dataloader))
    
    # Define a grid for displaying images
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    fig.suptitle("Sanity Check: img1 vs img2 and Corresponding Labels", fontsize=16)

    for i in range(num_samples):
        # Convert tensors to numpy for plotting
        img1 = reverse_transform(img1_batch[i])  # Convert tensor to PIL image
        img2 = reverse_transform(img2_batch[i])  # Convert tensor to PIL image
        label = labels_batch[i].item()
        
        # Plot img1
        
        axes[i, 0].imshow(img1)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"img1")

        # Plot img2
        axes[i, 1].imshow(img2)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"img2\nLabel: {label}")
    
    plt.tight_layout()
    plt.show()

def createCombinedDataset(image_dirs, gt_matrix_paths):

    datasets = []
    for image_dir, gt_matrix in zip(image_dirs, gt_matrix_paths):
        datasets.append(CustomPairwiseDataset(image_dir, gt_matrix))
    
    combined_dataset = ConcatDataset(datasets)

    return combined_dataset

if __name__ == "__main__":

    # Check for CustomPairwiseDataset

    image_dirs = ['data/CityCentre_Images',
              'data/NewCollege_Images']
    gt_matrix_paths = ['data/CityCentreTextFormat.txt',
                   'data/NewCollegeTextFormat.txt']

    combined_dataset = createCombinedDataset(image_dirs, gt_matrix_paths)

    #Split dataset into train and testsets
    train_size = int(0.8 * len(combined_dataset))
    validation_size = len(combined_dataset) - train_size
    train_dataset, validation_dataset = random_split(combined_dataset, [train_size, validation_size], 
                                                     generator=torch.Generator().manual_seed(42))

    #print length of combined dataset
    print('Length of Combined Dataset:',len(combined_dataset))

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    plot_dataloader_samples(train_dataloader, num_samples=5)