import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
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
