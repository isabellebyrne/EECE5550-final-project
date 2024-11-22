from model import FeatureExtractorCNN
import torch
import torch.nn as nn
import torch.optim as optim
from similarityMatrix import similarityMatrix, visualizeMatrix

#temorary solution for the problem with the KMP duplicate library
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    dummy_input = torch.randn(128, 3, 32, 32) # Batch of 128 images with feature vectors of size 128
    model = FeatureExtractorCNN()
    features = model(dummy_input).detach().numpy()
    s_matrix = similarityMatrix(features)
    visualizeMatrix(s_matrix)
