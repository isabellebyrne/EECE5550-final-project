#load model files from directory
import torch 
from torch.functional import F

def pca(features, reduced_dim = 500):

    features_l2_norm = F.normalize(features, p=2, dim=1)
    features_l2_norm -= torch.mean(features_l2_norm, dim=1, keepdim=True)

    cov_matrix = torch.matmul(features_l2_norm.T, features_l2_norm) 
    U, S, V = torch.svd(cov_matrix)

    principal_components = U[:, :reduced_dim]
    projected_features = torch.matmul(features_l2_norm, principal_components)

    #S is your singular values
    return projected_features, S


def whitening(reduced_features, singular_values, epsilon=1e-5):

    singular_values = singular_values[:reduced_features.shape[1]]
    whitening_matrix = 1.0 / (torch.sqrt(singular_values) + epsilon)
    whitened_features = reduced_features * whitening_matrix

    return whitened_features