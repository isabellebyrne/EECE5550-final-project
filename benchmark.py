import numpy as np
from similarityMatrix import similarityMatrix

def compare_symmetric_matrices(A, B):
    distance = np.linalg.norm(A - B, ord="fro")
    return distance

def evaluate_feature_extrator_performance(model, input, loop_matrix):
    features = model(input).detach().numpy()
    s_matrix = similarityMatrix(features)
    loop_s_matrix = similarityMatrix(loop_matrix)
    distance = compare_symmetric_matrices(s_matrix, loop_s_matrix)
    return distance