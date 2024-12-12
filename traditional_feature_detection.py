import numpy as np
import cv2 as cv
from similarityMatrix import compute_similarity_matrix, display_matrix
import matplotlib.pyplot as plt

orb = cv.ORB_create()
features = []
for i in range(1, 2147):
    img = cv.imread(f'data/NewCollege_Images/NewCollege_Images/{i:04}.jpg')
    keypoints, descriptors = orb.detectAndCompute(img, None)
    try:
        flattened = descriptors.flatten()
        if flattened.shape[0] != 16000:
            print(f'shape: {flattened.shape}')
            continue
        features.append(flattened)
    except:
        print(f'issue at {i}')
        continue
    
newcollege_fv = np.array(features)
print(newcollege_fv.shape)

s_matrix = compute_similarity_matrix(newcollege_fv)
display_matrix(s_matrix)