import numpy as np
import matplotlib.pyplot as plt


def similarityMatrix(m1):
    m = np.zeros([len(m1), len(m1)])
    for i, u in enumerate(m1):
        for j, v in enumerate(m1):
            if j < i:
                u = u / np.linalg.norm(u)
                v = v / np.linalg.norm(v)
                diff = u - v
                s = np.dot(diff, diff)
                m[i][j] = s
                m[j][i] = s
    m = 1 - m / np.max(m)
    return m


def visualizeMatrix(m):
    plt.matshow(m)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    m1 = np.random.random((500, 500)) # 'array of i random feature vectors for i images
    visualizeMatrix(similarityMatrix(m1))
    

    