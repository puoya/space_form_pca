import numpy as np
import spaceform_pca_lib as sfpca

class parameters:
    def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
        self.D = D
        self.d = d
        self.N = N
        self.sigma = sigma

param = parameters()
H = sfpca.random_orthogonal_matrix(param)
print(H)
print(np.matmul(H.T,H))