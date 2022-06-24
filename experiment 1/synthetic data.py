import sys
sys.path.append('../')
import numpy as np
import spaceform_pca_lib as sfpca
import scipy.linalg
##############################################################
class parameters:
    def __init__(self):
        self.N = 10000 ## no. of points
        self.D = 10 ## input dimension
        self.d = 1 ## target dimension
        self.sigma = 0.000 # noise std
param = parameters()
##############################################################
X , S, _ = sfpca.random_spherical_data(param)
#print(X)
# for n in range(10000):
#     x = X[:,n]
#     print(np.linalg.norm(x))
X_, S_ = sfpca.estimate_spherical_subspace(X,param)
dist = sfpca.subspace_dist(S,S_)
print(dist)
##############################################################