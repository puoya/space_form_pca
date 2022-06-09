import sys
sys.path.append('../')
import numpy as np
import spaceform_pca_lib as sfpca
import scipy.linalg
##############################################################
class parameters:
    def __init__(self):
        self.N = 10000 ## no. of points
        self.D = 100 ## input dimension
        self.d = 10 ## target dimension
        self.sigma = 0.3 # noise std
param = parameters()
##############################################################
X , S, _ = sfpca.random_spherical_data(param)
X_, S_ = sfpca.estimate_spherical_subspace(X,param)
dist = sfpca.subspace_dist(S,S_)
print(dist)
##############################################################