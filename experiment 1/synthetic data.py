import sys
sys.path.append('../')
import numpy as np
import spaceform_pca_lib as sfpca
import scipy.linalg
##############################################################
class parameters:
    def __init__(self):
        self.N = 1000 ## no. of points
        self.D = 100 ## input dimension
        self.d = 9 ## target dimension
        self.sigma = .1 # noise std
param = parameters()
##############################################################
X , S, noise_lvl_input = sfpca.random_spherical_data(param)
X_, S_ = sfpca.estimate_spherical_subspace(X,param)
noise_lvl_output = sfpca.compute_noise_lvl(X_,S)
dist = sfpca.subspace_dist(S,S_)
print(noise_lvl_input)
print(noise_lvl_output)
print(dist)
##############################################################