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
        self.d = 20 ## target dimension
        self.sigma = 0.1 # noise std
param = parameters()
##############################################################
X , S, noise_lvl = sfpca.random_spherical_data(param)
X_, S_ = sfpca.estimate_spherical_subspace(X,param)
##############################################################
error = 0
N = param.N
for n in range(N):
    x = X[:,n]
    x_ = X_[:,n]
    error = error + np.arccos( np.inner(x,x_) )/N
##############################################################    
print(error)
print(noise_lvl)