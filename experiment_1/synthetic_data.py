from dis import dis
import sys
import time
sys.path.append('../')
import spaceform_pca_lib as sfpca
import numpy as np

##############################################################
class parameters:
    def __init__(self, N, D, d, sigma):
        self.N = N ## no. of points
        self.D = D ## input dimension
        self.d = d ## target dimension
        self.sigma = sigma # noise std

def experiment(N, D, d, sigma):
    #settings params
    param = parameters(N, D, d, sigma)

    X , S, noise_lvl_input = sfpca.random_spherical_data(param)

    X_, S_ = sfpca.estimate_spherical_subspace(X,param)
    noise_lvl_output = sfpca.compute_noise_lvl(X_,S)
    dist = sfpca.subspace_dist(S,S_)

    return noise_lvl_input, noise_lvl_output, dist

def experimentDataset(N, D, d, sigma):
    #settings params
    param = parameters(N, D, d, sigma)

    X , S, noise_lvl_input = sfpca.random_spherical_data(param)
    # X_, S_ = sfpca.estimate_spherical_subspace(X,param)
    # noise_lvl_output = sfpca.compute_noise_lvl(X_,S)
    # dist = sfpca.subspace_dist(S,S_)

    return X, S, noise_lvl_input, param


########################################################################
#Using generated data 

#save S_ it its own file called sfpa.npy, spga, dai


def sfpaFromData(X, S, param):
    start = time.time()
    X_, S_ = sfpca.estimate_spherical_subspace(X,param)
    end = time.time()
    runtime = round(end - start, 5) #runtime in seconds
    noise_lvl_output = sfpca.compute_noise_lvl(X_,S)
    dist = sfpca.subspace_dist(S,S_)
    
    return noise_lvl_output, dist, S_, runtime

def spgaFromData(X, S, param):
    start = time.time()
    X_, S_ = sfpca.estimate_spherical_subspace_pga(X,param)
    end = time.time()
    runtime = round(end - start, 5) #runtime in seconds
    noise_lvl_output = sfpca.compute_noise_lvl(X_,S)
    dist = sfpca.subspace_dist(S,S_)

    return noise_lvl_output, dist, S_, runtime

def daiFromData(X, S, param):
    start = time.time()
    X_, S_ = sfpca.estimate_spherical_subspace_dai(X,param)
    end = time.time()
    runtime = round(end - start, 5) #runtime in seconds 
    noise_lvl_output = sfpca.compute_noise_lvl(X_,S)
    dist = sfpca.subspace_dist(S,S_)

    return noise_lvl_output, dist, S_, runtime