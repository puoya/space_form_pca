import sys
sys.path.append('../')
import spaceform_pca_lib as sfpca

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