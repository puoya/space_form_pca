from math import ceil
import sys

sys.path.append('../')
import numpy as np
import spaceform_pca_lib as sfpca
import scipy.linalg
import time 

# num trials for each experiment
numIterations = 100

#range of values to use for each paramters
stdRange = [0.01, 0.05, 0.1] 
D_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
N_range = ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 
            4000, 5000, 6000, 7000, 8000, 9000, 10000])

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

##############################################################
def D_Exp(): 
    outputArray = np.zeros((numIterations, len(stdRange), len(D_range), 3))
    outFile = "D_exp"
    N = 10**4
    d = 1
    for iter in range(numIterations):
        for sigma_idx, sigma in enumerate(stdRange):
            for D_idx, D in enumerate(D_range):
                res = experiment(N, D, d, sigma) #result of exp with curr paramaters
                outputArray[iter][sigma_idx][D_idx][0] = res[0]
                outputArray[iter][sigma_idx][D_idx][1] = res[1]
                outputArray[iter][sigma_idx][D_idx][2] = res[2]

    np.save(outFile, outputArray)

def d_Exp(): 
    outputArray = np.zeros((numIterations, len(stdRange), len(d_range), 2))
    outFile = "d_exp"
    N = 10**4
    D = 100
    for iter in range(numIterations):
        for sigma_idx, sigma in enumerate(stdRange):
            for d_idx, d in enumerate(d_range):
                res = experiment(N, D, d, sigma) #result of exp with curr paramaters
                outputArray[iter][sigma_idx][d_idx][0] = res[0]
                outputArray[iter][sigma_idx][d_idx][1] = res[1]

    np.save(outFile, outputArray)

def N_Exp(): 
    outputArray = np.zeros((numIterations, len(stdRange), len(N_range), 3))
    outFile = "N_exp"
    D = 100
    d = 1
    for iter in range(numIterations):
        for sigma_idx, sigma in enumerate(stdRange):
            for N_idx, N in enumerate(N_range):
                res = experiment(N, D, d, sigma) #result of exp with curr paramaters
                outputArray[iter][sigma_idx][N_idx][0] = res[0]
                outputArray[iter][sigma_idx][N_idx][1] = res[1]
                outputArray[iter][sigma_idx][N_idx][2] = res[2]
    np.save(outFile, outputArray)

def std_exp():
    long_std_range = np.arange(0.05, 5.02, 0.05)
    outputArray = np.zeros((numIterations, len(long_std_range)))
    outFile = "std_exp"
    N = 10000
    D = 100
    d = 1
    for iter in range(numIterations):
        for sigma_idx, sigma in enumerate(long_std_range):
                sigma = round(sigma, 2)
                outputArray[iter][sigma_idx] = experiment(N, D, d, sigma)[0]
    np.save(outFile, outputArray)


start = time.time()
# D_Exp()
# d_Exp()
# N_Exp()
std_exp()
end = time.time()
print( round((end - start)/60, 2) )