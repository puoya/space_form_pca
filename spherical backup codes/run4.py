import numpy as np
import spaceform_pca_lib as sfpca
import os
import pandas as pd
from scipy.stats import wasserstein_distance
import scipy.stats as stats
import random


class parameters:
    def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
        self.D = D
        self.d = d
        self.N = N
        self.sigma = sigma
######################################################################
######################################################################
######################################################################
# Define the compute_aidm function
def compute_aidm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    D = np.shape(X)[1]

    Y = np.zeros((N,D))
    gmean = np.zeros((N,1))
    for n in range(N):
        x = X[n,:]+min(10**(-6), 0.01/D)
        Y[n,:] = np.log(x/ stats.gmean(x))
    G = np.matmul(Y,Y.T)
    dG = np.reshape(np.diag(G), (N,1))
    dG = np.matmul(dG,np.ones((1,N)) )
    DM= np.sqrt(-2*G + dG + dG.T)
    return DM
######################################################################
######################################################################
######################################################################
# Define the compute_wdm function
def compute_wdm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    DM = np.zeros((N,N))
    for i in range(N):
        x = X[i,:]
        for j in range(i+1,N):
            y = X[j,:]
            DM[i,j] = wasserstein_distance(x, y)
            DM[j,i] = DM[i,j]
    return DM
######################################################################
######################################################################
######################################################################
# Define the compute_sdm function
def compute_sdm(X):
    """
    Computes the spherical distance matrix given the point set X_ 
    Parameters:
    - X: The points in a spherical space
    Returns:
    - DM: The spherical distance matrix
    """
    G = np.matmul(X.T, X)
    np.clip(G, -1, 1, out=G)
    DM = np.arccos(G)
    return DM
######################################################################
######################################################################
######################################################################
# Define the compute_tvdm function
def compute_tvdm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    DM = np.zeros((N,N))
    for i in range(N):
        x = X[i,:]
        for j in range(i+1,N):
            y = X[j,:]
            DM[i,j] = total_variation(x, y)
            DM[j,i] = DM[i,j]
    return DM
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
# Define the compute_kldm function
def compute_kldm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    DM = np.zeros((N,N))
    for i in range(N):
        x = X[i,:]
        for j in range(i+1,N):
            y = X[j,:]
            DM[i,j] = kl_divergence(x, y)
            DM[j,i] = DM[i,j]
    return DM
######################################################################
######################################################################
######################################################################
# Define the distance distortion function
def distance_distortion(DM_, DM):
    """
    Computes the distance distortion given the estimated X_ and the true distance matrix DM.
    Parameters:
    - X_: The estimated points in the new subspace.
    - DM: The true distance matrix.
    Returns:
    - error: The computed distortion.
    """
    distortion = np.linalg.norm(DM - DM_)/ np.linalg.norm(DM) * 100
    return distortion
######################################################################
######################################################################
######################################################################
def total_wasserstein_distance(X, Y):
    X = X.T
    Y = Y.T
    X = X**2
    Y = Y**2
    # Initialize total distance
    total_distance = 0.0
    # Iterate over each pair of distributions
    for x, y in zip(X, Y):
        # Compute Wasserstein distance between the distributions
        distance = wasserstein_distance(x, y)
        # Add the distance to the total
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def total_variation_distance(X, Y):
    X = X.T
    Y = Y.T
    X = X**2
    Y = Y**2
    # Initialize total distance
    total_distance = 0.0
    # Iterate over each pair of distributions
    for x, y in zip(X, Y):
        # Compute Wasserstein distance between the distributions
        distance = total_variation(x, y)
        # Add the distance to the total
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def total_distance(X, Y):
    X = X.T
    Y = Y.T
    # Initialize total distance
    total_distance = 0.0
    # Iterate over each pair of distributions
    for x, y in zip(X, Y):
        # Compute Wasserstein distance between the distributions
        g = np.matmul(x.T, y)
        g = max(g,-1)
        g = min(g,1)
        distance = np.arccos(g)
        # Add the distance to the total
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def aitchison_distance(x, y):
    """
    Compute the Aitchison distance between two compositions.
    Parameters:
    x (array-like): First composition.
    y (array-like): Second composition.
    Returns:
    float: Aitchison distance between x and y.
    """
    D = len(x)
    x = np.array(x)+min(10**(-6), 0.01/D)
    y = np.array(y)+min(10**(-6), 0.01/D)
    # Pseudo-inverse of the compositions
    inv_x = np.log(x / stats.gmean(x))
    inv_y = np.log(y / stats.gmean(y))
    # Compute the Aitchison distance
    dist = np.sqrt(np.sum((inv_x - inv_y) ** 2))
    return dist
######################################################################
######################################################################
######################################################################
def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two distributions.
    
    Parameters:
    p (array-like): First distribution.
    q (array-like): Second distribution.

    Returns:
    float: KL divergence from P to Q.
    """
    # Ensure the distributions are numpy arrays and avoid division by zero
    D = len(p)
    p = np.array(p, dtype=np.float64) + min(10**(-6), 0.01/D)
    q = np.array(q, dtype=np.float64) + min(10**(-6), 0.01/D)
    
    # Compute the KL divergence
    return np.sum(p * np.log(p / q))
######################################################################
######################################################################
######################################################################
def total_variation(p, q):
    """Compute the Total Variation distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))
######################################################################
######################################################################
######################################################################
def total_aitchison_distance(X, Y):
    """
    Compute the total Aitchison distance between pairs of compositions.
    Parameters:
    X (array-like): Matrix of compositions, where each row represents a composition.
    Y (array-like): Matrix of compositions, where each row represents a composition.
    Returns:
    float: Total Aitchison distance between compositions in X and Y.
    """
    X = X.T
    Y = Y.T
    X = X**2
    Y = Y**2
    # Initialize total distance
    total_distance = 0.0
    # Iterate over each pair of compositions
    for x, y in zip(X, Y):
        # Compute Aitchison distance between the compositions
        distance = aitchison_distance(x, y)
        # Add the distance to the total
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def load_data(dataset_name):
    if dataset_name == 'GUniFrac':
        address = "/Users/puoya.tabaghi/Downloads/clean/spherical/data/GUniFrac/doc/csv/"
        X = np.load(address+'X.npy')
        X = X.T
    elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
        address = "/Users/puoya.tabaghi/Downloads/clean/spherical/data/doi_10_5061_dryad_pk75d__v20150519/"
        X = np.load(address+'X.npy')
        X = X.T
        N, D = np.shape(X)
        for n in range(N):
            X[n,:] = X[n,:]/ np.sum(X[n,:])
            X[n,:] = np.sqrt(X[n,:])
        X = X.T
    elif dataset_name == 'document':
        address = "/Users/puoya.tabaghi/Downloads/clean/spherical/data/document/"
        X = np.load(address+'X.npy').astype(float)
        address = "/Users/puoya.tabaghi/Downloads/clean/spherical/data/document/"
        label = np.load(address+'label.npy')
        idx = (label==0) + (label == 1)
        X = X[idx,:]
        label = label[idx]
        np.random.seed(42)
        N0 = 400
        N = np.shape(X)[0]
        idx = np.random.choice(N, N0, replace=False)
        X = X[idx,:]
        for n in range(N0):
            X[n,:] = X[n,:]/ np.sum(X[n,:])
            X[n,:] = np.sqrt(X[n,:])
            #print(np.sum(X[n,:]**2))
        X = X.T
    return X
######################################################################
######################################################################
######################################################################

#dataset_name = 'GUniFrac'
#dataset_name = 'doi_10_5061_dryad_pk75d__v20150519'
dataset_name = 'document'

# Directory where to save the results
directory = "../results/" + dataset_name + "/"
if not os.path.exists(directory):
    os.makedirs(directory)

results_filename = os.path.join(directory, "distortions_results.csv")
if os.path.exists(results_filename):
    # Load the DataFrame from the CSV file
    results_df = pd.read_csv(results_filename)
    # Print the loaded results
    print("Results already exist. Loaded results:")
else:
    print("Results file does not exist. Computing the results...")
    X = load_data(dataset_name)
    # Initial parameters
    D,N = np.shape(X)
    D = D-1
    DM = compute_sdm(X)
    AIDM = compute_aidm(X)
    #WDM = compute_wdm(X)
    KLDM = compute_kldm(X)
    TVDM = compute_tvdm(X)
    #print(KLDM)
    # A list of functions to run
    functions = [
        sfpca.estimate_spherical_subspace,
        #sfpca.estimate_spherical_subspace_liu,
        #sfpca.estimate_spherical_subspace_dai,
        #sfpca.estimate_spherical_subspace_pga,
        sfpca.estimate_spherical_subspace_pga_2,
    ]

    # Prepare a list to collect the data
    data_list = []
    # Main execution loop
    for d in range(2,D):
        print(f"Running with d = {d}")
        param = parameters(D=D, d=d, N=N, sigma = 0)
        cnt = 0
        for func in functions:
            cnt = cnt +1
            # Execute function
            X_, S_ = func(X, param)
            # Calculate distortion
            #
            #
            # Calculate distortion
            #WDM_ = compute_wdm(X_)
            AIDM_ = compute_aidm(X_)
            KLDM_ = compute_kldm(X_)
            DM_ = compute_sdm(X_)
            TVDM_ = compute_tvdm(X_)
            dist_distortion = distance_distortion(DM_, DM)
            #wass_distortion = distance_distortion(WDM_, WDM)
            ai_distortion = distance_distortion(AIDM_, AIDM)
            kl_distortion = distance_distortion(KLDM_, KLDM)
            tv_distortion = distance_distortion(TVDM_, TVDM)
            
            sphere_dist = total_distance(X, X_)
            aitchison_dist = total_aitchison_distance(X, X_)
            #wass_dist = total_wasserstein_distance(X, X_)
            tv_dist = total_variation_distance(X, X_)

            # data_list.append({"d": d, "Method": func.__name__, "dist_distortion": dist_distortion,  "wass_distortion":wass_distortion,"ai_distortion":ai_distortion,
            #     "kl_distortion":kl_distortion, "sphere_dist":sphere_dist,"aitchison_dist":aitchison_dist, "wass_dist":wass_dist })
            data_list.append({"d": d, "Method": func.__name__, 
                "dist_distortion": dist_distortion, "sphere_dist":sphere_dist,
                "ai_distortion":ai_distortion, "aitchison_dist":aitchison_dist,
                 "kl_distortion":kl_distortion, #"wass_dist":wass_dist, 
                 "tv_distortion":tv_distortion, "tv_dist":tv_dist})
            print(cnt, "[", dist_distortion, ai_distortion, kl_distortion, tv_distortion, sphere_dist, aitchison_dist,tv_dist, "]")
        # Convert the list to a DataFrame
        results_df = pd.DataFrame(data_list)
        #print(results_df)
        # Save the DataFrame to CSV
        results_df.to_csv(results_filename, index=False)
        print(f"Results saved to {results_filename}")
        if dist_distortion < .1:
            break
    