import numpy as np
import spaceform_pca_lib as sfpca
import os
import pandas as pd
from scipy.stats import wasserstein_distance
import scipy.stats as stats
######################################################################
######################################################################
######################################################################
class parameters:
    def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
        self.D = D
        self.d = d
        self.N = N
        self.sigma = sigma
######################################################################
######################################################################
######################################################################
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
    G = np.matmul(X.T, X)
    np.clip(G, -1, 1, out=G)
    DM = np.arccos(G)
    return DM
######################################################################
######################################################################
######################################################################
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
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
######################################################################
######################################################################
######################################################################
def compute_jsdm(X):
    X = X.T
    X = X ** 2
    N = np.shape(X)[0]
    JSDM = np.zeros((N, N))
    for i in range(N):
        x = X[i, :]
        for j in range(i + 1, N):
            y = X[j, :]
            JSDM[i, j] = jensen_shannon_divergence(x, y)
            JSDM[j, i] = JSDM[i, j]
    return JSDM
######################################################################
######################################################################
######################################################################
def distance_distortion(DM_, DM):
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
#dataset_name = 'GUniFrac'  # exlude pga,
dataset_name = 'doi_10_5061_dryad_pk75d__v20150519' # exlude liu pga
# Directory where to save the results
directory = "../results/" + dataset_name + "/"
if not os.path.exists(directory):
    os.makedirs(directory)

results_filename = os.path.join(directory, "distortions_results.csv")
if not os.path.exists(results_filename):
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
    KLDM = compute_kldm(X)
    TVDM = compute_tvdm(X)
    JSDM = compute_jsdm(X)

    # A list of functions to run
    functions = [
        sfpca.estimate_spherical_subspace,
        #sfpca.estimate_spherical_subspace_liu,
        sfpca.estimate_spherical_subspace_dai,
        #sfpca.estimate_spherical_subspace_pga,
        sfpca.estimate_spherical_subspace_pga_2,
    ]

    # Prepare a list to collect the data
    data_list = []
    for d in range(2,D):
        print(f"Running with d = {d}")
        param = parameters(D=D, d=d, N=N, sigma = 0)
        cnt = 0
        for func in functions:
            cnt = cnt +1
            directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
            X_ = np.load(directory_+'X_.npy')
            S_ = np.load(directory_+'S_.npy', allow_pickle=True)

            AIDM_ = compute_aidm(X_)
            KLDM_ = compute_kldm(X_)
            DM_ = compute_sdm(X_)
            TVDM_ = compute_tvdm(X_)
            JSDM_ = compute_jsdm(X_)

            sphere_dist = total_distance(X, X_)
            dist_distortion = distance_distortion(DM_, DM)
            ai_distortion = distance_distortion(AIDM_, AIDM)
            kl_distortion = distance_distortion(KLDM_, KLDM)
            tv_distortion = distance_distortion(TVDM_, TVDM)
            js_distortion = distance_distortion(JSDM_, JSDM)



            data_list.append({"d": d, "Method": func.__name__, 
                "sphere_dist":sphere_dist,
                "ai_distortion":ai_distortion, 
                "kl_distortion":kl_distortion,
                "js_distortion":js_distortion,
                "tv_distortion":tv_distortion,
                "dist_distortion": dist_distortion
                })
            print("[",sphere_dist, ai_distortion, kl_distortion, js_distortion, tv_distortion,dist_distortion, "]")
            results_df = pd.DataFrame(data_list)
        print('############################################')
        results_df.to_csv(results_filename, index=False)
        if d >= min(np.shape(X))-1:
            break