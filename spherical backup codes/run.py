import numpy as np
import spaceform_pca_lib as sfpca
import os
import pandas as pd
from scipy.stats import wasserstein_distance

class parameters:
    def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
        self.D = D
        self.d = d
        self.N = N
        self.sigma = sigma


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

# Define the distance distortion function
def distance_distortion(X_, DM):
    """
    Computes the distance distortion given the estimated X_ and the true distance matrix DM.
    
    Parameters:
    - X_: The estimated points in the new subspace.
    - DM: The true distance matrix.
    - idx: The indices for the upper triangular part of the distance matrix.
    
    Returns:
    - error: The computed distortion.
    """

    DM_ = compute_sdm(X_)
    N = np.shape(DM_)[0]

    idx = np.triu(np.ones((N, N)), 1) == 1  # Upper triangular matrix indices

    distortion = np.linalg.norm(DM[idx] - DM_[idx])/ np.linalg.norm(DM[idx]) * 100
    return distortion



def total_wasserstein_distance(X, Y):
    # Initialize total distance
    total_distance = 0.0
    
    # Iterate over each pair of distributions
    N = 0
    for x, y in zip(X, Y):
        N = N +1
        # Compute Wasserstein distance between the distributions
        distance = wasserstein_distance(x, y)
        
        # Add the distance to the total
        total_distance += distance
    
    return total_distance

def aitchison_distance(x, y, epsilon=1e-20):
    """
    Compute the Aitchison distance between two compositions x and y.
    
    Parameters:
        x (array-like): First composition.
        y (array-like): Second composition.
    
    Returns:
        float: The Aitchison distance between x and y.
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Perturb zero entries
    x_perturbed = x + epsilon
    y_perturbed = y + epsilon
    
    # Compute the Aitchison distance
    log_ratio = np.log(x_perturbed) - np.log(y_perturbed)
    distance = np.sqrt(np.sum(log_ratio**2))
    
    return distance

def total_aitchison_distance(X, Y):
    """
    Compute the total sum of Aitchison distances between two matrices of compositions X and Y.
    
    Parameters:
        X (array-like): First matrix of compositions (NxD).
        Y (array-like): Second matrix of compositions (NxD).
    
    Returns:
        float: The total sum of Aitchison distances between X and Y.
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Get the number of compositions and dimensionality
    N, D = X.shape
    
    # Initialize total distance
    total_distance = 0.0
    # Compute Aitchison distances between each pair of compositions
    for n in range(N):
        x = X[n,:]
        y = Y[n,:]
        total_distance += aitchison_distance(x, y)
    
    return total_distance

# Directory where to save the results
directory = "data/GUniFrac/results/distance_distorions/"
#directory = 'data/doi_10_5061_dryad_pk75d__v20150519/results/distance_distorions/'
results_filename = directory+ "distortions_results.csv"
# directory = "/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/doc/csv/"
# results_filename = os.path.join(directory, "throat_otu.csv")
# df = pd.read_csv(results_filename)
# df_dropped = df.drop(columns=['Unnamed: 0'])
# df_dropped = df_dropped.astype(float)
# X = df_dropped.to_numpy()
# #print(type(X))
# for n in range(np.shape(X)[0]):
#     X[n,:] = X[n,:] / np.sum(X[n,:])
#     X[n,:] = np.sqrt(X[n,:])
# np.save("/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/doc/csv/X.npy",X)
    


# directory = "/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/"
# directory = "data/doi_10_5061_dryad_pk75d__v20150519/"
# #name = 'BMI_group'
# name = "SmokingStatus"
# labels = np.load(directory + name+"_categories.npy")


# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)


if os.path.exists(results_filename):
    # Load the DataFrame from the CSV file
    results_df = pd.read_csv(results_filename)
    # Print the loaded results
    print("Results already exist. Loaded results:")
if os.path.exists(results_filename):
    print("Results file does not exist. Computing the results...")

    address = "/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/doc/csv/"
    address = 'data/doi_10_5061_dryad_pk75d__v20150519/'
    X = np.load(address+'X.npy')
    X = X.T
    D,N = np.shape(X)
    for n in range(N):
        X[:,n] = X[:,n] / np.sum(X[:,n])
        X[:,n] = np.sqrt(X[:,n])
    DM = compute_sdm(X)
    # Initial parameters

    D = D-1
    d_values = range(1,30)  # Example d values, adjust as necessary

    # A list of functions to run
    functions = [
        sfpca.estimate_spherical_subspace,
        sfpca.estimate_spherical_subspace_liu,
        sfpca.estimate_spherical_subspace_dai,
        sfpca.estimate_spherical_subspace_pga,
        sfpca.estimate_spherical_subspace_pga_2,
    ]


    # Prepare a list to collect the data
    data_list = []


    # Main execution loop
    for d in d_values:
        print(f"Running with d = {d}")
        param = parameters(D=D, d=d, N=N, sigma = 0)
        
        for func in functions:
            print(func.__name__)
            # Execute function
            X_, S_ = func(X, param)
            #distortion = total_wasserstein_distance( (X.T)**2,(X_.T)**2)
            distortion = total_wasserstein_distance( (X_.T)**2,(X.T)**2)
            #print(np.linalg.norm((X_.T)-(X.T)**2))
            
        
            # DM_ = compute_sdm(X_)
            # DM1 = DM_[labels == 1,:]
            # DM1 = DM1[:,labels == 1]

            # tmp = np.sum(DM1)
            # DM1 = DM_[labels == 0,:]
            # DM1 = DM1[:,labels == 0]
            # tmp = tmp + np.sum(DM1)


            # print("Method", func.__name__, "Distortion", tmp/(np.sum(DM_)-tmp))
            #data_list.append({"d": d, "Method": func.__name__, "Distortion": tmp/(np.sum(DM_)-tmp) })

            # Calculate distortion
            #distortion = distance_distortion(X_, DM)
            #print(distortion)
            data_list.append({"d": d, "Method": func.__name__, "Distortion": distortion})
            print("d", d, "Method", func.__name__, "Distortion", distortion)

    # Convert the list to a DataFrame
    results_df = pd.DataFrame(data_list)

    # Save the DataFrame to CSV
    results_filename = os.path.join(directory, "distortions_results.csv")
    results_df.to_csv(results_filename, index=False)

    print(f"Results saved to {results_filename}")