import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import spaceform_pca_lib as sfpca
import os
import pandas as pd
from scipy.stats import wasserstein_distance
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_mutual_information(X, y):
    #print(X)
    X = X**2
    #print(X)
    """
    Compute mutual information between features X and target y.

    Parameters:
    - X: Feature matrix of shape (N, D+1) containing spherical points
    - y: Labels array of shape (N,) containing multiclass labels

    Returns:
    - mutual_information: Array of shape (D+1,) containing mutual information for each feature
    """

    # Compute mutual information
    mutual_infos = []
    for label in np.unique(y):
        y_binary = (y == label).astype(int)
        mutual_info = mutual_info_classif(X, y_binary)
        mutual_infos.append(mutual_info)
    # Compute the mean mutual information across all class labels
    #print(mutual_infos)
    mean_mutual_information = np.mean(mutual_infos)

    # Compute mutual information
    #mutual_information = mutual_info_classif(X, y_binary)

    # Compute the mean mutual information
    #mean_mutual_information = np.mean(mutual_information)



    return mean_mutual_information
######################################################################
######################################################################
######################################################################
def compute_pearson_correlation(X, y):
    """
    Compute the Pearson correlation coefficient between features X and target y.

    Parameters:
    - X: Feature matrix of shape (N, D+1) containing spherical points
    - y: Labels array of shape (N,) containing multiclass labels

    Returns:
    - pearson_correlation: Array of shape (D+1,) containing Pearson correlation coefficients
    """

    # Compute Pearson correlation coefficient for each feature
    # print(X)
    # for n in range(np.shape(X)[0]):
    #     print(np.linalg.norm(X[n,:]))
    X = X**2
    # for n in range(np.shape(X)[0]):
    #     print(np.sum(X[n,:]))
    #     print(n)
    # print(X)
    pearson_correlations = []
    for label in np.unique(y):
        y_binary = (y == label).astype(int)
        pearson_correlation = np.array([pearsonr(X[:, i], y_binary)[0] for i in range(X.shape[1])])
        pearson_correlations.append(np.abs(pearson_correlation))
    # Compute the mean mutual information across all class labels
    #print(mutual_infos)
    mean_pearson_correlation = np.mean(pearson_correlations)

    # pearson_correlation = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    # mean_pearson_correlation = np.mean( np.abs(pearson_correlation))

    return mean_pearson_correlation
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
        address = "/Users/puoya.tabaghi/Downloads/clean/spherical/data/GUniFrac/doc/csv/label.csv"
        y = np.load('/Users/puoya.tabaghi/Downloads/clean/spherical/data/GUniFrac/doc/csv/SmokingStatus_categories.npy')
    elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
        address = "/Users/puoya.tabaghi/Downloads/clean/spherical/data/doi_10_5061_dryad_pk75d__v20150519/"
        X = np.load(address+'X.npy')
        X = X.T
        N, D = np.shape(X)
        for n in range(N):
            X[n,:] = X[n,:]/ np.sum(X[n,:])
            X[n,:] = np.sqrt(X[n,:])
        X = X.T

        y = np.load('/Users/puoya.tabaghi/Downloads/clean/spherical/data/doi_10_5061_dryad_pk75d__v20150519/age_categories.npy') 
        #BMI_group_categories.npy age_categories.npy
        #np.set_printoptions(threshold=sys.maxsize)

        #print(y)
    return X,y
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
class FourLayerNN(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(FourLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_neural_network(X, y, test_size=0.1, stratify=None, seed=None, num_iterations=10):
    accuracies = []

    for iteration in range(num_iterations):
        # Set random seed for reproducibility
        np.random.seed(seed+iteration)
        torch.manual_seed(seed+iteration)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=seed)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # Define neural network
        input_dim = X.shape[1]
        hidden1_dim = 512
        hidden2_dim = 128
        hidden3_dim = 32
        output_dim = len(np.unique(y))
        model = FourLayerNN(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test, predicted.numpy())
            accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    return average_accuracy

dataset_name = 'GUniFrac'
#dataset_name = 'doi_10_5061_dryad_pk75d__v20150519'
# Directory where to save the results
directory = "../results/" + dataset_name + "/"
if not os.path.exists(directory):
    os.makedirs(directory)

results_filename = os.path.join(directory, "classification_smoking.csv")
if not os.path.exists(results_filename):
    # Load the DataFrame from the CSV file
    results_df = pd.read_csv(results_filename)
    # Print the loaded results
    print("Results already exist. Loaded results:")
else:
    print("Results file does not exist. Computing the results...")
    X,y = load_data(dataset_name)
    # Initial parameters
    D,N = np.shape(X)
    D = D-1
    functions = [
        sfpca.estimate_spherical_subspace,
        sfpca.estimate_spherical_subspace_liu,
        sfpca.estimate_spherical_subspace_dai,
        sfpca.estimate_spherical_subspace_pga,
        sfpca.estimate_spherical_subspace_pga_2,
    ]

    # Prepare a list to collect the data
    data_list = []
    #ind = y != 3
    # Main execution loop
    for d in range(2,D):
        print(f"Running with d = {d}")
        param = parameters(D=D, d=d, N=N, sigma = 0)
        cnt = 0
        seed = 100000*d
        for func in functions:
            cnt = cnt +1
            # Execute function
            X_, S_ = func(X, param)
            #np.matmul(X_.T, (S_.H))
            average_mi = compute_mutual_information(X_.T, y)
            average_pc = compute_pearson_correlation(X_.T,y)
            #print(average_pc)
            average_acc = train_neural_network(X_.T**2, y, seed=seed)
            #print(average_pc)
            #train_neural_network( np.matmul(X_.T, (S_.H)), y,seed)
            #print(X_)
            # Calculate distortion
            #
            #
            # Calculate distortion
            

            data_list.append({"d": d, "Method": func.__name__, "average_mi": average_mi ,"average_pc": average_pc
                ,"average_acc": average_acc})
            #     "kl_distortion":kl_distortion, "sphere_dist":sphere_dist,"aitchison_dist":aitchison_dist, "wass_dist":wass_dist })
            # data_list.append({"d": d, "Method": func.__name__, 
            #     "dist_distortion": dist_distortion, "sphere_dist":sphere_dist,
            #     "ai_distortion":ai_distortion, "aitchison_dist":aitchison_dist,
            #      "kl_distortion":kl_distortion, "wass_dist":wass_dist, 
            #      "tv_distortion":tv_distortion, "tv_dist":tv_dist})
            print(cnt, "[", average_mi,average_pc,average_acc, "]")
        # Convert the list to a DataFrame
        results_df = pd.DataFrame(data_list)
        # Save the DataFrame to CSV
        results_df.to_csv(results_filename, index=False)
        print(f"Results saved to {results_filename}")
        if d > 60:
            break
    

