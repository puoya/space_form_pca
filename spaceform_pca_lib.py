import os
import glob
import torch
import pickle
import logging

import numpy as np
import pandas as pd
import scipy.linalg
from tqdm import tqdm

from Bio import Phylo
from pathlib import Path

import geom.poincare as poincare
import geom.euclidean as euclidean
from learning.frechet import Frechet
from scipy.sparse.linalg import eigs
import geom.hyperboloid as hyperboloid
from geom.horo import busemann, project_kd
from learning.pca import EucPCA, TangentPCA,PGA, HoroPCA, BSA
from utils.metrics import compute_metrics
###########################################################################
###########################################################################
###########################################################################
def extract_trees(dataset_name):
    # Define the output directory and create it if it doesn't exist
    output_dir = "datasets/" + dataset_name + "/output_trees"
    os.makedirs(output_dir, exist_ok=True)

    # Log and print information about parsing the trees
    tqdm.write("Parse tree files.")
    logging.info("Parse tree files.")

    # Define the source file directory
    source_file_path = "datasets/" + dataset_name + "/tree_directory"
    
    # Check if the source file directory exists
    if not os.path.exists(source_file_path):
        tqdm.write("The source file directory does not exist.")
        logging.info("The source file directory does not exist.")
        return

    directory = Path(source_file_path)
    # Get all .tre and .trees files from the directory
    all_files = list(directory.glob('*.tre')) + list(directory.glob('*.trees'))
    file_names = [file.stem for file in all_files] 

    # If no .tre or .trees files are found, log and print the information
    if not all_files:
        tqdm.write(f"No .tre or .trees files found in the directory: {dataset_name}")
        logging.info(f"No .tre or .trees files found in the directory: {dataset_name}")
        return

    # Open each file and perform the desired operation
    i = 0
    for source_file_path in all_files:
        i += 1
        tree_name = "tree_" + str(i)
        n = 0

        # Count the number of lines (trees) in the source file
        with source_file_path.open("r") as source_file:
            num_lines = sum(1 for _ in source_file)

        # Initialize the progress bar for the current source file
        with tqdm(total=num_lines, unit="tree", dynamic_ncols=True) as pbar:
            with source_file_path.open("r") as source_file:
                # Log and print the filename and number of trees
                logging.info(f"Filename: {source_file_path.stem}")
                logging.info(f"Number of trees: {num_lines}")
                tqdm.write(f"Filename: {source_file_path.stem}")
                tqdm.write(f"Number of trees: {num_lines}")
                
                # Read each line from the file
                for line in source_file:
                    # Increment the counter
                    n += 1
                    # Construct the filename for each tree
                    filename = os.path.join(output_dir, f"{tree_name}_{n}.tre")
                    # Open a new file for writing the tree
                    with open(filename, "w") as tree_file:
                        # Write the tree (line) to the file
                        tree_file.write(line.strip() + '\n')  # strip() removes any extra whitespace/newline and '\n' adds a clean newline

                    # Update the progress bar
                    pbar.set_postfix({"file": n})
                    pbar.update(1)
###########################################################################
###########################################################################
###########################################################################
def process_all_trees(dataset_name):
    # Log and print information about the processing
    tqdm.write("Compute distance matrices for trees.")
    logging.info("Compute distance matrices for trees.")

    tqdm.write(f"Dataset: {dataset_name}")
    logging.info(f"Dataset: {dataset_name}")
    
    # Define input and output directories
    input_directory = "datasets/" + dataset_name + "/output_trees"

    # Check if the input directory exists
    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    output_directory = "datasets/" + dataset_name + "/distance_matrices"
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Get list of all .tre files in the input directory
    tree_files = [f for f in os.listdir(input_directory) if f.endswith(".tre")]

    # Check if there are any .tre files to process
    if len(tree_files) == 0:
        tqdm.write(f"No .tre files found in the directory.")
        logging.info(f"No .tre files found in the directory.")
        return

    # Loop through all .tre files with a progress bar
    with tqdm(total=len(tree_files), unit="file", dynamic_ncols=True) as pbar:
        for filename in tree_files:
            tree_path = os.path.join(input_directory, filename)
            # Calculate the distance matrix from the tree file
            distance_matrix = tree_distance_matrix(tree_path)
            # Prepare the output filename
            output_filename = "D_" + os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(output_directory, output_filename)
            # Save the distance matrix to a file
            np.save(output_path, distance_matrix)
            # Log and update progress bar
            pbar.set_postfix({"file": filename})
            logging.info(f"Processed file: {filename}")
            pbar.update(1)
###########################################################################
###########################################################################
###########################################################################
def tree_distance_matrix(tree_file):
    # Read the phylogenetic tree from the given file in Newick format
    tree = Phylo.read(tree_file, "newick")
    
    # Get all the terminal (leaf) nodes from the tree
    terminals = tree.get_terminals()
    
    # Determine the number of terminal nodes
    n = len(terminals)
    
    # Initialize an n x n matrix to store the distances between terminals
    distance_matrix = np.zeros((n, n))
    
    # Iterate over each pair of terminals to compute distances
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the distance between terminal i and terminal j
            distance = tree.distance(terminals[i], terminals[j])
            
            # Since the distance matrix is symmetric, set both [i][j] and [j][i] to the computed distance
            distance_matrix[i][j] = distance_matrix[j][i] = distance
    
    # Return the completed distance matrix
    return distance_matrix
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_mds(dataset_name, dimension=None):
    # Log and print the start of the Naive Hyperbolic Embedding Step
    logging.info("Naive Hyperbolic Embedding Step.")
    tqdm.write("Naive Hyperbolic Embedding Step.")

    # Log and print the name of the dataset being processed
    tqdm.write(f"Dataset: {dataset_name}")
    logging.info(f"Dataset: {dataset_name}")

    # Define the input directory for distance matrices
    input_directory = f'datasets/{dataset_name}/distance_matrices'
    # Check if the input directory exists
    if not os.path.exists(input_directory):
        tqdm.write("The input directory does not exist.")
        logging.info("The input directory does not exist.")
        return

    # Define the output directory for hyperbolic points
    output_directory = f'datasets/{dataset_name}/hyperbolic_points'
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_directory) if f.endswith(".npy")]

    # If no .npy files are found, log and exit
    if len(npy_files) == 0:
        tqdm.write("No .npy files found in the directory.")
        logging.info("No .npy files found in the directory.")
        return

    # Set up a progress bar for processing files
    with tqdm(total=len(npy_files), unit="file", dynamic_ncols=True) as pbar:
        for filename in npy_files:
            file_path = os.path.join(input_directory, filename)
            # Load the distance matrix from the .npy file
            matrix = np.load(file_path)
            
            # Compute the scale factor to normalize the matrix
            scale = 10 / np.max(matrix)
            output_scale_file_path = os.path.join(output_directory, f"scale_{filename[7:-4]}.npy")
            # Save the scale factor
            np.save(output_scale_file_path, np.array([scale]))

            # Scale the distance matrix
            matrix_scaled = scale * matrix
            # Compute the Gramian matrix
            gramian = -np.cosh(matrix_scaled)
            
            # Determine the number of points (N) and the embedding dimension
            N = np.shape(gramian)[0]
            if dimension is None:
                dimension = np.shape(gramian)[1] - 1  # Assume Gramian is square, and dimension is one less
            
            # Compute points from the Gramian matrix
            X = lgram_to_points(dimension, gramian)

            # Transform and project each vector in X to the hyperbolic space
            for n in range(N):
                x = X[:, n]
                X[:, n] = project_vector_to_hyperbolic_space(x)
            
            # Save the resulting X matrix
            output_X_file_path = os.path.join(output_directory, f"X_{filename[7:-4]}.npy")
            np.save(output_X_file_path, X)
            
            # Log the processed file
            logging.info(f"Processed file: {filename}")
            pbar.set_postfix({"file": filename})
            pbar.update(1)
###########################################################################
###########################################################################
###########################################################################
def compute_pca_results(dataset_name, method):
    """
    Compute PCA results for a given dataset using the specified method.

    Parameters:
    dataset_name (str): Name of the dataset.
    method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

    Returns:
    None
    """
    logging.info("Hyperbolic PCA Step.")
    tqdm.write("Hyperbolic PCA Step.")

    # Check if the provided method is valid
    if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
        tqdm.write(f"Method {method} is not supported.")
        logging.error(f"Method {method} is not supported.")
        return
    
    # Define directory paths
    distance_file = 'hyperbolic_points'
    input_directory = f'datasets/{dataset_name}/{distance_file}' 

    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    output_directory = f'datasets/{dataset_name}/{method}/subspaces' 
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    logging.info(f"Dataset: {dataset_name} \t Method: {method}")
    tqdm.write(f"Dataset: {dataset_name} \t Method: {method}")

    # List all .npy files in the input directory that start with 'X'
    npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
    if len(npy_files) == 0:
        tqdm.write(f"No .npy files found in the directory.")
        logging.info(f"No .npy files found in the directory.")
        return

    # Set up a progress bar to show the progress of processing files
    with tqdm(total=len(npy_files), unit="file", dynamic_ncols=True) as pbar:    
        for filename in npy_files:
            try:
                # Get the full path of the current file
                file_path = os.path.join(input_directory, filename)
                # Extract the index from the filename (strip 'X' and '.npy')
                file_index = filename[2:-4]
                # Load the data from the .npy file
                X = np.load(file_path)
                
                # Perform different processing based on the method
                if method in ['sfpca', 'pga']:
                    if method == 'sfpca':
                        # Estimate the hyperbolic subspace using 'sfpca'
                        S = estimate_hyperbolic_subspace(X)
                    else:
                        # Estimate the hyperbolic subspace using 'pga'
                        S = estimate_hyperbolic_subspace_pga(X)
                    # Save the estimated subspace object as a .pkl file
                    with open(os.path.join(output_directory, 'subspace_' + file_index + '.pkl'), 'wb') as file:
                        pickle.dump(S, file)
                elif method in ['horopca', 'bsa']:
                    # Run dimensionality reduction using the specified method
                    Q, mu = run_dimensionality_reduction(method, X)
                    # Save the results as .pt files
                    torch.save(Q, os.path.join(output_directory, 'Q_' + file_index + '.pt'))
                    torch.save(mu, os.path.join(output_directory, 'mu_' + file_index + '.pt'))

                # Update the progress bar with the current file being processed
                pbar.set_postfix({"file": filename})
                pbar.update(1)
                logging.info(f"Processed file: {filename}")

            except Exception as e:
                # Log any errors that occur during processing
                logging.error(f"Error processing file {filename}: {e}")
                continue
###########################################################################
###########################################################################
###########################################################################
def compute_mds_results(dataset_name, method):
    """
    Compute MDS results for a given dataset using the specified method.

    Parameters:
    dataset_name (str): Name of the dataset.
    method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

    Returns:
    None
    """
    logging.info("Compute MDS Costs.")
    tqdm.write("Compute MDS Costs.")

    # Check if the provided method is valid
    if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
        tqdm.write(f"Method {method} is not supported.")
        logging.error(f"Method {method} is not supported.")
        return

    # Define directory paths
    distance_file = 'hyperbolic_points'
    base_directory = f'datasets/{dataset_name}'
    subspace_directory = f'{base_directory}/{method}/subspaces'
    input_directory = f'{base_directory}/{distance_file}'

    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    tqdm.write(f"Dataset:{dataset_name} \t Method:{method}")
    logging.info(f"Dataset:{dataset_name} \t Method:{method}")

    if not os.path.exists(subspace_directory):
        tqdm.write(f"The subspace directory does not exist.")
        logging.info(f"The subspace directory does not exist.")
        return

    output_directory = f'{base_directory}/{method}/mds'
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    results_df = pd.DataFrame(columns=['file_index', 'dimension', 'mds_error'])

    # Iterate over all relevant files in the input directory

    npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
    if len(npy_files) == 0:
        tqdm.write(f"No .npy files found in the directory.")
        logging.info(f"No .npy files found in the directory.")
        return

    with tqdm(total=len(npy_files), unit=f"file", dynamic_ncols=True) as pbar:
        for filename in npy_files:
            file_path = os.path.join(input_directory, filename)
            # Extract the file index from the filename
            file_index = filename[2:-4]
            X = np.load(file_path)
            #print(f'MDS distortion: {method} \t file number:{file_index}' )
            distance_matrix = compute_hyperbolic_distance_matrix(X)

            if method in ['sfpca', 'pga']:
                # Load the subspace instance
                S = load_subspace_instance(subspace_directory, file_index)
                num_dimensions = np.shape(S.H)[1]
            elif method in ['horopca', 'bsa']:
                Q, mu_ref = load_subspace_instance_qmu(subspace_directory, file_index)
                num_dimensions = np.shape(Q)[0] + 1

            pbar.set_postfix({"file": filename})
            pbar.update(1)
            logging.info(f"Processed file: {filename}")

            # Iterate through each dimension and compute the distance matrix and MDS error
            for dimension in range(1, num_dimensions):
                inaccurate = False
                if method in ['sfpca','pga']:
                    distance_matrix_d, inaccurate = compute_distance_matrix(X, S, dimension=dimension, method=method) 
                else:
                    distance_matrix_d, inaccurate = compute_distance_matrix(X, Q, mu_ref, dimension, method)

                # Compute the MDS error
                mds_error = np.linalg.norm(distance_matrix - distance_matrix_d, 'fro') / np.linalg.norm(distance_matrix, 'fro')

                if inaccurate:
                    mds_error = np.nan

                results_df = pd.concat([results_df, pd.DataFrame({
                    'file_index': [file_index],
                    'dimension': [dimension],
                    'mds_error': [mds_error]
                })], ignore_index=True)
            
    # Save the results to a CSV file
    output_file_path = os.path.join(output_directory, 'distortions.csv')
    results_df.to_csv(output_file_path, index=False)
###########################################################################
###########################################################################
###########################################################################
def compute_quartet_results(dataset_name, method):
    """
    Compute Quartet Score results for a given dataset using the specified method.

    Parameters:
    dataset_name (str): Name of the dataset.
    method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

    Returns:
    None
    """
    logging.info("Compute Quartet Scores.")
    tqdm.write("Compute Quartet Scores.")

    # Check if the provided method is valid
    if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
        tqdm.write(f"Method {method} is not supported.")
        logging.error(f"Method {method} is not supported.")
        return

    # Define directory paths
    distance_file = 'hyperbolic_points'
    base_directory = f'datasets/{dataset_name}'
    subspace_directory = f'{base_directory}/{method}/subspaces'
    input_directory = f'{base_directory}/{distance_file}'
    
    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    if not os.path.exists(subspace_directory):
        tqdm.write(f"The subspace directory does not exist.")
        logging.info(f"The subspace directory does not exist.")
        return

    output_directory = f'{base_directory}/{method}/quartet'
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    logging.info(f"Dataset: {dataset_name} \t Method: {method}")
    tqdm.write(f"Dataset: {dataset_name} \t Method: {method}")

    # List all .npy files in the input directory that start with 'X'
    npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
    if len(npy_files) == 0:
        tqdm.write(f"No .npy files found in the directory.")
        logging.info(f"No .npy files found in the directory.")
        return

    L = 10**5
    results_df = pd.DataFrame(columns=['file_index', 'dimension', 'quartet_score'])

    n = 0
    # Iterate over all relevant files in the input directory
    with tqdm(total=len(npy_files), unit=f"file", dynamic_ncols=True) as pbar:
        for filename in npy_files:
            n = n + 1
            np.random.seed(n)

            file_path = os.path.join(input_directory, filename)
            # Extract the file index from the filename
            file_index = filename[2:-4]
            X = np.load(file_path)
            N = np.shape(X)[1]

            pbar.set_postfix({"file": filename})
            pbar.update(1)
            logging.info(f"Processed file: {filename}")

            distance_matrix = compute_hyperbolic_distance_matrix(X)

            random_matrix = np.random.uniform(0, 1, (2*L, 4))*N
            integer_matrix = select_unique_combinations(random_matrix,L)
            L_ = np.shape(integer_matrix)[0]

            topology = []
            for i in range(L_):
                index = integer_matrix[i,:]
                distance_matrix_i = distance_matrix[index,:].copy()
                distance_matrix_i = distance_matrix_i[:,index]
                topology.append( determine_best_topology(distance_matrix_i)) 
            topology = np.array(topology)


            if method in ['sfpca', 'pga']:
                # Load the subspace instance
                S = load_subspace_instance(subspace_directory, file_index)
                num_dimensions = np.shape(S.H)[1]
            elif method in ['horopca', 'bsa']:
                Q, mu_ref = load_subspace_instance_qmu(subspace_directory, file_index)                
                num_dimensions = np.shape(Q)[0] + 1

            # Iterate through each dimension and compute the distance matrix and MDS error
            for dimension in range(1, num_dimensions):
                if method in ['sfpca','pga']:
                    distance_matrix_d, inaccurate = compute_distance_matrix(X, S, dimension=dimension, method=method)  
                else:
                    distance_matrix_d,inaccurate = compute_distance_matrix(X, Q, mu_ref, dimension, method)

                topology_d = []
                for i in range(L_):
                    index = integer_matrix[i,:]
                    distance_matrix_di = distance_matrix_d[index,:].copy()
                    distance_matrix_di = distance_matrix_di[:,index]
                    topology_d.append( determine_best_topology(distance_matrix_di)) 
                topology_d = np.array(topology_d)

                accuracy =  np.sum(np.abs(topology_d-topology) == 0)/L_

                if inaccurate:
                    accuracy = np.nan
            
                results_df = pd.concat([results_df, pd.DataFrame({
                    'file_index': [file_index],
                    'dimension': [dimension],
                    'quartet_score': [accuracy]
                })], ignore_index=True)
            # # Save the results to a CSV file
            output_file_path = os.path.join(output_directory, 'accuracies.csv')
            results_df.to_csv(output_file_path, index=False)

    # # Save the results to a CSV file
    # output_file_path = os.path.join(output_directory, 'accuracies.csv')
    # results_df.to_csv(output_file_path, index=False)
###########################################################################
###########################################################################
###########################################################################
class subspace:
    def __init__(self, H = 0, Hp = 0, p = 0,subspace=0):
        self.H = H
        self.Hp = Hp
        self.p = p
        self.subspace = subspace
###########################################################################
###########################################################################
###########################################################################
def J_norm(v):
    """
    Compute the norm of a vector in the Lorentzian space defined by matrix J.

    Parameters:
    v (ndarray): The input vector.

    Returns:
    float: The computed norm.
    """
    # Construct the vector v*J
    vJ = v.copy()
    vJ[0] = -v[0]

    # Compute the norm in the Lorentzian space
    norm_value = np.dot(vJ.ravel(),v.ravel())
    
    return norm_value
###########################################################################
###########################################################################
###########################################################################
def determine_best_topology(distance_matrix):
    """
    Determine the topology based on the four-point condition for a given 4x4 distance matrix.

    Parameters:
    distance_matrix (ndarray): A 4x4 distance matrix.

    Returns:
    int: The topology index (1, 2, or 3).
    """

    # Calculate the sums of the opposite pairs
    sum_opposite_1 = distance_matrix[0, 1] + distance_matrix[2, 3]
    sum_opposite_2 = distance_matrix[0, 2] + distance_matrix[1, 3]
    sum_opposite_3 = distance_matrix[0, 3] + distance_matrix[1, 2]

    # Find the minimum sum among the three calculated sums
    minimum_sum = min(sum_opposite_1, sum_opposite_2, sum_opposite_3)

    # Determine the topology index based on which sum is the minimum
    if minimum_sum == sum_opposite_1:
        topology_index = 1
    elif minimum_sum == sum_opposite_2:
        topology_index = 2
    else:
        topology_index = 3

    return topology_index
###########################################################################
###########################################################################
###########################################################################
def select_unique_combinations(random_matrix, L):
    """
    Select unique integer combinations from a random matrix, ensuring no duplicates in each row.

    Parameters:
    random_matrix (np.ndarray): Matrix of random values.
    L (int): Number of unique rows to select.

    Returns:
    np.ndarray: Matrix with L unique rows.
    """
    # Convert the random matrix to integers
    integer_matrix = random_matrix.astype(int)
    
    # Sort each row to easily identify duplicates
    sorted_matrix = np.sort(integer_matrix, axis=1)
    
    # Identify rows with duplicate elements
    has_duplicates = np.any(sorted_matrix[:, :-1] == sorted_matrix[:, 1:], axis=1)
    
    # Filter out rows with duplicates
    unique_matrix = integer_matrix[~has_duplicates]
    
    # Select the first L unique rows
    unique_matrix = unique_matrix[:L, :]
    
    return unique_matrix
###########################################################################
###########################################################################
def compute_hyperbolic_exponential(Vt, S):
    """
    Compute the hyperbolic exponential map for given tangent vectors.

    Parameters:
    Vt (ndarray): Tangent vectors (DxN).
    S (object): An object containing the base point 'p'.

    Returns:
    ndarray: The resulting points in the hyperbolic space (DxN).
    """
    
    base_point = np.squeeze(S.p)
    
    # Get dimensions
    D = np.shape(Vt)[0] - 1
    N = np.shape(Vt)[1]
    
    # Initialize the result matrix
    result_matrix = np.zeros((D + 1, N))
    
    for n in range(N):
        v = Vt[:, n]
        norm_v = np.sqrt(J_norm(v))

        if norm_v != 0:
            sinh_norm_v_over_norm_v = np.sinh(norm_v) / norm_v
        else:
            sinh_norm_v_over_norm_v = 1

        x = np.cosh(norm_v) * base_point + sinh_norm_v_over_norm_v * v

        norm_x = J_norm(x)
        
        if norm_x > 0:
            print('Error: Norm should not be positive')
        
        # Normalize the result vector
        result_matrix[:, n] = x / np.sqrt(-norm_x)
    
    return result_matrix
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_log(X, S):
    """
    Compute the hyperbolic logarithm map for given points in hyperbolic space.

    Parameters:
    X (ndarray): Points in hyperbolic space (DxN).
    S (object): An object containing the base point 'p'.

    Returns:
    ndarray: The tangent vectors at the base point 'p' (DxN).
    """
    base_point = S.p
    
    # Get dimensions
    D, N = np.shape(X)
    D -= 1  # Adjust D to match the context of the hyperbolic space
    
    # Initialize the result matrix for tangent vectors
    tangent_vectors = np.zeros((D + 1, N))
    
    # Construct the J matrix for the hyperbolic space
    J = np.eye(D + 1)
    J[0, 0] = -1
    
    for n in range(N):
        x = X[:, n]
        
        # Compute the theta value
        theta = -np.matmul(np.matmul(x.T, J), base_point)
        theta = np.maximum(theta, 1)
        theta = np.arccosh(theta)
        
        # Compute the tangent vector
        if theta != 0:
            theta_over_sinh_theta = theta / np.sinh(theta)
        else:
            theta_over_sinh_theta = 1
        
        tangent_vectors[:, n] = theta_over_sinh_theta * (x - base_point * np.cosh(theta))
    
    return tangent_vectors
###########################################################################
###########################################################################
###########################################################################
def find_first_max_index(array):
    """
    Find the first index of the maximum value in the array.

    Parameters:
    array (ndarray): Input array.

    Returns:
    int: The first index of the maximum value, or -1 if the array is empty.
    """
    if array.size == 0:
        return -1  # Return -1 to indicate that the array is empty.
    
    max_index = np.argmax(array)  # Get the index of the maximum value
    
    return max_index
###########################################################################
###########################################################################
###########################################################################
def estimate_hyperbolic_subspace(X, d = None):
    """
    Estimate the hyperbolic subspace for the given data.

    Parameters:
    X (ndarray): Input data matrix (DxN).
    parameters (object): An object containing parameters 'd', 'N', and 'D'.

    Returns:
    tuple: A tuple containing the subspace object and eigenvalues.
    """
    D,N = np.shape(X)
    D -= 1
    if d is None:
        d = D
    
    # Compute the covariance matrix
    covariance_matrix = np.matmul(X, X.T) / N
    D = np.shape(covariance_matrix)[0] - 1

    # Compute the eigenvalues and eigenvectors in the hyperbolic space
    eigenvalues, eigen_signs, eigenvectors = compute_j_eigenvalues_accurate(covariance_matrix, d)
    
    # Filter and sort the eigenvalues and corresponding eigenvectors
    positive_eigenvalue_indices = eigen_signs > 0
    positive_eigenvalues = eigenvalues[positive_eigenvalue_indices]
    sorted_indices = np.argsort(positive_eigenvalues)[::-1]
    
    # Extract the base point (eigenvectors with negative eigenvalue)
    base_point = np.squeeze(eigenvectors[:, ~positive_eigenvalue_indices])

    # Extract and sort the hyperbolic subspace basis vectors
    hyperbolic_subspace_basis = eigenvectors[:, positive_eigenvalue_indices]
    hyperbolic_subspace_basis = hyperbolic_subspace_basis[:, sorted_indices]

    # Handle multiple potential base points
    if len(np.shape(base_point)) > 1:
        base_point_index = find_first_max_index(eigenvalues[~positive_eigenvalue_indices])
        base_point = base_point[:, base_point_index]
    
    # Ensure the first component of the base point is positive
    if base_point[0] < 0:
        base_point = -base_point
    
    # Reshape base_point and concatenate with the hyperbolic subspace basis
    base_point_reshaped = base_point.reshape(D+1, 1)
    H_matrix = np.concatenate((base_point_reshaped, hyperbolic_subspace_basis), axis=1)
    
    # Create the subspace object and set its attributes
    S = subspace()
    S.H = H_matrix
    S.p = base_point
    S.Hp = hyperbolic_subspace_basis
    S.eigenvalues = eigenvalues
    
    return S
###########################################################################
###########################################################################
###########################################################################
def gram_schmidt(components, n_components):
    def inner(u, v):
        return torch.sum(u * v)
    Q = []
    for k in range(n_components):
        v_k = components[k]
        proj = 0.0
        for v_j in Q:
            v_j = v_j[0]
            coeff = inner(v_j, v_k) / inner(v_j, v_j).clamp_min(1e-15)
            proj += coeff * v_j
        v_k = v_k - proj
        v_k = v_k / torch.norm(v_k).clamp_min(1e-15)
        Q.append(torch.unsqueeze(v_k, 0))
    return torch.cat(Q, dim=0)
###########################################################################
###########################################################################
###########################################################################
def compute_distance_matrix(X, Q_or_S, mu_ref=None, dimension=None, method=None):
    """
    Compute the distance matrix using the specified method.

    Parameters:
    X (ndarray): Input data matrix (D+1, N).
    Q_or_S (ndarray or object): Projection matrix Q or an object containing the transformation matrix 'H'.
    mu_ref (ndarray, optional): Reference point for reflections (required for certain methods).
    dimension (int): The target dimension for the subspace.
    method (str): The method to use ('bsa', 'horopca', 'pga', or other).

    Returns:
    tuple: The hyperbolic distance matrix (HDM) and a boolean indicating if the result is inaccurate.
    """
    D, N = np.shape(X)
    D -= 1  # Adjust D to match the context of the hyperbolic space

    # Construct the J matrix for the hyperbolic space
    J = np.eye(D + 1)
    J[0, 0] = -1

    inaccurate = False

    if method in ['bsa', 'horopca']:
        Q = Q_or_S
        # Transpose X for easier indexing
        X = X.T

        # Extract Y from X and normalize
        Y = X[:, 1:].copy()
        for n in range(N):
            y = Y[n, :]
            Y[n, :] = y / (1 + X[n, 0].copy())

        # Convert Y to tensor and reflect at zero in Poincare ball
        z = torch.from_numpy(Y)  # N by D in Poincare
        x = poincare.reflect_at_zero(z, mu_ref)

        # Gram-Schmidt process to orthonormalize Qd
        Qd = gram_schmidt(Q[:dimension, :], dimension)

        if method == 'bsa':
            proj = poincare.orthogonal_projection(x, Qd, normalized=True)
            Q_orthogonal = euclidean.orthonormal(Qd)
            x_p = proj @ Q_orthogonal.transpose(0, 1)
        else:
            if dimension == 1:
                proj = project_kd(Qd, x)[0]
            else:
                hyperboloid_ideals = hyperboloid.from_poincare(Qd, ideal=True)
                hyperboloid_x = hyperboloid.from_poincare(x)
                hyperboloid_proj = hyperboloid.horo_projection(hyperboloid_ideals, hyperboloid_x)[0]
                proj = hyperboloid.to_poincare(hyperboloid_proj)

            # Orthonormalize Qd in Euclidean space
            Q_orthogonal = euclidean.orthonormal(Qd)

            # Project x to orthogonal Q
            x_p = proj @ Q_orthogonal.transpose(0, 1)

        # Compute pairwise distances in Poincare ball
        distance_matrix = poincare.pairwise_distance(x_p)
        distance_matrix = distance_matrix.numpy()
    else:
        S = Q_or_S
        if method == 'pga':
            # Compute the hyperbolic logarithm map of the data
            tangent_vectors = hyperbolic_log(X, S)

            # Extract the first `dimension` columns of the hyperbolic subspace basis
            H_matrix = S.Hp[:, :dimension]

            # Project the tangent vectors onto the subspace
            projection_matrix = np.matmul(H_matrix, H_matrix.T)
            projected_tangent_vectors = np.matmul(projection_matrix, tangent_vectors)

            # Compute the hyperbolic exponential map of the projected tangent vectors
            projected_data = compute_hyperbolic_exponential(projected_tangent_vectors, S)
            Gd = np.matmul(np.matmul(projected_data.T, J), projected_data)
        else:
            # Extract the first d+1 columns of the transformation matrix H
            H = S.H[:, :dimension + 1]

            # Construct the projection matrix
            Jk = np.eye(dimension + 1)
            Jk[0, 0] = -1

            # Project the data into the new subspace
            projection_matrix = np.matmul(np.matmul(H, Jk), H.T)
            X_projected = np.matmul(np.matmul(projection_matrix, J), X)

            # Normalize the projected data
            for n in range(N):
                x = X_projected[:, n]
                norm_value = -J_norm(x)

                if norm_value < 0:
                    inaccurate = True
                else:
                    X_projected[:, n] = x / np.sqrt(norm_value)

            # Compute the Gram matrix in the new subspace
            Gd = np.matmul(np.matmul(X_projected.T, J), X_projected)
        
        # Compute the distance matrix in the projected space
        Gd[Gd >= -1] = -1  # Ensure the values are within the valid range
        distance_matrix = np.arccosh(-Gd)
        np.fill_diagonal(distance_matrix, 0)

    return distance_matrix, inaccurate
###########################################################################
###########################################################################
###########################################################################
def load_subspace_instance(directory, file_index):
    try:
        filename = f'subspace_{file_index}.pkl'
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            instance = pickle.load(file)
        return instance
    except Exception as e:
        tqdm.write(f"Error occurred while loading: {e}")
        logging.info(f"Error occurred while loading: {e}")
        return None
###########################################################################
###########################################################################
###########################################################################
def load_subspace_instance_qmu(subspace_directory, file_index):
    try:
        Q_path = os.path.join(subspace_directory, f'Q_{file_index}.pt')
        mu_ref_path = os.path.join(subspace_directory, f'mu_{file_index}.pt')
        
        Q = torch.load(Q_path)
        mu_ref = torch.load(mu_ref_path)
        
        return Q, mu_ref
    except Exception as e:
        tqdm.write(f"Error occurred while loading: {e}")
        logging.info(f"Error occurred while loading: {e}")
        return None, None
###########################################################################
###########################################################################
###########################################################################
def compute_hyperbolic_distance_matrix(X):
    """
    Compute the hyperbolic distance matrix for the given data.

    Parameters:
    X (ndarray): Input data matrix (DxN).

    Returns:
    ndarray: The hyperbolic distance matrix (distance_matrix).
    """
    
    D, N = np.shape(X)
    D -= 1  # Adjust D to match the context of the hyperbolic space
    
    # Construct the J matrix for the hyperbolic space
    J = np.eye(D + 1)
    J[0, 0] = -1
    
    # Compute the Gram matrix in the hyperbolic space
    G = np.matmul(np.matmul(X.T, J), X)
    
    # Clamp values to ensure they are suitable for arccosh
    G[G >= -1] = -1
    
    # Compute the distance matrix using arccosh
    distance_matrix = np.arccosh(-G)
    
    # Set the diagonal to zero (distance from a point to itself)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix
###########################################################################
###########################################################################
###########################################################################
def normalize_v(v, threshold):
    """
    Normalize a vector and check if further normalization is needed based on a threshold.

    Parameters:
    v (ndarray): The input vector to be normalized.
    threshold (float): The threshold value to determine if further normalization is necessary.

    Returns:
    tuple: A tuple containing the normalized vector and a boolean indicating if further normalization was performed.
    """
    normalized = False
    
    # Compute the Euclidean norm of the vector
    norm_vector = np.linalg.norm(v)
    
    # Check if the norm exceeds the threshold
    if norm_vector > threshold:
        # Normalize the vector
        v = v / norm_vector
        
        # Compute the hyperbolic norm of the normalized vector
        D = len(v) - 1
        norm_hyperbolic = np.sqrt(np.abs(J_norm(v)))
        
        # Check if the hyperbolic norm exceeds the threshold
        if norm_hyperbolic > threshold:
            # Further normalize the vector
            v = v / norm_hyperbolic
            normalized = True
    
    return v, normalized
###########################################################################
###########################################################################
###########################################################################
def j_orthonormalize(v, subspace_basis, eigenvalue_signs):
    """
    Perform J-orthonormalization of a vector with respect to a subspace.

    Parameters:
    v (ndarray): The vector to orthonormalize.
    subspace_basis (ndarray): Basis vectors of the subspace.
    eigenvalue_signs (list): Signs of eigenvalues used in projection.

    Returns:
    ndarray: The J-orthonormalized vector.
    """
    D = len(v) - 1
    J = np.eye(D + 1)
    J[0, 0] = -1
    
    # Reshape the vector for matrix operations
    vector = np.reshape(v, (D + 1, 1))
    
    # Check if eigenvalue signs are provided
    if len(eigenvalue_signs) > 0:
        # Construct projection matrix to ensure J-orthonormality
        diag_matrix = np.diag(eigenvalue_signs)
        projection_matrix = np.eye(D + 1) - np.matmul(np.matmul(subspace_basis, diag_matrix), np.matmul(subspace_basis.T, J))
        vector = np.matmul(projection_matrix, vector)

    return vector
###########################################################################
###########################################################################
###########################################################################
def expected_sgn(count):
    if count ==1:
        sgn = -1
    else:
        sgn = 1
    return sgn
###########################################################################
###########################################################################
###########################################################################
def compute_residual_matrix(Cx, eigenvalues, eigenvectors):
    """
    Compute the residual matrix by removing the contribution of eigenvalues and eigenvectors.

    Parameters:
    Cx (ndarray): The original matrix (DxD).
    eigenvalues (ndarray): Array of eigenvalues.
    eigenvectors (ndarray): Matrix of eigenvectors (DxD).

    Returns:
    ndarray: The residual matrix.
    """
    residual_matrix = Cx.copy()  # Initialize the residual matrix as a copy of the original matrix
    
    # Subtract the contribution of each eigenvalue and corresponding eigenvector
    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        eigenvector_i = eigenvectors[:, i]
        residual_matrix -= lambda_i * np.outer(eigenvector_i, eigenvector_i)
    
    # Normalize the residual matrix by its maximum absolute value
    residual_matrix /= np.max(np.abs(residual_matrix))
    
    return residual_matrix
###########################################################################
###########################################################################
########################################################################### 
def find_good_initial_j_eigenvector(Cx, true_sign, eigenvalues, eigenvectors, eigen_signs, threshold):
    """
    Find a good initial J-eigenvector for the given residual matrix.

    Parameters:
    Cx (ndarray): covariance matrix (DxD).
    true_sign (float): The true sign of the eigenvector.
    eigenvectors (ndarray): Matrix of previously computed eigenvectors (DxD).
    eigen_signs (ndarray): Array of eigenvalue signs.
    threshold (float): Threshold for normalization.

    Returns:
    ndarray: The initial J-eigenvector.
    """

    Cx_residual = compute_residual_matrix(Cx, eigenvalues, eigenvectors)

    D = np.shape(Cx_residual)[0] - 1

    # Construct J matrix
    J = np.eye(D + 1)
    J[0, 0] = -1

    # Compute the leading eigenvector of the J-transformed residual matrix
    _, v = eigs(np.matmul(Cx_residual, J), k=1, which='LM')
    v = np.real(v)

    # Orthonormalize and normalize the leading eigenvector
    v = j_orthonormalize(v, eigenvectors, eigen_signs)
    v, status = normalize_v(v, threshold)

    # Check the sign of the normalized eigenvector
    sign = J_norm(v)

    # Add noise and re-normalize if the vector is not suitable
    while (not status) or (sign * true_sign < 0):
        noise = threshold * np.random.randn(D + 1, 1)
        v = j_orthonormalize(v + noise, eigenvectors, eigen_signs)
        v, status = normalize_v(v, threshold)
        sign = J_norm(v)

    return v
###########################################################################
###########################################################################
###########################################################################
def find_random_j_eigenvector(D, true_sign, eigenvectors, eigen_signs, threshold):
    """
    Find a valid random J-eigenvector

    Parameters:
    D (int): Dimension of the vectors.
    true_sign (float): The true sign of the eigenvector.
    eigenvectors (ndarray): Matrix of previously computed eigenvectors (DxD).
    eigen_signs (ndarray): Array of eigenvalue signs.
    threshold (float): Threshold for normalization.

    Returns:
    ndarray: A suitable random J-eigenvector.
    """

    status = False
    sign = -1
    while not status or sign * true_sign < 0:
        # Generate a random vector
        random_vector = np.random.randn(D + 1, 1)
        
        # Orthonormalize the random vector
        random_vector = j_orthonormalize(random_vector, eigenvectors, eigen_signs)
        
        # Normalize the orthonormalized vector
        random_vector, status = normalize_v(random_vector, threshold)
        
        # Calculate the sign of the normalized vector
        sign = J_norm(random_vector)

    return random_vector
###########################################################################
###########################################################################
########################################################################### 
def perturb_to_find_valid_j_eigenvector(v, true_sign, D, eigenvectors, eigenvalue_signs, threshold):
    """
    Perturbs the initial vector to find a valid J eigenvector.
    
    Parameters:
        v (np.ndarray): The initial vector to start perturbing from.
        true_sign (float): The true sign for the J norm.
        D (int): The dimension of the problem.
        eigenvectors (np.ndarray): The current set of eigenvectors.
        eigenvalue_signs (np.ndarray): The signs of the current eigenvalues.
        threshold (float): The perturbation threshold.

    Returns:
        np.ndarray: A valid J eigenvector.
    """
    is_valid = False
    current_sign = J_norm(v)
    
    while not is_valid or (current_sign * true_sign < 0):
        # Add small random noise to the initial vector to perturb it
        noise = threshold * np.random.randn(D + 1, 1)
        perturbed_vector = v + noise

        # Orthonormalize the perturbed vector against the current eigenvectors
        perturbed_vector = j_orthonormalize(perturbed_vector, eigenvectors, eigenvalue_signs)
        
        # Normalize the perturbed vector
        perturbed_vector, is_valid = normalize_v(perturbed_vector, threshold)
        
        # Compute the sign of the J norm of the perturbed vector
        current_sign = J_norm(perturbed_vector)
    
    return perturbed_vector
###########################################################################
###########################################################################
###########################################################################
def update_j_eigenvectors(eigenvectors, new_eigenvector):
    """
    Updates the set of eigenvectors with a new eigenvector.

    Parameters:
        eigenvectors (np.ndarray): Array of existing eigenvectors.
        new_eigenvector (np.ndarray): New eigenvector to be added.

    Returns:
        np.ndarray: Updated array of eigenvectors.
    """
    if np.shape(eigenvectors)[0] == 0:
        # If no eigenvectors exist yet, initialize with the new eigenvector
        updated_eigenvectors = new_eigenvector
    else:
        # Concatenate the new eigenvector to the existing set of eigenvectors
        updated_eigenvectors = np.concatenate((eigenvectors, new_eigenvector), axis=1)
    
    return updated_eigenvectors
###########################################################################
###########################################################################
###########################################################################
def perform_multiplications(Cxj, v, threshold, accuracy_factor, eigenvectors, eigenvalue_signs):
    """
    Performs one iteration of matrix-vector multiplications with normalization and orthonormalization steps.
    
    Parameters:
        Cxj (np.ndarray): The matrix used for multiplication.
        v (np.ndarray): The input vector to be multiplied.
        threshold (float): The threshold for normalization.
        accuracy_factor (int): Number of times orthonormalization is applied.
        eigenvectors (np.ndarray): The set of eigenvectors for orthonormalization.
        eigenvalue_signs (np.ndarray): Signs of the eigenvalues for orthonormalization.

    Returns:
        np.ndarray: The resulting vector after the iteration.
    """

    # Step 1: Matrix-vector multiplication
    v_out = np.matmul(Cxj, v)
    
    # Step 2: Normalize the resulting vector
    v_out, _ = normalize_v(v_out, threshold)
    
    # Step 3: Apply orthonormalization and normalization iteratively
    for _ in range(accuracy_factor):
        v_new = j_orthonormalize(v_out, eigenvectors, eigenvalue_signs)
        v_new, _ = normalize_v(v_out, threshold)

    return v_out
###########################################################################
###########################################################################
###########################################################################
def compute_one_j_eigenvalue(Cx, J, j_eigenvector):
    """
    Computes the j-th eigenvalue for given matrices and eigenvector.
    
    Parameters:
        Cx (np.ndarray): The covariance matrix.
        J (np.ndarray): The matrix J.
        j_eigenvector (np.ndarray): The j-eigenvector.

    Returns:
        float: The computed j_eigenvalue.
    """
    
    # Step 1: Perform the first matrix multiplication: (C * J) * j_eigenvector
    intermediate_vector = np.matmul(np.matmul(Cx, J), j_eigenvector)
    
    # Step 2: Perform the second matrix multiplication: (C * J) * intermediate_vector
    result_vector = np.matmul(np.matmul(Cx, J), intermediate_vector)
    
    # Step 3: Calculate the eigenvalue as the square root of the ratio of norms
    j_eigenvalue = np.sqrt(np.linalg.norm(result_vector) / np.linalg.norm(j_eigenvector))
    
    return j_eigenvalue
###########################################################################
###########################################################################
###########################################################################
def compute_j_eigenvalues(Cx,d):
    D = np.shape(Cx)[0]-1
    J = np.eye(D+1)
    J[0,0] = -1
    evals = []
    eval_signs = []
    condition = True
    count = 0
    evecs = []
    while condition:
        count = count + 1 

        _, v = eigs(np.matmul(Cx,J), k=1, which = 'LM')
        v = v/np.sqrt(np.abs(J_norm(v))) 
        v = np.real(v)
        sgn = np.sign(J_norm(v))
        if count == 1:
            evecs = v
        else:
            evecs = np.concatenate( (evecs,v) , axis = 1)
        lmbd = np.matmul(np.matmul(v.T,J), np.matmul(np.matmul(Cx,J),v))

        lmbd = np.squeeze(lmbd)
        evals = np.append(evals,lmbd)
        eval_signs = np.append(eval_signs,sgn)
        Cx = Cx - lmbd*np.matmul(v,v.T)
        condition = are_eigenvalues_valid(eval_signs,d)
    return evals, eval_signs, evecs
###########################################################################
###########################################################################
###########################################################################
def compute_j_eigenvalues_accurate(Cx,d):
    condition = True
    count = 0
    threshold = 10**(-20)
    ev_threshold = 10**(-30)

    evals = []
    eval_signs = []
    evecs = []
    line = []

    D = np.shape(Cx)[0]-1
    J = np.eye(D+1)
    J[0,0] = -1

    Cxj = np.matmul(Cx,J)
    accuracy_factor = 1
    while condition:
        count = count + 1 
        true_sgn = expected_sgn(count)
        ###################################################################################  
        v = find_good_initial_j_eigenvector(Cx,true_sgn,evals,evecs,eval_signs,threshold)
        flag = True
        errors = [1]
        while flag:
            norm_v = np.linalg.norm(v)
            v2 = perform_multiplications(Cxj, v, threshold, accuracy_factor, evecs, eval_signs)
            norm_v2 = np.linalg.norm(v2)
            sgn = J_norm(v2)
            #####################################################
            if norm_v2 < ev_threshold:
                line = 1
                ev = 1
                errors = [1]
                flag = False
                v2 = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
            else:
                ev = np.linalg.norm(v/norm_v - v2/norm_v2)
                ev = min(ev ,np.linalg.norm(v/norm_v + v2/norm_v2))
            #####################################################
            if (ev < ev_threshold) and (sgn*true_sgn >=0):
                flag = False
                errors = [1]
                line = 2
            else:
                ev_inf = min(errors[-1], ev)
                errors.append(ev_inf)
                K = 1000
                if len(errors) > K+1 and np.mod(len(errors),K) == 0: 
                    last_K = errors[-K:]
                    mean_err = np.mean(last_K)
                    std_err = np.std(last_K)

                    sgn = J_norm(v2)
                    good_enough = (mean_err == 0) or (std_err < ev_threshold/count) 
                    if  good_enough and (sgn*true_sgn >=0):
                        flag = False
                        errors = [1]
                        line = 3
                    elif len(evals) >= 1:
                        if sgn*true_sgn < 0:
                            accuracy_factor = accuracy_factor + 1
                            if accuracy_factor > 100:
                                flag = False
                                line = 4
                                errors = [1]
                                v2 = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
                        else:
                            min_val = np.min(evals)
                            no_of_remaining_vals = D+1-len(eval_signs)
                            remaining_energy = min_val/2 * no_of_remaining_vals
                            total_energy = np.sum(evals)+min_val/2 * no_of_remaining_vals

                            accurate_enough = remaining_energy/total_energy < 10**(-4)
                            iterate_upperb = len(errors) > (D+1)*K*np.log(1+len(evals))
                            if accurate_enough or (len(evals) >= 2 and iterate_upperb):
                                flag = False
                                #status = False
                                line = 5
                                errors = [1]
                                v2 = perturb_to_find_valid_j_eigenvector(v2, true_sgn, D, evecs, eval_signs, threshold)

            v = j_orthonormalize(v2,evecs,eval_signs)
            v,_ = normalize_v(v,threshold)
            norm_v = np.linalg.norm(v)
            if norm_v < ev_threshold:
                status = False
                errors = [1]
                flag = False
                line = 6
                v = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
                
        v = j_orthonormalize(v,evecs,eval_signs)
        v,status = normalize_v(v,threshold)
        sgn = np.sign(J_norm(v))
        if (not status) or (sgn*true_sgn < 0):
            status = False
            line = 7
            v = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
            sgn = J_norm(v)
                    
        lmbd = compute_one_j_eigenvalue(Cx, J, v)
        sgn = np.sign(J_norm(v))
        
        evecs = update_j_eigenvectors(evecs,v)
        evals = np.append(evals,lmbd)
        eval_signs = np.append(eval_signs,sgn)
        condition = are_eigenvalues_valid(eval_signs,d)
        #print(count,evals)
    return evals, eval_signs, evecs
###########################################################################
###########################################################################
###########################################################################
def are_eigenvalues_valid(eigenvalue_signs, d):
    """
    Checks if the set of eigenvalues meets the specified conditions.
    
    Parameters:
        eigenvalue_signs (np.ndarray): Array of signs of the current eigenvalues.
        target_positive_count (int): Desired number of positive eigenvalue signs.

    Returns:
        bool: True if the condition is met, False otherwise.
    """
    # Count the number of positive and negative eigenvalue signs
    positive_sign_count = np.sum(eigenvalue_signs > 0)
    negative_sign_count = np.sum(eigenvalue_signs < 0)
    
    # Determine if the condition is met
    condition = not (positive_sign_count == d and negative_sign_count >= 1)
    
    return condition
###########################################################################
###########################################################################
###########################################################################
def project_vector_to_hyperbolic_space(vector):
    """
    Project a given vector to hyperbolic space using iterative optimization.

    Parameters:
    vector (np.ndarray): The input vector to be projected.

    Returns:
    np.ndarray: The projected vector in hyperbolic space.
    """
    # Determine the dimension from the input vector
    #######################################
    dimension = len(vector)-1

    # Define the Lorentzian metric tensor
    lorentzian_metric = np.ones(dimension+1)
    lorentzian_metric[0] = -1
    
    # Check if the vector is above the Lorentzian sheet
    above_sheet = vector[0] > 0
    
    # Set the tolerance level for convergence
    tolerance = 10**(-16)
    center = 0
    
    # Initialize the optimal error and projection values
    optimal_error = 10**(10)
    optimal_projector = np.eye(dimension+1)
    projected_point = vector
    for i in range(70):
        lambda_range = 10**(-i)

        if above_sheet:
            lambda_min = max(center-lambda_range, -1+tolerance)
            lambda_max = min(center+lambda_range, 1-tolerance)
            number = 100
        else:
            lambda_min = max(center-lambda_range*10000, 1+tolerance)
            lambda_max = center+lambda_range*10000
            number = 100
        lambda_values = np.linspace(lambda_min,lambda_max,num=number)
        
        for lambda_val in lambda_values:
            projection_operator_lambda = 1./(1+lambda_val*lorentzian_metric) 
            projected_point_lambda = projection_operator_lambda  * vector # elementwise product
            if abs(J_norm(projected_point_lambda)+1) < optimal_error:
                optimal_projector = projection_operator_lambda
                optimal_error = abs(J_norm(projected_point_lambda)+1)
                center = lambda_val
        
        new_projected_point = optimal_projector * vector
        if abs(J_norm(new_projected_point)+1) < tolerance:
            return new_projected_point
        projected_point = new_projected_point

    if optimal_error > 10**(-5):
        print('Warning: High projection error:', optimal_error)
    return projected_point
###########################################################################
###########################################################################
########################################################################### 
def lgram_to_points(dimension, gram_matrix):
    """
    Convert a Lorentzian Gram matrix to points using eigen decomposition.

    Parameters:
    dimension (int): Dimension of the target space.
    gram_matrix (np.ndarray): Gram matrix to be decomposed.

    Returns:
    np.ndarray: Coordinates corresponding to the input Gram matrix.
    """
    min_dimension = min(dimension, np.shape(gram_matrix)[0]-1)
    # Perform eigen decomposition of the Gram matrix
    eigenvalues, eigenvectors = np.linalg.eig(gram_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Identify the smallest eigenvalue and its index
    min_eigenvalue = np.amin(eigenvalues)
    min_index = np.argmin(eigenvalues)
    min_eigenvector = eigenvectors[:, min_index]
    
    
    
    # Remove the smallest eigenvalue and sort the rest in descending order
    eigenvalues = np.delete(eigenvalues, min_index)
    eigenvectors = np.delete(eigenvectors, min_index, axis=1)
    
    sorted_indices = np.argsort(-eigenvalues)

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]
    
    top_eigenvalues = eigenvalues[:dimension]
    top_eigenvectors = eigenvectors[:,:dimension]
    
    
    # Create the final set of eigenvalues for constructing the coordinate matrix
    final_eigenvalues = np.concatenate((abs(min_eigenvalue), top_eigenvalues), axis=None)
    final_eigenvalues[final_eigenvalues <= 0] = 0
    final_eigenvalues = np.sqrt(final_eigenvalues)

    # Adjust eigenvectors to match the selected eigenvalues
    final_eigenvectors = np.column_stack((min_eigenvector, top_eigenvectors))
    
    # Construct the coordinate matrix
    X = np.matmul(np.diag(final_eigenvalues), final_eigenvectors.T)
    
    # Ensure the first coordinate is positive
    if X[0, 0] < 0:
        X = -X

    if min_dimension < dimension:
        zero_rows = np.zeros((dimension-min_dimension, np.shape(gram_matrix)[0]))
        X = np.vstack((X, zero_rows))
    
    return X
###########################################################################
###########################################################################
###########################################################################
def estimate_hyperbolic_subspace_pga(X, d = None):
    """
    Estimate the hyperbolic subspace using Principal Geodesic Analysis (PGA).

    Parameters:
    X (ndarray): The input data matrix (DxN).
    parameters (object): An object containing parameters such as dimension (d), dimensionality (D), and number of samples (N).

    Returns:
    tuple: A tuple containing the transformed data matrix and the subspace object (X_, S).
    """
    S = subspace()  # Initialize the subspace object
    D, N = np.shape(X)
    D -= 1
    if d is None:
        d = D
    
    # Unpack parameters
    tau = 0.1
    tolerance = 10**(-10)
    
    # Compute initial mean vector p
    p = np.mean(X, axis=1)
    
    # Normalize p
    p /= np.sqrt(-J_norm(p))
    S.p = p
    
    # Initialize error and condition for convergence
    error = 1
    condition = True
    cnt = 0
    # Iteratively update p until convergence
    while condition:
        # Compute hyperbolic logarithm of the data with respect to current subspace
        cnt = cnt +1 
        tangent_vectors = hyperbolic_log(X, S)
        
        # Compute update delta_p
        delta_p = tau * np.mean(tangent_vectors, axis=1)
        delta_p = delta_p.reshape(D + 1, 1)
        
        # Perform hyperbolic exponential map to update p
        p = compute_hyperbolic_exponential(delta_p, S).ravel()
        
        # Calculate error between current p and previous p
        new_error = np.linalg.norm(p - S.p) / np.sqrt(D + 1)
        
        # Check convergence condition
        condition = np.abs(error - new_error) > tolerance
        
        # Update error and subspace.p
        error = new_error
        S.p = p
    
    # Finalize subspace estimation
    tangent_vectors = hyperbolic_log(X, S)
    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(tangent_vectors, tangent_vectors.T))
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort eigenvalues and eigenvectors
    index = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    
    # Extract top d eigenvectors as Hp
    Hp = eigenvectors[:, 0:d]
    S.Hp = Hp
    
    # Construct H matrix
    p = S.p
    H = np.concatenate((p.reshape(D + 1, 1), Hp), axis=1)
    S.H = H
    
    # Project data onto the estimated subspace and compute exponential map
    #Vt = np.matmul(np.matmul(Hp, Hp.T), V)
    #transformed_data = compute_hyperbolic_exponential(Vt, S)
    
    return S
###########################################################################
###########################################################################
###########################################################################
def run_dimensionality_reduction(model_type, X):
    lr=5e-2
    n_runs=5

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    torch.set_default_dtype(torch.float64)

    pca_models = {
        'pca': {'class': EucPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'tpca': {'class': TangentPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'pga': {'class': PGA, 'optim': True, 'iterative': True, "n_runs": n_runs},
        'bsa': {'class': BSA, 'optim': True, 'iterative': False, "n_runs": n_runs},
        'horopca': {'class': HoroPCA, 'optim': True, 'iterative': False, "n_runs": n_runs},
    }

    
    D, N = np.shape(X)
    D -= 1

    X = X.T
    Y = X[:, 1:].copy()
    for n in range(N):
        y = Y[n, :]
        Y[n, :] = y / (1 + X[n, 0].copy())

    z = torch.from_numpy(Y)  # N by D in poincare
    # Compute the mean and center the data
    #logging.info("Computing the Frechet mean to center the embeddings")
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(z, return_converged=True)
    #logging.info(f"Mean computation has converged: {has_converged}")
    x = poincare.reflect_at_zero(z, mu_ref)

    # Run dimensionality reduction methods
    #logging.info(f"Running {model_type} for dimensionality reduction")
    metrics = []
    dist_orig = poincare.pairwise_distance(x)
    k = 0
    if model_type in pca_models.keys():
        model_params = pca_models[model_type]
        model_params["n_runs"] = 1
        for _ in range(model_params["n_runs"]):
            model = model_params['class'](dim=D, n_components=D-1, lr=lr, max_steps=500)
            if torch.cuda.is_available():
                model.cuda()
            model.fit(x, iterative=model_params['iterative'], optim=model_params['optim'])
            while np.isnan(model.compute_metrics(x)['distortion']):
                k += 1
                model = model_params['class'](dim=D, n_components=D-k, lr=lr, max_steps=500)
                if torch.cuda.is_available():
                    model.cuda()
                model.fit(x, iterative=model_params['iterative'], optim=model_params['optim'])
                if k == D-1:
                    break
            if not np.isnan(model.compute_metrics(x)['distortion']):
                embeddings = model.map_to_ball(x).detach().cpu().numpy()
                Q = model.get_components()
            else:
                Q = np.nan
        return Q.detach().cpu(), mu_ref.detach().cpu()
    else:
        logging.info(f"Model {model_type} is not implemented.")
        return np.nan, np.nan


