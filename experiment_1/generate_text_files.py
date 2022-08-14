import numpy as np
import os
import pandas as pd
import time 

COLUMN_NAMES = ['d', 'D', 'N', 'sigma', 'noise_lvl_input', 'noise_lvl_output', 'dist', 'runtime']
EXP_NAMES = ['spca', 'spga', 'spga_new', 'dai', 'dai_new', 'liu_mode1', 'liu_mode2']

#range of values to use of the parameters
stdRange = [0.01, 0.05, 0.1] 
D_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
N_range = ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 
            4000, 5000, 6000, 7000, 8000, 9000, 10000])

def pathExists(path):
    return os.path.exists(path)

def readData(path):
    if(path.endswith('.npy')):
        return np.load(path, allow_pickle=True)
    else:
        return pd.read_pickle(path)

def createDataFile(path, expName, d, D, N, sigma):
    data = []
    data.append(COLUMN_NAMES)
    for iter in range(100):
        expPath = path + str(iter+1) + '/'+ expName + '/'
        #Experiment wasn't done
        if (pathExists(expPath) == False):
            continue
        if (pathExists(path + str(iter)+ '/noise_level_input.npy') == False):
            continue
        noise_lvl_input = str(readData(path + str(iter+1)+ '/noise_level_input.npy'))

        if (pathExists(expPath+'noise_lvl_output.npy') == False):
            continue
        noise_lvl_output = str(readData(expPath+'noise_lvl_output.npy'))

        if (pathExists(expPath+'dist.npy') == False):
            continue       
        dist = str(readData(expPath+'dist.npy'))

        if (pathExists(expPath+'runtime.npy') == False):
            continue
        runtime = str(readData(expPath+'runtime.npy'))

        currRow = [d, D, N, sigma, noise_lvl_input, noise_lvl_output, dist, runtime]
        data.append(currRow)
    
    if (len(data) > 1):
        widths = [max([len(item) for item in col]) for col in zip(*data)]
        fmt = ''.join(['{{:{}}}'.format(width+4) for width in widths])
        with open(path+expName+'_'+"data.txt", "a") as f:
            for row in data:
                print(fmt.format(*row), file=f)   


def D_Exp(): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/D_exp_/'
    N = 10**4
    d = 1
    for sigma in stdRange:
        for D in D_range:
            #folder name for current param
            currFolder = 'D_'+ str(D) + '_sigma_' + str(sigma) + '_data/'
            currPath = path+currFolder
            for exp in EXP_NAMES:
                createDataFile(currPath, exp, str(d), str(D), str(N), str(sigma))

def d_Exp(): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/d_exp/'
    N = 10**4
    D = 100
    for sigma in stdRange:
        for d in d_range:
            #folder name for current param
            currFolder = 'd_'+ str(d) + '_sigma_' + str(sigma) + '_data/'
            currPath = path + currFolder 
            for exp in EXP_NAMES:
                createDataFile(currPath, exp, str(d), str(D), str(N), str(sigma))

def N_Exp(): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/N_exp/'
    D = 100
    d = 1
    for sigma in stdRange:
        for N in N_range:
            #folder name for current param
            currFolder = 'N_'+ str(N) + '_sigma_' + str(sigma) + '_data/' 
            currPath = path + currFolder 
            for exp in EXP_NAMES:
                createDataFile(currPath, exp, str(d), str(D), str(N), str(sigma))

def std_Exp(): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/std_exp/'
    N = 10000
    D = 100
    d = 1
    long_std_range = np.arange(0.05, 5.02, 0.05)

    for sigma in long_std_range:
        sigma = round(sigma, 2)
        #folder name for current param
        currFolder = 'sigma_' + str(sigma) + '_data/'
        currPath = path + currFolder 
        for exp in EXP_NAMES:
            createDataFile(currPath, exp, str(d), str(D), str(N), str(sigma))

start = time.time()

D_Exp()
d_Exp()
N_Exp()
std_Exp()

end = time.time()
print( round((end - start)/60, 2) )