import numpy as np
import time 
import synthetic_data as sd
import os

# num trials for each experiment
numIterations = 100

#range of values to useof the parameters
stdRange = [0.01, 0.05, 0.1] 
D_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
N_range = ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 
            4000, 5000, 6000, 7000, 8000, 9000, 10000])

##############################################################
def D_Exp(): 
    #initial path to directory 
    path = '/calab_data/mirarab/home/mmkhanza/sphericalPCA/datasets/D_exp_/'
    N = 10**4
    d = 1
    for sigma in stdRange:
        for D in D_range:
            #folder name for current param
            currFolder = 'D_'+ str(D) + '_sigma_' + str(sigma) + '_data/'

            #creating folder if it doesn't exist
            try:
                os.mkdir(path+currFolder)
            except:
                pass

            for iter in range(numIterations):
                # X, S, noise_level, param
                res = sd.experimentDataset(N, D, d, sigma) 
                #adding param.npy on first iteration 
                if(iter == 0):
                    fullPath = path + currFolder + 'param'
                    np.save(fullPath, res[3])

                #creating folder for experiment number if it doesn't exist
                expFolder = fullPath = path + currFolder + str(iter+1) 
                try:
                    os.mkdir(expFolder)
                except:
                    pass

                fullPath = path + currFolder + str(iter+1) + '/' + 'X'
                np.save(fullPath, res[0])

                fullPath = path + currFolder + str(iter+1) + '/' + 'S'
                np.save(fullPath, res[1])

                fullPath = path + currFolder + str(iter+1) + '/' + 'noise_level_input'
                np.save(fullPath, res[2])


def d_Exp(): 
    #initial path to directory 
    path = '/calab_data/mirarab/home/mmkhanza/sphericalPCA/datasets/d_exp/'

    N = 10**4
    D = 100
    for sigma in stdRange:
        for d in d_range:
            #folder name for current param
            currFolder = 'd_'+ str(d) + '_sigma_' + str(sigma) + '_data/'

            #creating folder if it doesn't exist
            try:
                os.mkdir(path+currFolder)
            except:
                pass

            for iter in range(numIterations):
                # X, S, noise_level, param
                res = sd.experimentDataset(N, D, d, sigma) 
                #adding param.npy on first iteration 
                if(iter == 0):
                    fullPath = path + currFolder + 'param'
                    np.save(fullPath, res[3])

                #creating folder for experiment number if it doesn't exist
                expFolder = fullPath = path + currFolder + str(iter+1) 
                try:
                    os.mkdir(expFolder)
                except:
                    pass

                fullPath = path + currFolder + str(iter+1) + '/' + 'X'
                np.save(fullPath, res[0])

                fullPath = path + currFolder + str(iter+1) + '/' + 'S'
                np.save(fullPath, res[1])

                fullPath = path + currFolder + str(iter+1) + '/' + 'noise_level_input'
                np.save(fullPath, res[2])

def N_Exp(): 
    #initial path to directory 
    path = '/calab_data/mirarab/home/mmkhanza/sphericalPCA/datasets/N_exp/'

    D = 100
    d = 1

    for sigma in stdRange:
        for N in N_range:
             #folder name for current param
            currFolder = 'N_'+ str(N) + '_sigma_' + str(sigma) + '_data/'

            #creating folder if it doesn't exist
            try:
                os.mkdir(path+currFolder)
            except:
                pass         

            for iter in range(numIterations):
                 # X, S, noise_level, param
                res = sd.experimentDataset(N, D, d, sigma) 
                #adding param.npy on first iteration 
                if(iter == 0):
                    fullPath = path + currFolder + 'param'
                    np.save(fullPath, res[3])

                #creating folder for experiment number if it doesn't exist
                expFolder = fullPath = path + currFolder + str(iter+1) 
                try:
                    os.mkdir(expFolder)
                except:
                    pass

                fullPath = path + currFolder + str(iter+1) + '/' + 'X'
                np.save(fullPath, res[0])

                fullPath = path + currFolder + str(iter+1) + '/' + 'S'
                np.save(fullPath, res[1])

                fullPath = path + currFolder + str(iter+1) + '/' + 'noise_level_input'
                np.save(fullPath, res[2])               


def std_exp():
    #initial path to directory 
    path = '/calab_data/mirarab/home/mmkhanza/sphericalPCA/datasets/std_exp/'

    long_std_range = np.arange(0.05, 5.02, 0.05)

    N = 10000
    D = 100
    d = 1
    for sigma in long_std_range:
        sigma = round(sigma, 2)

        #folder name for current param
        currFolder = 'sigma_' + str(sigma) + '_data/'

        #creating folder if it doesn't exist
        try:
            os.mkdir(path+currFolder)
        except:
            pass   
        
        for iter in range(numIterations):
            # X, S, noise_level, param
            res = sd.experimentDataset(N, D, d, sigma) 
            #adding param.npy on first iteration 
            if(iter == 0):
                fullPath = path + currFolder + 'param'
                np.save(fullPath, res[3])

            #creating folder for experiment number if it doesn't exist
            expFolder = fullPath = path + currFolder + str(iter+1) 
            try:
                os.mkdir(expFolder)
            except:
                pass

            fullPath = path + currFolder + str(iter+1) + '/' + 'X'
            np.save(fullPath, res[0])

            fullPath = path + currFolder + str(iter+1) + '/' + 'S'
            np.save(fullPath, res[1])

            fullPath = path + currFolder + str(iter+1) + '/' + 'noise_level_input'
            np.save(fullPath, res[2]) 
    


"""
Below runs each of the experiments and times it. You may want 
to run one at a time due to the amount of time it takes for 
each experiment to run. 
"""

start = time.time()
D_Exp()
# d_Exp()
# N_Exp()
# std_exp()
end = time.time()
print( round((end - start)/60, 2) )