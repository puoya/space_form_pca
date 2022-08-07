import numpy as np
import time 
import synthetic_data as sd
import pandas as pd
import os

#range of values to useof the parameters
stdRange = [0.01, 0.05, 0.1] 
D_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
N_range = ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 
            4000, 5000, 6000, 7000, 8000, 9000, 10000])

##############################################################
def readData(path):
    if(path.endswith('.npy')):
        return np.load(path, allow_pickle=True)
    else:
        return pd.read_pickle(path)


def D_Exp(exp, start, end): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/D_exp_/'

    for sigma in stdRange:
        for D in D_range:
            #folder name for current param
            currFolder = 'D_'+ str(D) + '_sigma_' + str(sigma) + '_data/'
            param = -1 #initialized to -1 as placeholder

            for iter in range(start, end+1):
                #adding param.npy on first iteration 
                if(iter == start):
                    fullPath = path + currFolder + 'param.pkl'
                    param = readData(fullPath)

                fullPath = path + currFolder + str(iter+1) + '/X.npy'
                X = readData(fullPath)

                fullPath = path + currFolder + str(iter+1) + '/S.pkl'
                S = readData(fullPath)

                if(exp == 1):
                    fullPath = path + currFolder + str(iter+1) + '/spca/'
                    noise_lvl_output, dist, S_, runtime = sd.sfpaFromData(X, S, param)
                elif(exp == 2):
                    fullPath = path + currFolder + str(iter+1) + '/spga/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                else: 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)

                #creating folder if it doesn't exist
                try:
                    os.mkdir(fullPath)
                except:
                    pass #folder already exists 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)    

def d_Exp(exp, start, end): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/d_exp/'

    for sigma in stdRange:
        for d in d_range:
            #folder name for current param
            currFolder = 'd_'+ str(d) + '_sigma_' + str(sigma) + '_data/'
            param = -1 #initialized to -1 as placeholder

            for iter in range(start, end+1):
                #adding param.npy on first iteration 
                if(iter == start):
                    fullPath = path + currFolder + 'param.pkl'
                    param = readData(fullPath)

                fullPath = path + currFolder + str(iter+1) + '/X.npy'
                X = readData(fullPath)

                fullPath = path + currFolder + str(iter+1) + '/S.pkl'
                S = readData(fullPath)

                if(exp == 1):
                    fullPath = path + currFolder + str(iter+1) + '/spca/'
                    noise_lvl_output, dist, S_, runtime = sd.sfpaFromData(X, S, param)
                elif(exp == 2):
                    fullPath = path + currFolder + str(iter+1) + '/spga/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                else: 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)

                #creating folder if it doesn't exist
                try:
                    os.mkdir(fullPath)
                except:
                    pass #folder already exists 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)        

def N_Exp(exp, start, end): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/N_exp/'

    for sigma in stdRange:
        for N in N_range:
             #folder name for current param
            currFolder = 'N_'+ str(N) + '_sigma_' + str(sigma) + '_data/'
            param = -1 #initialized to -1 as placeholder

            for iter in range(start, end+1):
                #adding param.npy on first iteration 
                if(iter == start):
                    fullPath = path + currFolder + 'param.pkl'
                    param = readData(fullPath)

                fullPath = path + currFolder + str(iter+1) + '/X.npy'
                X = readData(fullPath)

                fullPath = path + currFolder + str(iter+1) + '/S.pkl'
                S = readData(fullPath)

                if(exp == 1):
                    fullPath = path + currFolder + str(iter+1) + '/spca/'
                    noise_lvl_output, dist, S_, runtime = sd.sfpaFromData(X, S, param)
                elif(exp == 2):
                    fullPath = path + currFolder + str(iter+1) + '/spga/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                else: 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)

                #creating folder if it doesn't exist
                try:
                    os.mkdir(fullPath)
                except:
                    pass #folder already exists 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)   

def std_Exp(exp, start, end): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/std_exp/'

    long_std_range = np.arange(0.05, 5.02, 0.05)

    for sigma in long_std_range:
        sigma = round(sigma, 2)

        #folder name for current param
        currFolder = 'sigma_' + str(sigma) + '_data/'
        param = -1 #initialized to -1 as placeholder

        for iter in range(start, end+1):
            #adding param.npy on first iteration 
            if(iter == start):
                fullPath = path + currFolder + 'param.pkl'
                param = readData(fullPath)

            fullPath = path + currFolder + str(iter+1) + '/X.npy'
            X = readData(fullPath)

            fullPath = path + currFolder + str(iter+1) + '/S.pkl'
            S = readData(fullPath)

            if(exp == 1):
                fullPath = path + currFolder + str(iter+1) + '/spca/'
                noise_lvl_output, dist, S_, runtime = sd.sfpaFromData(X, S, param)
            elif(exp == 2):
                fullPath = path + currFolder + str(iter+1) + '/spga/'
                noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
            else: 
                fullPath = path + currFolder + str(iter+1) + '/dai/'
                noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)

            #creating folder if it doesn't exist
            try:
                os.mkdir(fullPath)
            except:
                pass #folder already exists 

            np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
            np.save(fullPath + 'dist', dist)
            np.save(fullPath + 'S_', S_)   
            np.save(fullPath + 'runtime', runtime)         


start = time.time()
#first param is start iteration, second param is end iteration (inclusive), third is experiment 
D_Exp(0, 99,1)
N_Exp(0, 99,1)
d_Exp(0, 99,1)
std_Exp(0, 99, 1)

#####################################
D_Exp(0, 30,2)
D_Exp(31, 60,2)
D_Exp(61, 99,2)

N_Exp(0, 30,2)
N_Exp(31, 60,2)
N_Exp(61, 99,2)

d_Exp(0, 30,2)
d_Exp(31, 60,2)
d_Exp(61, 99,2)

std_Exp(0, 30,2)
std_Exp(31, 60,2)
std_Exp(61, 99,2)
####################################
D_Exp(0, 30,3)
D_Exp(31, 60,3)
D_Exp(61, 99,3)

N_Exp(0, 30,3)
N_Exp(31, 60,3)
N_Exp(61, 99,3)

d_Exp(0, 30,3)
d_Exp(31, 60,3)
d_Exp(61, 99,3)

std_Exp(0, 30,3)
std_Exp(31, 60,3)
std_Exp(61, 99,3)


end = time.time()
print( round((end - start)/60, 2) )