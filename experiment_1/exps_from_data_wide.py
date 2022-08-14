import numpy as np
import time 
import synthetic_data as sd
import pandas as pd
import os

#range of values to use of the parameters
stdRange = [0.01, 0.05, 0.1] 
D_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
N_range = ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 
            4000, 5000, 6000, 7000, 8000, 9000, 10000])

def folderMade(fullPath):
    try:
        os.mkdir(fullPath)
    except:
        return True #folder already exists 
    return False 
##############################################################
def readData(path):
    if(path.endswith('.npy')):
        return np.load(path, allow_pickle=True)
    else:
        return pd.read_pickle(path)


def D_Exp(exp): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/D_exp_/'

    for iter in range(100):
        for sigma in stdRange:
            for D in D_range:
                #folder name for current param
                currFolder = 'D_'+ str(D) + '_sigma_' + str(sigma) + '_data/'
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2)  

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)    

def d_Exp(exp): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/d_exp/'

    for iter in range(100):
        for sigma in stdRange:
            for d in d_range:
                #folder name for current param
                currFolder = 'd_'+ str(d) + '_sigma_' + str(sigma) + '_data/'
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2) 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)        

def N_Exp(exp): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/N_exp/'

    for iter in range(100):
        for sigma in stdRange:
            for N in N_range:
                #folder name for current param
                currFolder = 'N_'+ str(N) + '_sigma_' + str(sigma) + '_data/' 
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2)                  

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)   

def std_Exp(start, end, exp): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/std_exp/'

    long_std_range = np.arange(0.05, 5.02, 0.05)
    for iter in range(100):
        for sigma in long_std_range:
            sigma = round(sigma, 2)

            #folder name for current param
            currFolder = 'sigma_' + str(sigma) + '_data/'
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
            elif(exp == 3): 
                fullPath = path + currFolder + str(iter+1) + '/dai/'
                if(folderMade(fullPath)):
                         continue
                noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
            elif(exp == 4):
                fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
            elif(exp == 5):
                fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                if(folderMade(fullPath)):
                         continue
                noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
            elif(exp == 6):
                fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                if(folderMade(fullPath)):
                         continue
                noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
            elif(exp == 7):
                fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                if(folderMade(fullPath)):
                         continue
                noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2) 

            np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
            np.save(fullPath + 'dist', dist)
            np.save(fullPath + 'S_', S_)   
            np.save(fullPath + 'runtime', runtime)         

def _D_Exp(exp): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/D_exp_/'

    for iter in range(100):
        for D in D_range:
            for sigma in stdRange:
                #folder name for current param
                currFolder = 'D_'+ str(D) + '_sigma_' + str(sigma) + '_data/'
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2) 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)    

def _d_Exp(exp): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/d_exp/'

    for iter in range(100):
        for d in d_range:
            for sigma in stdRange:
                #folder name for current param
                currFolder = 'd_'+ str(d) + '_sigma_' + str(sigma) + '_data/'
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2) 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)        

def _N_Exp(exp): 
    #initial path to directory 
    path = '/mirarablab_data/sfpca/N_exp/'

    for iter in range(100):
        for N in N_range:
            for sigma in stdRange:
                #folder name for current param
                currFolder = 'N_'+ str(N) + '_sigma_' + str(sigma) + '_data/' 
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    if(folderMade(fullPath)):
                         continue
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2)                 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)   


start = time.time()
#first param is start iteration, second param is end iteration (inclusive), third is experiment 

############## dai ######################
# D_Exp(3)

# N_Exp(3)

# d_Exp(3)

# std_Exp(3)

# _D_Exp(3)

# _N_Exp(3)

# _d_Exp(3)

############## dai new ######################
# D_Exp(5)

# N_Exp(5)

# d_Exp(5)

# std_Exp(5)

# _D_Exp(5)

# _N_Exp(5)

# _d_Exp(5)

############## liu mode 1 ######################
# D_Exp(6)

# N_Exp(6)

# d_Exp(6)

# std_Exp(6)

# _D_Exp(6)

# _N_Exp(6)

# _d_Exp(6)

############## liu mode 2 ######################
# D_Exp(7)

# N_Exp(7)

# d_Exp(7)

# std_Exp(7)

# _D_Exp(7)

# _N_Exp(7)

# _d_Exp(7)


end = time.time()
print( round((end - start)/60, 2) )