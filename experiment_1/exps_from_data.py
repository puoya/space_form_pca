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

##############################################################
def readData(path):
    if(path.endswith('.npy')):
        return np.load(path, allow_pickle=True)
    else:
        return pd.read_pickle(path)


def D_Exp(start, end, exp): 
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2) 

                #creating folder if it doesn't exist
                try:
                    os.mkdir(fullPath)
                except:
                    pass #folder already exists 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)    

def d_Exp(start, end, exp): 
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2) 

                #creating folder if it doesn't exist
                try:
                    os.mkdir(fullPath)
                except:
                    pass #folder already exists 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)        

def N_Exp(start, end, exp): 
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
                elif(exp == 3): 
                    fullPath = path + currFolder + str(iter+1) + '/dai/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 4):
                    fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                    noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
                elif(exp == 5):
                    fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                    noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
                elif(exp == 6):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
                elif(exp == 7):
                    fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                    noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2)                 
                #creating folder if it doesn't exist
                try:
                    os.mkdir(fullPath)
                except:
                    pass #folder already exists 

                np.save(fullPath + 'noise_lvl_output', noise_lvl_output)
                np.save(fullPath + 'dist', dist)
                np.save(fullPath + 'S_', S_)   
                np.save(fullPath + 'runtime', runtime)   

def std_Exp(start, end, exp): 
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
            elif(exp == 3): 
                fullPath = path + currFolder + str(iter+1) + '/dai/'
                noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
            elif(exp == 4):
                fullPath = path + currFolder + str(iter+1) + '/spga_new/'
                noise_lvl_output, dist, S_, runtime = sd.spgaFromData(X, S, param)
            elif(exp == 5):
                fullPath = path + currFolder + str(iter+1) + '/dai_new/'
                noise_lvl_output, dist, S_, runtime = sd.daiFromData(X, S, param)
            elif(exp == 6):
                fullPath = path + currFolder + str(iter+1) + '/liu_mode1/'
                noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 1)  
            elif(exp == 7):
                fullPath = path + currFolder + str(iter+1) + '/liu_mode2/'
                noise_lvl_output, dist, S_, runtime = sd.liuFromData(X, S, param, 2) 

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
################# spca ###################
D_Exp(0, 99,1)
# N_Exp(0, 99,1)
# d_Exp(0, 99,1)
# std_Exp(0, 99, 1)

############### pga #####################
# D_Exp(0, 30,2)
# D_Exp(31, 60,2)
# D_Exp(61, 99,2)

# N_Exp(0, 30,2)
# N_Exp(31, 60,2)
# N_Exp(61, 99,2)

# d_Exp(0, 30,2)
# d_Exp(31, 60,2)
# d_Exp(61, 99,2)

# std_Exp(0, 30,2)
# std_Exp(31, 60,2)
# std_Exp(61, 99,2)
############## dai ######################
# D_Exp(0, 30,3)
# D_Exp(31, 60,3)
# D_Exp(61, 99,3)

# N_Exp(0, 30,3)
# N_Exp(31, 60,3)
# N_Exp(61, 99,3)

# d_Exp(0, 30,3)
# d_Exp(31, 60,3)
# d_Exp(61, 99,3)

# std_Exp(0, 30,3)
# std_Exp(31, 60,3)
# std_Exp(61, 99,3)

############## pga new ######################
# D_Exp(0, 30,4)
# D_Exp(31, 60,4)
# D_Exp(61, 99,4)

# N_Exp(0, 30,4)
# N_Exp(31, 60,4)
# N_Exp(61, 99,4)

# d_Exp(0, 30,4)
# d_Exp(31, 60,4)
# d_Exp(61, 99,4)

# std_Exp(0, 30,4)
# std_Exp(31, 60,4)
# std_Exp(61, 99,4)

############## dai new ######################
# D_Exp(0, 30,5)
# D_Exp(31, 60,5)
# D_Exp(61, 99,5)

# N_Exp(0, 30,5)
# N_Exp(31, 60,5)
# N_Exp(61, 99,5)

# d_Exp(0, 30,5)
# d_Exp(31, 60,5)
# d_Exp(61, 99,5)

# std_Exp(0, 30,5)
# std_Exp(31, 60,5)
# std_Exp(61, 99,5)

############## liu mode 1 ######################
# D_Exp(0, 30,6)
# D_Exp(31, 60,6)
# D_Exp(61, 99,6)

# N_Exp(0, 30,6)
# N_Exp(31, 60,6)
# N_Exp(61, 99,6)

# d_Exp(0, 30,6)
# d_Exp(31, 60,6)
# d_Exp(61, 99,6)

# std_Exp(0, 30,6)
# std_Exp(31, 60,6)
# std_Exp(61, 99,6)

############## liu mode 2 ######################
# D_Exp(0, 30,7)
# D_Exp(31, 60,7)
# D_Exp(61, 99,7)

# N_Exp(0, 30,7)
# N_Exp(31, 60,7)
# N_Exp(61, 99,7)

# d_Exp(0, 30,7)
# d_Exp(31, 60,7)
# d_Exp(61, 99,7)

# std_Exp(0, 30,7)
# std_Exp(31, 60,7)
# std_Exp(61, 99,7)


end = time.time()
print( round((end - start)/60, 2) )