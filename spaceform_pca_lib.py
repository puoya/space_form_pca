import numpy as np
import scipy.linalg
class subspace:
    def __init__(self, H = 0, Hp = 0, p = 0):
        self.H = H
        self.Hp = Hp
        self.p = p
###########################################################################
def random_orthogonal_matrix(param):
    D = param.D
    d = param.d
    #####################################
    H = np.zeros((D+1,d+1))
    #####################################
    for i in range(d+1):
        PH_perp = np.eye(D+1)-np.matmul(H,H.T)
        v = np.random.normal(0, 1, (D+1,1))
        v = np.matmul(PH_perp,v)
        v = v/np.linalg.norm(v)
        H[:,i] = v[:,0]
    return H
###########################################################################
def random_spherical_subspace(param):
    H = random_orthogonal_matrix(param)
    #####################################
    p = H[:,0]
    Hp = np.delete(H,0,1)
    #####################################
    S = subspace()
    S.H = H
    S.Hp = Hp
    S.p = p
    #####################################
    return S
###########################################################################
def random_spherical_tangents(S,param):
    N = param.N
    sigma = param.sigma
    #####################################
    Hp = S.Hp
    p = S.p
    #####################################
    D = np.shape(Hp)[0]-1
    d = np.shape(Hp)[1]
    #####################################
    y = np.random.normal(0, np.pi/4, (d,N))
    Vt = np.matmul(Hp,y)
    #####################################
    p_perp = np.eye(D+1)-np.outer(p,p.T)
    noise = sigma*np.random.normal(0, np.pi/4, (D+1,N))
    noise = np.matmul(p_perp,noise)
    #####################################
    Vt = Vt + noise
    #####################################
    noise_lvl = 0
    for n in range(N):
        x = noise[:,n]
        noise_lvl = noise_lvl + np.linalg.norm(x)/N
    #####################################
    return Vt, noise_lvl
###########################################################################
def spherical_exp(Vt,S):
    p = S.p
    #####################################
    D = np.shape(Vt)[0]-1
    N = np.shape(Vt)[1]
    X = np.zeros( (D+1, N) )
    for n in range(N):
      v = Vt[:,n]
      norm_v = np.linalg.norm(v)
      x = np.cos(norm_v)*p+(np.sin(norm_v)/norm_v)*v
      X[:,n] = x #/np.linalg.norm(x)
    return X
###########################################################################
def random_spherical_data(param):
    S = random_spherical_subspace(param)
    #####################################
    Vt, noise_lvl = random_spherical_tangents(S,param)
    #####################################
    X = spherical_exp(Vt,S)
    #####################################
    return X,S,noise_lvl
###########################################################################
def estimate_spherical_subspace(X,param):
    S = subspace()
    d = param.d
    N = param.N
    #####################################
    Cx = np.matmul(X,X.T) #Cx = (Cx + Cx.T)/2
    evals , evecs = np.linalg.eig(Cx)
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    H = evecs[:,0:d+1]
    S.H = H
    S.p = evecs[:,0]
    S.Hp = evecs[:,1:d+1]
    #####################################
    PH = np.matmul(H,H.T)
    X_ = np.matmul(PH,X)
    for n in range(N):
        x_ = X_[:,n]
        X_[:,n] =  x_/np.linalg.norm(x_)
    #####################################
    return X_,S
###########################################################################
def subspace_dist(S,S_):
    H = S.H
    H_ = S_.H
    #####################################
    T = np.matmul(H.T,H_)
    #####################################
    SVs = scipy.linalg.svdvals(T)
    #print(T)
    #print(SVs)
    SVs = np.minimum(SVs,1)
    #print(SVs)
    #####################################
    dist =  np.sqrt( np.sum(np.arccos(SVs)**2) )
    return dist
###########################################################################