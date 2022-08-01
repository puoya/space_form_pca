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
def random_J_orthogonal_matrix(param):
    D = param.D
    d = param.d
    #####################################
    J = np.eye(D+1)
    J[0,0] = -1
    #####################################
    H = np.zeros((D+1,d))
    #####################################
    # generate p
    condition = True
    while condition:
        v = np.random.normal(0, 1, (D+1,1))
        v[0] = 100*np.random.normal(0, 1)
        norm_v = J_norm(v,D)
        if norm_v < 0:
            p = v/np.sqrt(-norm_v)
            if p[0] < 0:
                p = -p
            condition = False
    #####################################
    condition = True
    Pp_perp = np.eye(D+1)+np.matmul(np.matmul(p,p.T),J)
    i = 0
    while condition:
        v = np.random.normal(0, 1, (D+1,1))
        v = np.matmul(Pp_perp,v)
        PH_perp = np.eye(D+1)-np.matmul(np.matmul(H,H.T),J)
        v = np.matmul(PH_perp,v)
        norm_v = J_norm(v,D)
        if norm_v > 0:
            v = v/np.sqrt(norm_v)
            H[:,i] = v[:,0]
            i = i +1
        condition = (i < d)
    return H,p
###########################################################################
def J_norm(x,D):
    #####################################
    J = np.eye(D+1)
    J[0,0] = -1
    #####################################
    norm_x = np.matmul(x.T, np.matmul(J,x))
    norm_x = np.squeeze(norm_x)
    return norm_x
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
def random_hyperbolic_subspace(param):
    H,p = random_J_orthogonal_matrix(param)
    #####################################
    S = subspace()
    S.Hp = H
    S.p = p
    S.H = np.concatenate((p,H),1)
    #####################################
    # J = np.eye(101)
    # J[0,0] = -1
    # H_ = S.H
    # print(np.matmul(H_.T, np.matmul(J,H_)) )
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
    return Vt
###########################################################################
def random_hyperbolic_tangents(S,param):
    N = param.N
    sigma = param.sigma
    #####################################
    Hp = S.Hp
    p = S.p
    #####################################
    D = np.shape(Hp)[0]-1
    d = np.shape(Hp)[1]
    #####################################
    J = np.eye(D+1)
    J[0,0] = -1
    #####################################
    y = np.random.normal(0, np.pi/4, (d,N))
    Vt = np.matmul(Hp,y)
    #####################################
    Pp_perp = np.eye(D+1)+np.matmul(np.matmul(p,p.T),J)
    noise = sigma*np.random.normal(0, np.pi/4, (D+1,N))
    noise = np.matmul(Pp_perp,noise)
    #####################################
    Vt = Vt + noise
    #####################################
    return Vt
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
def hyperbolic_exp(Vt,S):
    p = S.p
    p = np.squeeze(p)
    #####################################
    D = np.shape(Vt)[0]-1
    N = np.shape(Vt)[1]
    X = np.zeros( (D+1, N) )
    for n in range(N):
      v = Vt[:,n]
      norm_v = np.sqrt(J_norm(v,D))
      x = np.cosh(norm_v)*p+(np.sinh(norm_v)/norm_v)*v
      norm_x = J_norm(x,D)
      X[:,n] = x/np.sqrt(-norm_x)
    return X
###########################################################################
def spherical_log(X,S):
    p = S.p
    #####################################
    D = np.shape(X)[0]-1
    N = np.shape(X)[1]
    V = np.zeros( (D+1,N) )
    for n in range(N):
      x = X[:,n]
      theta = np.arccos( np.matmul(x.T,p) )
      V[:,n] = ( theta/np.sin(theta) ) * ( x-p*np.cos(theta) )
    return V
###########################################################################
def random_spherical_data(param):
    S = random_spherical_subspace(param)
    #####################################
    Vt = random_spherical_tangents(S,param)
    #####################################
    X = spherical_exp(Vt,S)
    #####################################
    noise_lvl = compute_noise_lvl(X,S)
    return X,S,noise_lvl
###########################################################################
def random_hyperbolic_data(param):
    S = random_hyperbolic_subspace(param)
    #####################################
    Vt = random_hyperbolic_tangents(S,param)
    #####################################
    X = hyperbolic_exp(Vt,S)
    #####################################
    noise_lvl = compute_H_noise_lvl(X,S)
    return X,S,noise_lvl
###########################################################################
def compute_H_noise_lvl(X,S):
    H = S.H
    D = np.shape(H)[0]-1
    d = np.shape(H)[1]-1
    N = np.shape(X)[1]
    #####################################
    J = np.eye(D+1)
    J[0,0] = -1
    J_ = np.eye(d+1)
    J_[0,0] = -1
    #####################################
    X_ = np.matmul(np.matmul(H.T, J),X)
    noise_lvl = 0
    for n in range(N):
        x = X_[:,n]
        tmp = np.sqrt(-np.matmul(x.T,np.matmul(J_,x)))
        tmp = np.maximum(tmp,1)
        noise_lvl = noise_lvl + np.arccosh(tmp)/N
    return noise_lvl
###########################################################################
def compute_noise_lvl(X,S):
    H = S.H
    noise_lvl = np.linalg.norm(np.matmul(H.T,X),2,axis = 0)
    noise_lvl = np.minimum(noise_lvl,1)
    noise_lvl = np.arccos(noise_lvl)
    noise_lvl = np.mean(noise_lvl)
    return noise_lvl
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
def estimate_hyperbolic_subspace(X,param):
    S = subspace()
    d = param.d
    N = param.N
    #####################################
    Cx = np.matmul(X,X.T)/N 
    D = np.shape(Cx)[0]-1
    evals, eval_signs,evecs = compute_J_evs(Cx,param)
    #print(evals)
    #print(eval_signs)
    #print(np.shape(evecs) )
    #print(evecs)
    J = np.eye(D+1)
    J[0,0] = -1
    #print(np.matmul(np.matmul(evecs.T, J),evecs) )

    
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
def compute_J_evs(Cx,param):
    D = np.shape(Cx)[0]-1
    J = np.eye(D+1)
    J[0,0] = -1
    evals = []
    eval_signs = []
    condition = True
    count = 0
    evecs = []
    while condition:
        v = np.random.randn(D+1,1) 
        condition_i = True
        count = count + 1 
        while condition_i:
            v_ = np.matmul(np.matmul(Cx,J),v)
            v_ = np.matmul(np.matmul(Cx,J),v_)
            v_ = v_/np.sqrt(np.abs(J_norm(v_,D))) 
            condition_i = np.linalg.norm(v-v_)/(D+1) > count*(10**(-10))
            v = v_
        sgn = np.sign(J_norm(v,D))
        if count == 1:
            evecs = v
        else:
            evecs = np.concatenate( (evecs,v) , axis = 1)
        lmbd = np.matmul(np.matmul(v.T,J), np.matmul(np.matmul(Cx,J),v))
        lmbd = np.squeeze(lmbd)
        evals = np.append(evals,lmbd)
        eval_signs = np.append(eval_signs,sgn)
        Cx = Cx - lmbd*np.matmul(v,v.T)
        condition = check_evals(eval_signs,param)
    return evals, eval_signs, evecs

def check_evals(eval_signs,param):
    evals_p = np.sum(eval_signs>0)
    evals_n = np.sum(eval_signs<0)
    d = param.d
    condition = True
    if (evals_p == d) and (evals_n >= 1):
        condition = False
    return condition




def estimate_spherical_subspace_liu(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    normX = np.linalg.norm(X,'fro') 
    #####################################
    I = np.eye(D+1)
    H = I[:,0:d+1]
    V = np.random.randn(d+1,N)
    err = 1
    #####################################
    lambd = 1000
    mu = lambd
    condition = True
    while condition:
        for n in range(N):
            V[:,n] = (lambd-2)*V[:,n]+2*np.matmul(H.T,X[:,n])
            V[:,n] = V[:,n]/np.linalg.norm(V[:,n])
        M = 2*np.matmul( (X-np.matmul(H,V) ), V.T)+mu*H
        le,_,re = np.linalg.svd(M,full_matrices=False)
        H = np.matmul(le,re)
        X_ = np.matmul(H,V)
        err_ = np.linalg.norm(X-X_,'fro')/normX
        condition = np.abs(err-err_)  > 1e-4
        err =  err_
    S.H = H 
    S.p = H[0,:]
    S.Hp = H[0,1:d+1]
    return X_,S
########################################################################### 
def estimate_spherical_subspace_pga(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    tau = 0.1

    p = X[:,0]+0.1
    S.p  = p/np.linalg.norm(p)
    err = 1
    condition = True
    while condition:
        V = spherical_log(X,S)
        delta_p = tau * np.mean(V,1)
        delta_p = delta_p.reshape(D+1,1)
        p = np.concatenate( spherical_exp(delta_p,S) )
        err_ = np.linalg.norm(p-S.p)/(D+1)
        condition = np.abs(err-err_)  > 1e-8
        err = err_
        S.p = p
    #####################################    
    V = spherical_log(X,S)
    evals , evecs = np.linalg.eig( np.matmul(V,V.T))
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    Hp = evecs[:,0:d]
    S.Hp = Hp
    p = S.p
    H = np.concatenate( (p.reshape(D+1,1),Hp),axis = 1)
    S.H = H
    Vt = np.matmul( np.matmul(Hp,Hp.T) , V)
    X_ = spherical_exp(Vt,S)
    return X_,S
########################################################################### 
def estimate_spherical_subspace_dai(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    #####################################    
    e1 = np.zeros((D+1,))
    e1[0] = 1
    #####################################    
    tau = 0.1
    p = X[:,0]+0.1
    S.p  = p/np.linalg.norm(p)
    err = 1
    condition = True
    eps = 1e-5
    #####################################    
    while condition:
        V = spherical_log(X,S)
        delta_p = tau * np.mean(V,1)
        delta_p = delta_p.reshape(D+1,1)
        p = np.concatenate( spherical_exp(delta_p,S) )
        err_ = np.linalg.norm(p-S.p)/(D+1)
        condition = np.abs(err-err_)  > 1e-8
        err = err_
        S.p = p
    #####################################    
    p = S.p
    S.p = e1
    condition = True
    cost = 1e10
    while condition:
        cost_ = np.mean( np.arccos( np.matmul(X.T,p) )**2 )
        condition = np.abs(cost_-cost)  > 1e-14
        cost = cost_
        Vp = spherical_log(p.reshape(D+1,1),S)
        g = np.zeros((D+1,1))
        for i in range(D):
            x = Vp
            x[i+1,0] = x[i+1,0] + eps
            pi = spherical_exp(x,S) 
            cost_ = np.mean( np.arccos( np.matmul(X.T,pi) )**2 )
            g[i+1,0] = (cost_ - cost)/eps
        p = spherical_exp(Vp -0.1*g,S) 
    S.p = np.concatenate(p)
    V = spherical_log(X,S)
    evals , evecs = np.linalg.eig( np.matmul(V,V.T))
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    Hp = evecs[:,0:d]
    S.Hp = Hp
    p = S.p
    H = np.concatenate( (p.reshape(D+1,1),Hp),axis = 1)
    S.H = H
    Vt = np.matmul( np.matmul(Hp,Hp.T) , V)
    X_ = spherical_exp(Vt,S)
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