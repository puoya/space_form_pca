# import numpy as np
# import pandas as pd
# import pickle
# import torch
# import os

# import scipy.linalg
# from sklearn.metrics import normalized_mutual_info_score


# #address = 'data/glass_identification/'




    









# X = np.load(address+'X.npy')
# D,N = np.shape(X)

# #labels = np.load(address+'label.npy')
# G = np.matmul(X.T,X)
# G[G<-1] = -1
# G[G>1] = 1
# DM = np.arccos(G)


# error_sfpca = np.array([])
# error_liu = np.array([])
# error_dai = np.array([])
# error_pga = np.array([])
# error_pga2 = np.array([])

# nmi_sfpca = np.array([])
# nmi_liu = np.array([])
# nmi_dai = np.array([])
# nmi_pga = np.array([])
# nmi_pga2 = np.array([])
# for d in range(10,80,10): 
#     print(d)
#     param = parameters()
#     param.D = D-1
#     param.d = d+1
#     param.N = N
#     idx = np.triu(np.ones((N,N)), 1)==1
#     X_, S_ = sfpca.estimate_spherical_subspace(X,param)


#     G_ = np.matmul(X_.T,X_)
#     G_[G_<-1] = -1
#     G_[G_>1] = 1
#     DM_ = np.arccos(G_)
#     #print(DM_)
#     error_sfpca = np.append(error_sfpca, np.linalg.norm(DM[idx]-DM_[idx]))

#     # tmp_in  = 0
#     # for ki in range(K):
#     #     #print(DM_[labels == ki,labels == ki])
#     #     idx = (labels == np.unique(labels)[ki])
#     #     DM_2 = DM_[idx,:]
#     #     DM_2 = DM_2[:,idx]
#     #     tmp_in = tmp_in + np.sum(DM_2)
#     #print(tmp_in)
#     #print(tmp_in / (np.sum(DM_) - tmp_in))
#     #error_sfpca = np.append(error_sfpca, tmp_in / (np.sum(DM_) - tmp_in))

#     #labels_, cost, _ = k_means(X_, K)
#     #nmi = normalized_mutual_info_score(labels, labels_)
#     #nmi_sfpca = np.append(nmi_sfpca, nmi)
#     # print(labels_)
#     # print('now')
#     # print(labels)
#     #print(np.unique(labels))
#     #print(nmi)

#     # X_, S_ = sfpca.estimate_spherical_subspace_liu(X,param)
#     # G_ = np.matmul(X_.T,X_)
#     # G_[G_<-1] = -1
#     # G_[G_>1] = 1
#     # DM_ = np.arccos(G_)
#     # error_liu = np.append(error_liu, np.linalg.norm(DM[idx]-DM_[idx]))
#     # tmp_in  = 0
#     # for ki in range(K):
#     #     #print(DM_[labels == ki,labels == ki])
#     #     idx = (labels == np.unique(labels)[ki])
#     #     DM_2 = DM_[idx,:]
#     #     DM_2 = DM_2[:,idx]
#     #     tmp_in = tmp_in + np.sum(DM_2)
#     # #print(tmp_in)
#     # #print(tmp_in / (np.sum(DM_) - tmp_in))
#     # error_liu = np.append(error_liu, tmp_in / (np.sum(DM_) - tmp_in))

#     # labels_, cost, _ = k_means(X_, K)
#     # nmi = normalized_mutual_info_score(labels, labels_)
#     # print(nmi)
#     # print('asd')
#     #nmi_liu = np.append(nmi_liu, nmi)

#     X_, S_ = sfpca.estimate_spherical_subspace_dai(X,param)
#     G_ = np.matmul(X_.T,X_)
#     G_[G_<-1] = -1
#     G_[G_>1] = 1
#     DM_ = np.arccos(G_)
#     error_dai = np.append(error_dai, np.linalg.norm(DM[idx]-DM_[idx]))
#     # labels_, cost, _ = k_means(X_, K)
#     # nmi = normalized_mutual_info_score(labels, labels_)
#     # nmi_dai = np.append(nmi_dai, nmi)
#     # tmp_in  = 0
#     # for ki in range(K):
#     #     #print(DM_[labels == ki,labels == ki])
#     #     idx = (labels == np.unique(labels)[ki])
#     #     DM_2 = DM_[idx,:]
#     #     DM_2 = DM_2[:,idx]
#     #     tmp_in = tmp_in + np.sum(DM_2)
#     # #print(tmp_in)
#     # #print(tmp_in / (np.sum(DM_) - tmp_in))
#     # error_dai = np.append(error_dai, tmp_in / (np.sum(DM_) - tmp_in))

#     X_, S_ = sfpca.estimate_spherical_subspace_pga(X,param)
#     G_ = np.matmul(X_.T,X_)
#     G_[G_<-1] = -1
#     G_[G_>1] = 1
#     DM_ = np.arccos(G_)
#     error_pga = np.append(error_pga, np.linalg.norm(DM[idx]-DM_[idx]))
#     # labels_, cost, _ = k_means(X_, K)
#     # nmi = normalized_mutual_info_score(labels, labels_)
#     # nmi_pga = np.append(nmi_pga, nmi)
#     # tmp_in  = 0
#     # for ki in range(K):
#     #     #print(DM_[labels == ki,labels == ki])
#     #     idx = (labels == np.unique(labels)[ki])
#     #     DM_2 = DM_[idx,:]
#     #     DM_2 = DM_2[:,idx]
#     #     tmp_in = tmp_in + np.sum(DM_2)
#     # #print(tmp_in)
#     # #print(tmp_in / (np.sum(DM_) - tmp_in))
#     # error_pga = np.append(error_pga, tmp_in / (np.sum(DM_) - tmp_in))

#     X_, S_ = sfpca.estimate_spherical_subspace_pga_2(X,param)
#     G_ = np.matmul(X_.T,X_)
#     G_[G_<-1] = -1
#     G_[G_>1] = 1
#     DM_ = np.arccos(G_)
#     error_pga2 = np.append(error_pga2, np.linalg.norm(DM[idx]-DM_[idx]))
#     # labels_, cost, _ = k_means(X_, K)
#     # nmi = normalized_mutual_info_score(labels, labels_)
#     # nmi_pga2 = np.append(nmi_pga2, nmi)
#     # tmp_in  = 0
#     # for ki in range(K):
#     #     #print(DM_[labels == ki,labels == ki])
#     #     idx = (labels == np.unique(labels)[ki])
#     #     DM_2 = DM_[idx,:]
#     #     DM_2 = DM_2[:,idx]
#     #     tmp_in = tmp_in + np.sum(DM_2)
#     # #print(tmp_in)
#     # #print(tmp_in / (np.sum(DM_) - tmp_in))
#     # error_pga2 = np.append(error_pga2, tmp_in / (np.sum(DM_) - tmp_in))

# #print('Ours is:', error_sfpca)
# #print((error_liu-error_sfpca)/error_sfpca * 100)
# print((error_dai-error_sfpca)/error_sfpca * 100)
# print((error_pga-error_sfpca)/error_sfpca * 100)
# print((error_pga2-error_sfpca)/error_sfpca * 100)
# print(error_sfpca)

# # print((nmi_liu-nmi_sfpca)/nmi_sfpca * 100)
# # print((nmi_dai-nmi_sfpca)/nmi_sfpca * 100)
# # print((nmi_pga-nmi_sfpca)/nmi_sfpca * 100)
# # print((nmi_pga2-nmi_sfpca)/nmi_sfpca * 100)
#     #print(X_)
        
#         # #np.load
#         # p = S_.Hp

#         # np.save(address+ name+'/'+ alg+'/d_val_' + str(d+1)+'/X_2.npy',X_2)
#         # print(address+ name+'/'+ alg+'/d_val_' + str(d+1)+'/X_2.npy')
#         #print(address+ name+'/'+ alg+'/d_val_' + str(d+1)+'/X_2.npy')
#         #X3 = np.load(address+ name+'/'+ alg+'/d_val_' + str(d+1)+'/X_T.npy')
#         #print(np.linalg.norm(X_2-X3))
#         # print(f)
#         # print(X)

        

#         # J = np.eye(D+1)
#         # J[0,0] = -1
#         # print(np.matmul(np.matmul(p.T,J),p))

#         # 
#         # print(G)
#         # G[G >= -1] = -1 
#         # Dist = np.arccosh( -G )
#         # dist = Dist[idx]
#         # print(dist)
        



#         # address_ = address+ name+'/'+ alg+'/d_val_' + str(d+1)+'/X_.npy'
#         # X_2 = X = np.load(address_)
#         # print(np.linalg.norm(X_ - X_2))
#         # X_ = X_.detach().cpu().numpy()
#         # G = np.matmul(X_,X_.T)
#         # N = np.shape(G)[0]
#         # dG = np.reshape(np.diag(G), [N,1] ) 
#         # dG1 = np.matmul(dG, np.ones((1,N)))
#         # D = -2*G+dG1 + dG1.T
#         # dG_inv = np.diag( 1/(1-np.diag(G)))
#         # D_rec = np.arccosh( 1+2*np.matmul(np.matmul(dG_inv, D), dG_inv) )
#         # dist_rec = D_rec[idx]
#         # ratio = np.divide( (dist - dist_rec), dist)
#         # print(np.mean(dist_rec))
#         # #print(d+1, np.floor(np.mean(ratio)*1000)/1000)
#         # #error = np.append(error, np.mean(ratio))
#         # #print(alg, np.floor(np.mean(error)*100)/100 , np.floor(np.std(error)*100)/100 )
        
        

    
    

# #output_noise_lvl
# #runtime

