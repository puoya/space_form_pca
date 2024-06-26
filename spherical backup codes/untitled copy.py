
    #directory = 'data/doi_10_5061_dryad_pk75d__v20150519/results/distance_distorions/'
#results_filename = directory+ "distortions_results.csv"
# directory = "/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/doc/csv/"
# results_filename = os.path.join(directory, "throat_otu.csv")
# df = pd.read_csv(results_filename)
# df_dropped = df.drop(columns=['Unnamed: 0'])
# df_dropped = df_dropped.astype(float)
# X = df_dropped.to_numpy()
# #print(type(X))
# for n in range(np.shape(X)[0]):
#     X[n,:] = X[n,:] / np.sum(X[n,:])
#     X[n,:] = np.sqrt(X[n,:])
# np.save("/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/doc/csv/X.npy",X)
    


# directory = "/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/"
# directory = "data/doi_10_5061_dryad_pk75d__v20150519/"
# #name = 'BMI_group'
# name = "SmokingStatus"
# labels = np.load(directory + name+"_categories.npy")


# Check if the directory exists, if not, create it





#distortion = total_wasserstein_distance( (X.T)**2,(X_.T)**2)
            #distortion = total_wasserstein_distance( (X_.T)**2,(X.T)**2)
            #print(np.linalg.norm((X_.T)-(X.T)**2))
            
        
            # DM_ = compute_sdm(X_)
            # DM1 = DM_[labels == 1,:]
            # DM1 = DM1[:,labels == 1]

            # tmp = np.sum(DM1)
            # DM1 = DM_[labels == 0,:]
            # DM1 = DM1[:,labels == 0]
            # tmp = tmp + np.sum(DM1)