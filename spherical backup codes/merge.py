import numpy as np
import spaceform_pca_lib as sfpca
import os
import pandas as pd
from scipy.stats import wasserstein_distance
import scipy.stats as stats


#dataset_name = 'doi_10_5061_dryad_pk75d__v20150519'

dataset_name = 'GUniFrac'
directory = "../results/" + dataset_name + "/"
results_filename = os.path.join(directory, "distortions_results.csv")
print(results_filename)
df = pd.read_csv(results_filename)
print(df)
filtered_df = df[df['Method'] != 'estimate_spherical_subspace_pga']
#print(filtered_df)
#######################
distortion_columns = [col for col in filtered_df.columns if col.endswith('_distortion')]
filtered_df = filtered_df.copy()
filtered_df[distortion_columns] = 100 - filtered_df[distortion_columns]
#######################
#print(filtered_df)

average_values = filtered_df.groupby('d').mean()
#print(average_values)
filtered_df = filtered_df.merge(average_values, on='d', suffixes=('', '_avg'))
#print(filtered_df)

cols = df.columns.values[2:10]

for col in cols:
    filtered_df[col+'_p'] = (filtered_df[col] - filtered_df[col+'_avg']) / filtered_df[col+'_avg'] * 100
#print(filtered_df)
final_df_mean = filtered_df.groupby('Method').mean()
print(final_df_mean.filter(regex='_p$'))
final_df_std = filtered_df.groupby('Method').std()
#print(final_df_std.filter(regex='_p$'))