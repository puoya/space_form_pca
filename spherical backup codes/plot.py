import numpy as np
import spaceform_pca_lib as sfpca
import os
import pandas as pd
from scipy.stats import wasserstein_distance
import scipy.stats as stats
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#dataset_name = 'doi_10_5061_dryad_pk75d__v20150519'
#dataset_name = 'GUniFrac'
dataset_name = 'document'
directory = "../results/" + dataset_name + "/"

#experiment = 'information_results.csv'
#experiment = 'distortions_results.csv'
experiment = 'classification_results.csv'
#experiment = 'sparsity_results.csv'
results_filename = os.path.join(directory, experiment)
#print(results_filename)
df = pd.read_csv(results_filename)
#print(df)
filtered_df = df[df['Method'] != 'estimate_spherical_subspace_pga']
print(filtered_df)
#######################
# distortion_columns = [col for col in filtered_df.columns if col.endswith('_distortion')]
# filtered_df = filtered_df.copy()
# filtered_df[distortion_columns] = 100 - filtered_df[distortion_columns]
#######################
# #print(filtered_df.columns)
# grouped_mean = (filtered_df.groupby('Method').mean()*100).round(2)
# grouped_std = (filtered_df.groupby('Method').std()*100).round(2)

# # Creating a new DataFrame to store the mean ± std for each column
# result = pd.DataFrame()

# # Calculating mean ± std for each column
# for column in grouped_mean.columns:
#     result[column + '_Mean ± Std'] = ('$' + grouped_mean[column].astype(str) + ' \pm ' + grouped_std[column].astype(str) + '$')

# # Displaying the result
# #print(result)
#print(filtered_df.columns)
grouped_mean = (filtered_df.groupby('Method').mean())
grouped_std = (filtered_df.groupby('Method').std())

# Creating a new DataFrame to store the mean ± std for each column
result = pd.DataFrame()

# Calculating mean ± std for each column
for column in grouped_mean.columns:
    avg = (grouped_mean.mean().loc[column])
    result[column + '_Mean ± Std'] = ('$' + (grouped_mean[column]/avg*100-100).round(2).astype(str) + ' \pm ' + (grouped_std[column]/avg*100 ).round(2).astype(str) + '$')

# Displaying the result
print(result)


# filtered_df2 = filtered_df.groupby('Method').mean()
# print(filtered_df2)

# ############### plot ####################
categories = df['Method'].unique()
print(filtered_df.columns)
num_categories = len(categories)
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, num_categories))

plt.figure(figsize=(10, 6))

for category, color in zip(categories, colors):
    # Filter the dataframe based on the category
    df_filtered = df[df['Method'] == category]
    #print(df_filtered.index)
    
    # Plotting
    plt.scatter(df_filtered['d'], df_filtered['average_acc_rf'], label=f'Category {category}', color=color, s=8)


plt.title('Scatter Plots for All Categories')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
#plt.show()
plt.savefig('plot.eps', format='eps', dpi=600)
# ############## plot ####################


# # average_values = filtered_df.groupby('Method').mean()

# # #print(average_values)
# # filtered_df = filtered_df.merge(average_values, on='d', suffixes=('', '_avg'))


# #print(filtered_df)

# #cols = df.columns.values[2:4] # information
# # cols = df.columns.values[2:8] # classification
# # #cols = df.columns.values[2:8] # distortion

# # for col in cols:
# #     filtered_df[col+'_p'] = (filtered_df[col] - filtered_df[col+'_avg']) / filtered_df[col+'_avg'] * 100
# # print(filtered_df)
# # final_df_mean = filtered_df.groupby('Method').mean()
# # print(final_df_mean.filter(regex='_p$'))
# # final_df_std = filtered_df.groupby('Method').std()
# # print(final_df_std.filter(regex='_p$'))