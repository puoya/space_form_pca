import pandas as pd
import numpy as np

def categorical_to_numeric(column):
    """
    Converts categorical values in a DataFrame column to numerical categories.
    
    Parameters:
    - column: The pandas DataFrame column containing categorical values.
    
    Returns:
    - A new pandas Series with numerical categories.
    """
    # Get unique categorical values
    unique_categories = column.unique()
    
    # Create a mapping from categorical values to numerical categories
    category_mapping = {category: i for i, category in enumerate(unique_categories)}
    
    # Map categorical values to numerical categories
    numeric_categories = column.map(category_mapping)
    
    return numeric_categories


# Read the data from the Metadata.tab file
df = pd.read_csv("/Users/puoya.tabaghi/Downloads/spherical/data/GUniFrac/doc/csv/label.csv")
#df.columns = df.columns.str.strip()

name = 'SmokingStatus'
df[name+'_cat'] = categorical_to_numeric(df[name])
categories = df[name+'_cat'].values
print(categories)
directory = "data/GUniFrac/"
np.save(directory + name+"_categories.npy", categories)

# Extract the Age column
# ages = df['Age']

# Define a function to categorize age into groups
def categorize_age(age):
    if age <= 40:
        return 0
    elif 40 < age <= 60:
        return 1
    elif 60 < age :
        return 
    else:
        return 4  # Handle any other cases

#name = 'BMI_group'

# Example usage:
# Assuming df is your DataFrame and 'Sex' is the column you want to convert
# Replace 'Sex' with the actual column name you want to convert
# df[name+'_cat'] = categorical_to_numeric(df[name])


# # Print the DataFrame with the new 'Age_Group' column
# categories = df[name+'_cat'].values
# print(categories)
# directory = "data/doi_10_5061_dryad_pk75d__v20150519/"
# np.save(directory + name+"_categories.npy", categories)


