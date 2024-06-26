import pandas as pd
import numpy as np
import sys

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
    #print(unique_categories)
    
    # Create a mapping from categorical values to numerical categories
    category_mapping = {category: i for i, category in enumerate(unique_categories)}
    
    # Map categorical values to numerical categories
    numeric_categories = column.map(category_mapping)
    
    return numeric_categories


# Read the data from the Metadata.tab file
df = pd.read_csv("Metadata.tab", sep="\t")
name = 'Age'
# Extract the Age column
#ages = df[name]
np.set_printoptions(threshold=sys.maxsize)
#print(df[name].values)

# # Define a function to categorize age into groups
def categorize_age(age_series):
    conditions = [
        age_series <= 40,
        (age_series > 40) & (age_series <= 60),
        age_series > 60
    ]
    choices = [0, 1, 2]
    #return np.select(conditions, choices, default=np.random.randint(0, 3))
    return np.select(conditions, choices, default=3)

#print(df)
#name = 'BMI_group'
#print(df)
# Example usage:
# Assuming df is your DataFrame and 'Sex' is the column you want to convert
# Replace 'Sex' with the actual column name you want to convert
#df[name+'_cat'] = categorize_age(df[name])
df[name+'_cat'] = categorize_age(df[name])

# Print the DataFrame with the new 'Age_Group' column
categories = df[name+'_cat'].values
print(categories)
#print(categories)
directory = ""
np.save( name+"_categories.npy", categories)


