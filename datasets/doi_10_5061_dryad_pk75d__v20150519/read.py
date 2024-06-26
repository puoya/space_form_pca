import pandas as pd
from io import StringIO
import numpy as np
# Define the data as a string

# Create a DataFrame from the string data
metadata = pd.read_csv("Metadata.tab", sep="\t")

# Display the DataFrame
print(metadata)



# Read the text file
with open("HITChip.tab", "r") as file:
    # Read the first line to get column names
    columns = file.readline().strip().split("\t")
    
    # Read the rest of the file using pandas, specifying the separator as tab
    data = pd.read_csv(file, sep="\t", names=columns)

# Display the DataFrame
X = data.values

# Display the NumPy array
print(data)
np.save('X.npy', X.T)
