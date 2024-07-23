import pandas as pd
import numpy as np
import dataVisualization as dv
import os
import re

def CSVReader(file_path):
    df = pd.read_csv(file_path,skiprows=5)
    df = df.iloc[:, 1:]  # This selects all rows and all columns except the first one
    
    # Check if there are any NaN values in the DataFrame
    if df.isna().any().any():
        print("There are NaN values in the CSV.")
        df.fillna(37, inplace=True)
    else:
        print("No NaN values in the CSV.")

    # Dictionary to hold the matrices for each timestep
    timestep_matrices = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Convert the row into a numpy array, excluding the first column if it contains time or any non-temperature value
        time_step = index * 5
        temp_data = row[1:].values  # Adjust slicing if your time column is still present
        
        # Reshape the data into a 101x101 matrix
        try:
            matrix = temp_data.reshape(101, 101)
            rotated_matrix = np.rot90(matrix, k=-1)  # Rotate 90 degrees clockwise
            timestep_matrices[time_step] = rotated_matrix
        except ValueError as e:
            print(f"Reshape error at index {index} with error: {e}")

    #Print a single timestep matrix's dimension
    print("Shape of a matrix", timestep_matrices[0].shape)  # This should print (101, 101)
    #Print number of timesteps
    print("Number of timesteps",len(timestep_matrices))

    """match = re.search(r'k_tumour = (\d+\.\d+)', file_path)
    if match:
        param_value = match.group(1)
        mp4_filename = f'k_tumour_{param_value}'
        
    # Visualize the data
    dv.createAnimation(timestep_matrices, mp4_filename, 'modelsss')"""
    
    return timestep_matrices
