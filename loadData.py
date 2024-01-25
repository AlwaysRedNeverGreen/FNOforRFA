import scipy.io
import numpy as np

# Load the data from the file and return a dictionary of variables with the time steps as keys and the matrices as values
def loadData(file_path):
# Load the .mat file
    data = scipy.io.loadmat(file_path)

    # Convert the data to a dictionary
    data = {k: v for k, v in data.items()}

    # Extract time step keys and convert them to integers (assuming they all have the same format, e.g., T###)
    time_step_keys = sorted(k for k in data.keys() if k.startswith('T'))
    time_steps = [int(k[1:]) for k in time_step_keys]

    # Now find the range of your time steps
    start_time = min(time_steps)
    end_time = max(time_steps) + 1  # add 1 because range end is exclusive

    # If the interval is consistent, you can find it by subtracting the first two time steps
    if len(time_steps) > 1:
        interval = time_steps[1] - time_steps[0]
    else:
        interval = 1  # default to 1 if there's only one time step

    # Now you have a dynamic range of time steps
    dynamic_time_steps = range(start_time, end_time, interval)

    # Use the dynamic range in further processing
    variables = {}
    for t in dynamic_time_steps:
        key = f'T{t:03d}'
        if key in data:
            matrix = data[key]
            if matrix.shape != (101, 101):
                # Reshape the matrix to 101x101
                resized_matrix = np.resize(matrix, (101, 101))                
                # Rotate the matrix 90 degrees to the right
                rotated_matrix = np.rot90(resized_matrix, -1)
                variables[key] = rotated_matrix

            else:
                variables[key] = matrix
        else:
            print(f"Variable {key} not found in the .mat file")
    
    return variables
            
# Print the data
def printData(variables):
    for key, matrix in variables.items():
        print(f"Data for {key}:")
        print(matrix)
        print("\n")  # Adds an empty line for readability

