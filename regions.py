import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

def parameterRegions(tissue_value, tumor_value):
    # Define the size of the tensor
    size = 101

    # Create a meshgrid of coordinates
    x = np.linspace(0, 6, size)  # assuming 10cm total size
    y = np.linspace(0, 6, size)
    xx, yy = np.meshgrid(x, y)

    # Define the center and radius of the semi-circle
    center = (0, 3)  # center at the middle height on the left edge
    radius = 1.5  # cm

    tensor_np = np.full((size, size), tissue_value)

    # Calculate the distance from each point to the center
    distance_from_center = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)

    # Apply the semi-circle value within the defined radius and region (right side of tensor)
    mask = (distance_from_center <= radius) & (xx >= center[0])
    tensor_np[mask] = tumor_value

    rotated_tensor_np = np.rot90(tensor_np, k=-1)
    tensor_np_copy = rotated_tensor_np.copy()
    tensor_pt = torch.tensor(tensor_np_copy, dtype=torch.float)
    
    #tensor_df = pd.DataFrame(tensor_np_copy)
    #tensor_df.to_csv('tensor__auto_values.csv', index=False)
    # Visualize the tensor
    #plt.imshow(tensor_np_copy, extent=(0, 6, 0, 6))
    #plt.colorbar()
    #plt.title("Tensor with Semi-Circle Region Originating from Top Edge")
    #plt.show()

    return tensor_pt