"""
This module provides visualization tools for temperature distribution data. It includes functions to display static heatmaps of temperature data, generate animated visualizations over a series of timesteps

Functions:
- findMinMax(variables): Computes the minimum and maximum temperature values across all timesteps in the dataset for consistent color scaling in visualizations.
- viewData(variables): Displays a series of heatmaps for temperature distributions at different timesteps. This is useful for visualizing the data before training a model.
- animate(i, variables, min_temp, max_temp, time_steps): Helper function to generate frames for the animation, showing temperature distribution at each timestep.
- createAnimation(variables, case): Produces an animated visualization of temperature distributions over time, saved as a video file.
- plot_heatmap(matrix, timestep, title, vmin, vmax): Plots a single heatmap for a given temperature matrix at a specific timestep, with customizable color scaling.
- plot_comparison_heatmaps plots heatmaps of ground truth and predicted temperature distributions for comparison at a specific timestep.
Dependencies:
- matplotlib: Used for creating static and animated visualizations.
- numpy: Utilized for numerical operations, particularly finding the min and max values for temperature scaling.
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
 
def findMinMax(variables):
    all_values = np.concatenate([v.ravel() for v in variables.values()])
    return all_values.min(), all_values.max()
   
def viewData(variables):
    for key, value in variables.items():
        plt.imshow(value, cmap='hot', interpolation='nearest', vmin=0, vmax=100)
        plt.title(f"Temperature Distribution at {key}")
        plt.colorbar()
        plt.show()
        
def animate(i, variables, min_temp, max_temp, keys):
    plt.clf()
    key = keys[i]  # Use the key directly from the list of keys
    if key in variables:
        image_data = variables[key].squeeze()
        plt.imshow(image_data, cmap='plasma', interpolation='nearest', vmin=min_temp, vmax=max_temp)
        if 'Predicted' or 'Prediction' in key:  # Check if the key indicates a predicted timestep
            plt.title(f'{key}')
        else:
            plt.title(f'{key}')
        plt.colorbar()

def createAnimation(variables, case, model):
    for key in list(variables.keys()):  # Use list to avoid RuntimeError for changing dict size during iteration
        if isinstance(variables[key], torch.Tensor):
            variables[key] = variables[key].cpu().detach().numpy()
            
    min_temp, max_temp = 0,100
    keys = [k for k in variables.keys()]  # Directly use keys from the dictionary
    fig = plt.figure()
    anim = FuncAnimation(fig, animate, frames=len(keys), fargs=(variables, min_temp, max_temp, keys), interval=200)
    anim.save(f'{case}_heatmap_animation_{model}.mp4', writer='ffmpeg')
    plt.show()

def plot_heatmap(matrix, timestep, title="Heatmap", vmin=0, vmax=100):
    plt.imshow(matrix, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(f"{title} - {timestep}")
    plt.colorbar()
    plt.show()

def plot_comparison_heatmaps(y_true, y_pred, timestep, title_true="Ground Truth", title_pred="Prediction", vmin=0, vmax=100):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Temperature Distribution at {timestep}")

    # Plot Ground Truth
    im_true = axs[0].imshow(y_true, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[0].set_title(title_true)
    fig.colorbar(im_true, ax=axs[0], fraction=0.046, pad=0.04)

    # Plot Prediction
    im_pred = axs[1].imshow(y_pred, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[1].set_title(title_pred)
    fig.colorbar(im_pred, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()