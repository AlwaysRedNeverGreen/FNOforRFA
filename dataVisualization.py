"""
This module provides visualization tools for temperature distribution data. 
It includes functions to display static heatmaps of temperature data, generate animated visualizations over a series of timesteps

Functions:
- findMinMax(variables): Computes the minimum and maximum temperature values across all timesteps in the dataset for consistent color scaling in visualizations.
- viewData(variables): Displays a series of heatmaps for temperature distributions at different timesteps. This is useful for visualizing the data before training a model.
- animate(i, variables, min_temp, max_temp, time_steps): Helper function to generate frames for the animation, showing temperature distribution at each timestep.
- createAnimation(variables, case): Produces an animated visualization of temperature distributions over time, saved as a video file.
- plot_heatmap(matrix, timestep, title, vmin, vmax): Plots a single heatmap for a given temperature matrix at a specific timestep, with customizable color scaling.
- animateComparison Helper function to generate frames for the comparison animation, showing ground truth, predictions, and other data.
- createAnimationComparisonProduces an animated comparison of ground truth, predicted, and other data over time, saved as a video file.
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
        plt.imshow(value, cmap='plasma', interpolation='nearest', vmin=0, vmax=100)
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
    
def animateComparison(i, ground_truth, predictions, other, min_temp, max_temp):
    plt.clf()
    
    gt_key = list(ground_truth.keys())[i]
    pred_key = list(predictions.keys())[i]
    other_key = list(other.keys())[i]

    # Create subplots for each type of data
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # Ground Truth
    gt_data = ground_truth[gt_key].squeeze()
    im1 = ax1.imshow(gt_data, cmap='plasma', interpolation='nearest', vmin=min_temp, vmax=max_temp)
    ax1.set_title(gt_key)
    plt.colorbar(im1, ax=ax1)

    # Predictions
    pred_data = predictions[pred_key].squeeze()
    im2 = ax2.imshow(pred_data, cmap='plasma', interpolation='nearest', vmin=min_temp, vmax=max_temp)
    ax2.set_title(pred_key)
    plt.colorbar(im2, ax=ax2)

    # Other Data
    other_data = other[other_key].squeeze()
    im3 = ax3.imshow(other_data, cmap='plasma', interpolation='nearest', vmin=min_temp, vmax=max_temp)
    ax3.set_title(other_key)
    plt.colorbar(im3, ax=ax3)

def createAnimationComparison(ground_truth, predictions, differences, case,k,w,sig, model):
    frames = len(ground_truth)
    min_temp, max_temp = 0, 100
    fig = plt.figure(figsize=(15, 5))  # Adjust the size to your preference

    anim = FuncAnimation(fig, animateComparison, frames=frames,
                         fargs=(ground_truth, predictions, differences, min_temp, max_temp),
                         interval=200, repeat=True)

    anim.save(f'Predicted Videos/{case}_comparisonHeatMap_using_{model},k{k},w{w},sig{sig}.mp4', writer='ffmpeg')
    plt.show()