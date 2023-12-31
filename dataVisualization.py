import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
    
def findMinMax(variables):
    all_values = np.concatenate([v.ravel() for v in variables.values()])
    return all_values.min(), all_values.max()
    
def viewData(variables):
    for key, value in variables.items():
        #vmin, vmax = findMinMax(variables)
        #print("Le values:", vmin, vmax)
        # Assuming the value is a 2D matrix representing temperature distribution
        plt.imshow(value, cmap='hot', interpolation='nearest', vmin=0, vmax=100)
        plt.title(f"Temperature Distribution at {key}")
        plt.colorbar()
        plt.show()
        
def animate(i, variables, min_temp, max_temp, time_steps):
    plt.clf()
    key = f'T{time_steps[i]:03d}'
    if key in variables:
        plt.imshow(variables[key], cmap='plasma', interpolation='nearest', vmin=0, vmax=max_temp)
        plt.title(f"Temperature Distribution at {key}")
        plt.colorbar()

def createAnimation(variables, case):
    min_temp, max_temp = 0,100
    time_steps = sorted([int(k[1:]) for k in variables.keys() if k.startswith('T')])
    fig = plt.figure()
    anim = FuncAnimation(fig, animate, frames=len(time_steps), fargs=(variables, min_temp, max_temp, time_steps), interval=200)
    anim.save(f'{case}_heatmap_animation.mp4', writer='ffmpeg')
    plt.show()
    
def plot_heatmap(matrix, timestep, title="Heatmap", vmin=0, vmax=100):
    plt.imshow(matrix, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(f"{title} - {timestep}")
    plt.colorbar()
    plt.show()