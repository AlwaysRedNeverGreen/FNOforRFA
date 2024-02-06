"""
This module is designed for evaluating and visualizing the performance of a trained model on a given test dataset. It uses the trained model to predict temperature distributions for a series of timesteps and visualizes these predictions as heatmaps to compare against the ground truth data. 

The primary function, eval, iterates through the test dataset, applies the trained model to generate predictions for each batch, and visualizes these predictions. The visualization provides an intuitive understanding of the model's accuracy in predicting temperature distributions over time.

Function:
- eval(test_loader, loaded_model, input_seq_len): Takes a DataLoader containing the test dataset, a trained model, and the input sequence length used during training. It produces visualizations for each timestep predicted by the model, aiding in qualitative analysis of model performance.
"""

import torch
import dataVisualization as dv

def eval(test_loader, loaded_model, input_seq_len):
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_test, y_true = batch['x'], batch['y']
            y_pred = loaded_model(x_test) 

            for i in range(y_pred.size(1)):  # Assuming y_pred is of shape (batch_size, timesteps, height, width)
                    # Calculate the timestep being shown
                    timestep_idx = (batch_idx * test_loader.batch_size + i) * 5 + (input_seq_len - 1) * 5
                    predicted_timestep = f"T{timestep_idx + 5:03d}"  # The predicted timestep

                    # Get the prediction for the current timestep
                    y_pred_timestep = y_pred[0, i].cpu().numpy()  # Convert to numpy array
                    # Plot the heatmap
                    dv.plot_heatmap(y_pred_timestep, predicted_timestep, title="Predicted Temperature Distribution")