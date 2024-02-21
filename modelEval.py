"""
This module is designed for evaluating and visualizing the performance of a trained model on a given test dataset. It uses the trained model to predict temperature distributions for a series of timesteps and visualizes these predictions as heatmaps to compare against the ground truth data. 

The primary function, eval, iterates through the test dataset, applies the trained model to generate predictions for each batch, and visualizes these predictions. The visualization provides an intuitive understanding of the model's accuracy in predicting temperature distributions over time.

Function:
- eval(test_loader, loaded_model, input_seq_len): Takes a DataLoader containing the test dataset, a trained model, and the input sequence length used during training. It produces visualizations for each timestep predicted by the model, aiding in qualitative analysis of model performance.
"""
import torch
import dataVisualization as dv
import matplotlib.pyplot as plt
            
def eval(test_loader, loaded_model, input_seq_len, last_timestep_str, prediction_len=1, interval=5):
    last_timestep = int(last_timestep_str[1:])  # Skip the first character 'T' and convert the rest to an integer
    current_timestep = last_timestep + interval + (5*input_seq_len)
    counter = 0  # Counter to skip output for every other batch
    predictions = []
    ground_truths = []
    timesteps = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if counter % 2 == 0:
                x_test, y_true = batch['x'], batch['y']  # Extract input and output sequences from the batch
                y_pred = loaded_model(x_test)  # Apply the model to the test input

                for i in range(y_pred.size(1)):
                    y_pred_timestep = y_pred[0, i].cpu().numpy().squeeze()
                    y_true_sample = y_true[0, i].cpu().numpy().squeeze()
                    predicted_timestep = current_timestep + i * prediction_len * interval  # Calculate the timestep for the current prediction
                    predictions.append(y_pred_timestep)
                    ground_truths.append(y_true_sample)
                    timesteps.append(f"T{predicted_timestep}")
                    current_timestep += (test_loader.batch_size * prediction_len * interval) + 5  # Increment the current timestep
            counter += 1

    # Now, plot all predictions at the end
    for i in range(len(predictions)):
        print(f"Plotting comparison heatmap for timestep {timesteps[i]}")
        #print ground truth value
        print(ground_truths[i])
        dv.plot_comparison_heatmaps(ground_truths[i], predictions[i], timesteps[i])  # Visualize the comparison between the predicted and true temperature distributions