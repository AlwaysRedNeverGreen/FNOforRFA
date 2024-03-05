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
    current_timestep = last_timestep + 5
    predictions = [] # List to store the model's predictions
    ground_truths = [] # List to store the ground truth temperature distributions
    timesteps = [] # List to store the timesteps for each prediction
    error_list = [] # List to store the error for each prediction
    results_dict = {} # Dictionary to store the results for each timestep
    exit_counter = 0 
    skip_counter = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
                x_test, y_true = batch['x'], batch['y']  # Extract input and output sequences from the batch
                
                if batch_idx == 0: 
                    y_pred = loaded_model(x_test)  # Apply the model to the test input
                    print(f"Original input T{current_timestep}:\n {x_test}")
                
                elif exit_counter == prediction_len - 1:
                     exit_counter = 0
                     skip_counter = 1
                     print("Skipping")
                     continue
                 
                elif skip_counter == 1:
                    print("Skipped")
                    current_timestep += (5*input_seq_len)
                    y_pred = loaded_model(x_test)
                    skip_counter = 0
                    results_dict[f'Original input T{current_timestep}'] = x_test
                    print(f"Original input T{current_timestep}:\n {x_test}")
                    
                else:
                    #results_dict[f'Predicted Input T{current_timestep}'] = y_pred
                    print(f"Predicted Input T{current_timestep}:\n {y_pred}")
                    y_pred = loaded_model(y_pred )                    
                    exit_counter = exit_counter + 1
            
                for i in range(y_pred.size(1)):
                    current_timestep += (5*input_seq_len)
                    timesteps.append(current_timestep)
                    #print(f"Predicting timestep {current_timestep}")
                    
                    y_pred_timestep = y_pred[0, i].cpu().numpy().squeeze() # Extract the predicted temperature distribution for the current timestep
                    results_dict[f'Prediction T{current_timestep}'] = y_pred
                    predictions.append(y_pred_timestep)
                    print(f"Prediction at T{current_timestep}:\n {y_pred_timestep}")
                    
                    y_true_sample = y_true[0, i].cpu().numpy().squeeze() # Extract the ground truth temperature distribution for the current timestep
                    ground_truths.append(y_true_sample)
                    print(f"Ground Truth at {current_timestep}:\n {y_true_sample}")
                    
                    errors = (y_true - y_pred) ** 2
                    mse = torch.mean(errors)
                    rmse = torch.sqrt(mse)
                    error_list.append(rmse.item())
                    
                    print(f"Mean Squared Error for timestep {current_timestep}: {rmse.item()}")
    
    print(f"Mean of the error list: {sum(error_list) / len(error_list)}")
    #print(results_dict)
    #dv.createAnimation(results_dict, 'case03')  # Create an animation of the predictions over time
    
    #for i in range(len(predictions)):
        #print(f"Plotting comparison heatmap for timestep {timesteps[i]}")
        #dv.plot_comparison_heatmaps(ground_truths[i], predictions[i], timesteps[i])  # Visualize the comparison between the predicted and true temperature distributions