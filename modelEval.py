import torch
import dataVisualization as dv

"""This block is for visualizing the predictions of the model for evaluation purposes"""
def eval(model,test_loader, input_seq_len):
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(test_loader):
            x_test, y_true = batch['x'], batch['y']
            y_pred = model(x_test)  # Get model's predictions
            print("shape is",y_pred.shape)
            # Calculate the index for the predicted timestep
            for i in range(y_pred.size(1)):  # Assuming y_pred is of shape (batch_size, timesteps, height, width)
                # Calculate the timestep being shown
                timestep_idx = (batch_idx * test_loader.batch_size + i) * 5 + (input_seq_len - 1) * 5
                predicted_timestep = f"T{timestep_idx + 5:03d}"  # The predicted timestep

                # Get the prediction for the current timestep
                y_pred_timestep = y_pred[0, i].cpu().numpy()  # Convert to numpy array
                # Plot the heatmap
                dv.plot_heatmap(y_pred_timestep, predicted_timestep, title="Predicted Temperature Distribution")