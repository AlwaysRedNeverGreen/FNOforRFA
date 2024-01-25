import loadData as ld
import dataVisualization as dv
import splits as sp

from neuralop.models import FNO2d
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
import torch

file_path = 'Data/CASE01_DATA.mat'
input_seq_len=10 # The length of the input sequence

data_variables = ld.loadData(file_path) #Load the data from the file

"""This block is for visualizing the dataset"""
#ld.printData(data_variables) #Print the data
#dv.viewData(data_variables) #View the data
#dv.createAnimation(data_variables,"Case 2") #Create an animation of the data

train_ratio=0.8 #The ratio of the data to be used for training
train_loader, test_loader = sp.create_datasets(data_variables, input_seq_len=input_seq_len, prediction_len=1, train_ratio=0.8, batch_size=16) #Create the datasets

model = FNO2d(n_modes_width = 16, n_modes_height = 16, hidden_channels=32, in_channels=10, out_channels=1) #Create the model
n_params = count_params(model) #Count the number of parameters in the model
#print(f'\nOur model has {n_params} parameters.') #Tells us how many parameters are in our model which means how many weights we have to train

optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4) #Create the optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) #Create the scheduler

# Creating the losses
l2loss = LpLoss(d=2, p=2) # L2 loss for the heat equation
h1loss = H1Loss(d=2) # H1 loss for the heat equation

train_loss = h1loss # The loss we want to train
eval_losses={'h1': h1loss, 'l2': l2loss} # The losses we want to evaluate

#print('\n### MODEL ###\n', model)
#print('\n### OPTIMIZER ###\n', optimizer)
#print('\n### SCHEDULER ###\n', scheduler)
#print('\n### LOSSES ###')
#print(f'\n * Train: {train_loss}')
#print(f'\n * Test: {eval_losses}')

output_encoder = None # No encoder for the output

# Create the trainer
trainer = Trainer(model, n_epochs=30,
                  device="cpu",
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)

# Train the model
trainer.train(train_loader, test_loader,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

 # The length of the input sequence

model.eval()  # Set the model to evaluation mode

"""This block is for visualizing the predictions of the model for evaluation purposes"""
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