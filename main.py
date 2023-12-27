import loadData as ld
import dataVisualization as dv
import splits as sp

from neuralop.models import FNO2d
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
import torch

train_ratio=0.8

file_path = 'Data/CASE03_DATA.mat'
data_variables = ld.loadData(file_path)  # Ensure loadData returns the 'variables' dictionary
#ld.printData(data_variables)
#dv.viewData(data_variables)
#dv.createAnimation(data_variables,"Case 2")


train_loader, test_loader = sp.create_datasets(data_variables, input_seq_len=10, prediction_len=1, train_ratio=0.8, batch_size=16)

model = FNO2d(n_modes_width = 16, n_modes_height = 16, hidden_channels=32, in_channels=10, out_channels=1)
n_params = count_params(model)
#print(f'\nOur model has {n_params} parameters.') #Tells us how many parameters are in our model which means how many weights we have to train

optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

"""print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')"""

output_encoder = None  # Define the variable "output_encoder" before using it

trainer = Trainer(model, n_epochs=30,
                  device="cpu",
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)
trainer.train(train_loader, test_loader,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

input_seq_len=10
model.eval()  # Set the model to evaluation mode
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