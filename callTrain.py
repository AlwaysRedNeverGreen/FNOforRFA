"""
This script defines and manages the training process for a machine learning model using the FNO2d architecture from the Neural Operator library. 
The script is adapted and modified for specific training configurations, loss evaluation, and learning rate scheduling. 
It also integrates with Weights & Biases for experiment tracking. 
Training is performed on the specified device with options for distributed computing and verbose logging.
Note: training loss is set to none, this defaults the training to use the L2 loss.
"""

from neuralop.models import FNO2d
from trainer import Trainer
from neuralop import LpLoss
import torch
import wandb
import math
 
def callTraining(dataloaders,epochs,prediction_length):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = 86
    model = FNO2d(n_modes_width = resolution, n_modes_height = resolution, hidden_channels=64, projection_channels=101 , in_channels=4, out_channels=1) #Create the model

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs*12)

    l2loss = LpLoss(d=2, p=2) # L2 loss for the heat equation

    eval_losses={'l2': l2loss} # The losses we want to evaluate
    
    output_encoder = None # No encoder for the output

    wandb.init(project='PredictingTemps', config={'hyper': 'parameter_values'})

    trainer = Trainer(model=model, n_epochs=epochs,
                  device=device,
                  wandb_log=True,
                  log_test_interval=1,
                  log_output=True,
                  use_distributed=False,
                  verbose=True)
    
    trainer.training(dataloaders,resolution,
            output_encoder,
            model, 
            optimizer,
            scheduler, 
            regularizer=False, 
            training_loss=None,
            eval_losses=eval_losses,prediction_length=prediction_length)