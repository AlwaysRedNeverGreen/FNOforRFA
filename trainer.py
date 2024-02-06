from neuralop.models import FNO2d
from neuralop import Trainer
#from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
import torch
import wandb
import loadData as ld
import dataVisualization as dv
import splits as sp

epochs = 5 # The number of epochs to train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
def training(train_loader, test_loader):
    test_loaders = {"default": test_loader}
    model = FNO2d(n_modes_width = 16, n_modes_height = 16, hidden_channels=32, projection_channels=64 , in_channels=5, out_channels=1) #Create the model
    #n_params = count_params(model) #Count the number of parameters in the model

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4) #Create the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) #Create the scheduler

    l2loss = LpLoss(d=2, p=2) # L2 loss for the heat equation
    h1loss = H1Loss(d=2) # H1 loss for the heat equation

    train_loss = h1loss # The loss we want to train
    eval_losses={'h1': h1loss, 'l2': l2loss} # The losses we want to evaluate

    output_encoder = None # No encoder for the output

    wandb.init(project='FNO for Dataset 1', config={'hyper': 'parameter_values'})

    trainer = Trainer(model=model, n_epochs=epochs,
                  device=device,
                  wandb_log=True,
                  log_test_interval=1,
                  log_output=True,
                  use_distributed=False,
                  verbose=True)

    trainer.train(train_loader,
              test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)

    # Save the entire model (including architecture and trained parameters)
    torch.save(model, 'trained_model1.pth')
