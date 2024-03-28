from neuralop.models import FNO2d
from trainer import Trainer
from neuralop import LpLoss, H1Loss
import torch
import wandb

def callTraining(dataloaders,input_seq_len,epochs,model_path,prediction_length):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #test_loaders = {"default": test_loader}
    model = FNO2d(n_modes_width = 32, n_modes_height = 32, hidden_channels=32, projection_channels=101 , in_channels=input_seq_len, out_channels=1) #Create the model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) #Create the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) #Create the scheduler

    l2loss = LpLoss(d=2, p=2) # L2 loss for the heat equation
    h1loss = H1Loss(d=2) # H1 loss for the heat equation

    train_loss = h1loss # The loss we want to train
    eval_losses={'h1': h1loss, 'l2': l2loss} # The losses we want to evaluate
    
    output_encoder = None # No encoder for the output

    wandb.init(project='PredictingTemps', config={'hyper': 'parameter_values'})

    trainer = Trainer(model=model, n_epochs=epochs,
                  device=device,
                  wandb_log=True,
                  log_test_interval=1,
                  log_output=True,
                  use_distributed=False,
                  verbose=True)
    
    trainer.trainingMultiple(dataloaders,
            output_encoder,
            model, 
            optimizer,
            scheduler, 
            regularizer=False, 
            training_loss=None,
            eval_losses=eval_losses,prediction_length=prediction_length,model_path=model_path)