import torch
from timeit import default_timer
import wandb
import sys 
import copy
import neuralop.mpu.comm as comm

from neuralop.training.patching import MultigridPatching2D
from neuralop.training.losses import LpLoss


class Trainer:
    def __init__(self, model, n_epochs, wandb_log=True, device=None,
                 mg_patching_levels=0, mg_patching_padding=0, mg_patching_stitching=True,
                 log_test_interval=1, log_output=False, use_distributed=False, verbose=True):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        mg_patching_levels : int, default is 0
            if 0, no multi-grid domain decomposition is used
            if > 0, indicates the number of levels to use
        mg_patching_padding : float, default is 0
            value between 0 and 1, indicates the fraction of size to use as padding on each side
            e.g. for an image of size 64, padding=0.25 will use 16 pixels of padding on each side
        mg_patching_stitching : bool, default is True
            if False, the patches are not stitched back together and the loss is instead computed per patch
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is True
        """
        self.n_epochs = n_epochs
        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.mg_patching_levels = mg_patching_levels
        self.mg_patching_stitching = mg_patching_stitching
        self.use_distributed = use_distributed
        self.device = device

        if mg_patching_levels > 0:
            self.mg_n_patches = 2**mg_patching_levels
            if verbose:
                print(f'Training on {self.mg_n_patches**2} multi-grid patches.')
                sys.stdout.flush()
        else:
            self.mg_n_patches = 1
            mg_patching_padding = 0
            if verbose:
                print(f'Training on regular inputs (no multi-grid patching).')
                sys.stdout.flush()

        self.mg_patching_padding = mg_patching_padding
        self.patcher = MultigridPatching2D(model, levels=mg_patching_levels, padding_fraction=mg_patching_padding,
                                           use_distributed=use_distributed, stitching=mg_patching_stitching)
                    
    def training(self, train_loader, test_loaders, output_encoder,model, optimizer, scheduler, regularizer, training_loss=None, eval_losses=None, prediction_length=1, model_path=None):
        
        n_train = len(train_loader.dataset)
    
        # If test_loaders is not a dictionary, convert it to a dictionary
        if not isinstance(test_loaders, dict): 
            test_loaders = dict(test=test_loaders) 

        if self.verbose:
            print(f'Training on {n_train} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        if output_encoder is not None:
            output_encoder.to(self.device)
        
        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 

        lowest_loss = float('inf')
        best_model_params = None
        for epoch in range(self.n_epochs):
            t1 = default_timer()
            if torch.cuda.is_available():
                model = model.cuda() 
            model.train()  # Set model to training mode
            total_loss = 0
            total_rmse = 0
            
            for batch in train_loader:
                optimizer.zero_grad()  # Clear existing gradients
                x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                
                for t in range(prediction_length):
                    output = model(x)
                    loss = training_loss(output, y[:, t])
                    
                    #Perform backpropagation after each prediction
                    loss.backward()  # Compute gradient of loss w.r.t model parameters
                    optimizer.step()  # Update model parameters
                    optimizer.zero_grad()  # Clear gradients for the next update
                    
                    # Calculate RMSE for logging/monitoring (Gives errors in original units of the data)
                    rmse = torch.sqrt(torch.mean((output - y[:, t]) ** 2))
                    
                    total_loss += loss.item() # Accumulate loss
                    total_rmse += rmse.item() # Accumulate RMSE
                    
                    # Prepare x for the next prediction step if needed
                    x = output.detach()  # Optional: use the output as input for the next step

            # Log average loss and RMSE for the epoch
            avg_loss = total_loss / (len(train_loader) * prediction_length) #len(train_loader)*prediction_length gives the total number of predictions made in the epoch
            avg_rmse = total_rmse / (len(train_loader) * prediction_length)
            print(f'Epoch {epoch+1}: Average Loss of Epoch: {avg_loss:.4f}, Average RMSE of Epoch: {avg_rmse:.4f}')
            wandb.log({"avg_loss": avg_loss, "avg_rmse": avg_rmse}, step = epoch)
            
            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
                best_model_params = copy.deepcopy(model.state_dict())
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss) #Reduce the learning rate if the loss is not decreasing
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1 
            del x, y #Delete the x and y tensors to free up memory

            avg_loss /= self.n_epochs 
            
            if epoch % self.log_test_interval == 0: 
                msg = f'[{epoch+1}] time={epoch_train_time:.2f}, avg loss over all epochs={avg_loss:.4f}'
                values_to_log = dict(train_err=total_loss, time=epoch_train_time, avg_loss=avg_loss)

                for loader_name, loader in test_loaders.items():
                    if epoch == self.n_epochs - 1 and self.log_output:
                        to_log_output = True
                    else:
                        to_log_output = False

                    errors = self.evaluate(model, eval_losses, loader, output_encoder, log_prefix=loader_name)

                    for loss_name, loss_value in errors.items():
                        msg += f', {loss_name}={loss_value:.4f}'
                        values_to_log[loss_name] = loss_value

                if regularizer:
                    avg_lasso_loss /= self.n_epochs
                    msg += f', avg_lasso={avg_lasso_loss:.5f}'

                if self.verbose and is_logger:
                    print(msg)
                    sys.stdout.flush()

                # Wandb loging
                if self.wandb_log and is_logger:
                    for pg in optimizer.param_groups:
                        lr = pg['lr']
                        values_to_log['lr'] = lr
                    wandb.log(values_to_log, step=epoch, commit=True)
        if best_model_params is not None:
            # Specify your model's save path
            save_path = model_path
            torch.save(best_model_params, save_path)
            print(f'Model with lowest loss saved to {save_path}')

    def evaluate(self, model, loss_dict, data_loader, output_encoder=None, log_prefix=''):
        """Evaluate the model on a dictionary of losses."""
        model.eval()
        is_logger = not self.use_distributed or comm.get_world_rank() == 0
        errors = {f'{log_prefix}_{loss_name}': 0 for loss_name in loss_dict.keys()}
        n_samples = 0

        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample['x'], sample['y']
                n_samples += x.size(0)
                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                x = x.to(self.device)
                out = model(x)
                out, y = self.patcher.unpatch(out, y, evaluation=True)
                if output_encoder is not None:
                    out = output_encoder.decode(out)
                    
                if it == 0 and self.log_output and self.wandb_log and is_logger:
                    img = out if out.ndim == 2 else out.squeeze()[0]
                    wandb.log({f'image_{log_prefix}': wandb.Image(img.unsqueeze(-1).cpu().numpy())}, commit=False)
                    
                for loss_name, loss in loss_dict.items():
                    if out.dim() > 2 and out.size(1) > 1:
                        for i in range(out.size(1)):
                            loss_value = loss(out[:, i], y[:, i])
                            errors[f'{log_prefix}_{loss_name}'] += loss_value.mean().item()  # Ensure it is a scalar
                    else:
                        loss_value = loss(out, y)
                        errors[f'{log_prefix}_{loss_name}'] += loss_value.mean().item()  # Ensure it is a scalar
                        
        for key in errors.keys():
            errors[key] /= n_samples
        return errors