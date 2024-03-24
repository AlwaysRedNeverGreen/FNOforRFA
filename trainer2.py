import torch
from timeit import default_timer
import wandb
import sys 

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

    def train(self, train_loader, test_loaders, output_encoder,
              model, optimizer, scheduler, regularizer, 
              training_loss=None, eval_losses=None):

        """Trains the given model on the given datasets"""
        n_train = len(train_loader.dataset)

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
        
        for epoch in range(self.n_epochs):
            avg_loss = 0
            avg_lasso_loss = 0
            model.train()
            t1 = default_timer()
            train_err = 0.0

            for idx, sample in enumerate(train_loader):
                x, y = sample['x'], sample['y']
                
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Training on raw inputs of size {x.shape=}, {y.shape=}')

                x, y = self.patcher.patch(x, y)

                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'.. patched inputs of size {x.shape=}, {y.shape=}')

                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()
                print("PRE OUT")
                out = model(x)
                print(f'Epoch: {epoch}, Batch: {idx}, Output size: {out.shape}')
                print(f'Sample Output: {out[:1, :5].detach().cpu().numpy()}')  # Print first row and first 5 elements as an example
                
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Raw outputs of size {out.shape=}')

                out, y = self.patcher.unpatch(out, y)
                #Output encoding only works if output is stiched
                if output_encoder is not None and self.mg_patching_stitching:
                    out = output_encoder.decode(out)
                    y = output_encoder.decode(y)
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'.. Processed (unpatched) outputs of size {out.shape=}')

                loss = training_loss(out.float(), y)
                print(f'Loss: {loss}')

                if regularizer:
                    loss += regularizer.loss

                loss.backward()
                
                optimizer.step()
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1
            del x, y

            train_err/= n_train
            avg_loss /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 
                
                msg = f'[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'

                values_to_log = dict(train_err=train_err, time=epoch_train_time, avg_loss=avg_loss)

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
                    
    def training(self, train_loader, test_loaders, output_encoder,model, optimizer, scheduler, regularizer, training_loss=None, eval_losses=None, prediction_length=1):
        
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

        for epoch in range(self.n_epochs):
            t1 = default_timer() #Start the timer for the epoch
            # Assuming you have a model called `model`
            if torch.cuda.is_available():
                model = model.cuda()  # Moves and returns all model parameters and buffers to the GPU.

            model.train()
            train_err = 0
            degreeVariation = 0.0 #Sum of the RMSE for each time step
            for batch in train_loader:
                optimizer.zero_grad()
                x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                #print("This is ground truth Y:\n ", y)
                loss = 0 
                squarederror = 0

                for t in range(prediction_length):
                    output = model(x)
                    #print("This is the output: \n ", output)

                    loss += training_loss(output, y[:, t]) 
                    squarederror = (output - y[:, t]) ** 2 
                    
                    mse = torch.mean(squarederror) #mean squared error
                    rmse = torch.sqrt(mse) #Calculate the RMSE for the time step, this is the squarederror in original units degrees celsius
                    #sum the squarederror for each time step
                    degreeVariation += rmse.item() #Add the RMSE to the total RMSE for the epoch
                
                    #print("This is what the output is being compared to: \n ", y[:, t])
                    # Prepare x for the next prediction step
                    x = output
                loss.backward()
                optimizer.step()
                train_err += loss.item()
                
            avg_loss = train_err / len(train_loader) #average loss for the epoch
            avg_rmse = degreeVariation / len(train_loader) #average RMSE for the epoch
            print(f'Epoch [{epoch+1}/{self.n_epochs}]: Average Loss: {avg_loss}, Average RMSE: {avg_rmse}')
            wandb.log({"avg_loss": avg_loss, "avg_rmse": avg_rmse}, step = epoch)
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err) #Reduce the learning rate if the loss is not decreasing
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1 
            del x, y #Delete the x and y tensors to free up memory

            avg_loss /= self.n_epochs 
            
            if epoch % self.log_test_interval == 0: 
                msg = f'[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}'
                values_to_log = dict(train_err=train_err, time=epoch_train_time, avg_loss=avg_loss)

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