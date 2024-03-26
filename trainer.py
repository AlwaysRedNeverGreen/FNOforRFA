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
            avg_loss = total_loss / (len(train_loader)) #len(train_loader)*prediction_length gives the total number of predictions made in the epoch
            avg_rmse = total_rmse / (len(train_loader))
            print(f'Epoch {epoch+1}: Average Loss of Epoch: {avg_loss:.4f}, Average RMSE of Epoch: {avg_rmse:.4f}')
            wandb.log({"avg_loss": avg_loss, "avg_rmse": avg_rmse}, step = epoch)
            
            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
                best_model_params = copy.deepcopy(model.state_dict())
                save_path = model_path
                torch.save(best_model_params, save_path)
                print(f'Model with lowest loss saved to {save_path}')
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss) #Reduce the learning rate if the loss is not decreasing
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1 
            del x, y #Delete the x and y tensors to free up memory
            
            if epoch % self.log_test_interval == 0: 
                msg = f'[{epoch+1}] time={epoch_train_time:.2f}'
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

    def trainingMultiple(self, dataloaders, output_encoder,model, optimizer, scheduler, regularizer, training_loss=None, eval_losses=None, prediction_length=1, model_path=None):
            ntrain_total = 0
            ntest_total = 0
            for train_loaders,test_loaders in dataloaders:
                n_train = len(train_loaders.dataset)
                ntrain_total += n_train
                n_test = len(test_loaders.dataset)
                ntest_total += n_test
            
            if self.verbose:
                print(f'Training on {ntrain_total} samples')
                print(f'Testing on {ntest_total} samples')
                sys.stdout.flush()
            print("-" * 100)
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
            lowest_test_rmse = float('inf')
            for epoch in range(self.n_epochs):
                t1 = default_timer()
                
                if torch.cuda.is_available():
                    model = model.cuda() 
                    
                total_loss = 0
                total_rmse = 0
                dataset_id=0 #To keep track of which dataset is being trained on
                avg_loss = 0
                avg_rmse = 0
                total_test_h1loss = 0
                total_test_l2loss = 0
                total_test_rmse = 0
                
                for train_loaders, test_loaders in dataloaders:
                    model.train()  # Set model to training mode

                    avg_dataset_loss = 0 #To keep track of the average loss on the dataset
                    avg_dataset_rmse = 0 #To keep track of the average RMSE on the dataset

                    for batch in train_loaders:
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
                            
                            x = output.detach()  #use the output as input for the next step      
                    
                    avg_dataset_loss = total_loss / ntrain_total
                    avg_loss += avg_dataset_loss
                    #print(f'avg_dataset_loss: {avg_dataset_loss:.4f} added to avg_loss: {avg_loss:.4f}')
                    avg_dataset_rmse = total_rmse / ntrain_total
                    avg_rmse += avg_dataset_rmse
                    #print(f'avg_dataset_rmse: {avg_dataset_rmse:.4f} added to avg_rmse: {avg_rmse:.4f}')
                    
                    print(f'[Dataset {dataset_id}] Average dataset training Loss: {avg_dataset_loss:.4f}, Average dataset training RMSE: {avg_dataset_rmse:.4f}')
                    
                    if not isinstance(test_loaders, dict): 
                        test_loaders_dict = dict(test=test_loaders) 
                
                    for loader_name, loader in test_loaders_dict.items():
                        errors = self.evaluate(model, eval_losses, loader, output_encoder, log_prefix=loader_name)
                        print(f'[Dataset {dataset_id} Eval]', end=' ')
                        
                        for loss_name, loss_value in errors.items():
                            print(f'{loss_name}={loss_value:.4f}', end=' ')  # Print each loss on the same line
                            if loss_name == 'test_h1':
                                total_test_h1loss += errors['test_h1']
                            if loss_name == 'test_l2':
                                total_test_l2loss += errors['test_l2']
                            if loss_name == 'test_rmse':
                                total_test_rmse += errors['test_rmse'] 
                        print()  # Move to a new line after printing all losses
                        print()        
                         
                    dataset_id += 1 #Increment the dataset id
                x    
                # Log average loss and RMSE for the epoch
                #print(f'Epoch {epoch}: T Loss of Epoch: {total_loss:.4f}, T RMSE of Epoch: {total_rmse:.4f}')
                avg_loss = avg_loss / dataset_id
                avg_rmse = avg_rmse / dataset_id
                  # Separator
                print(f'Epoch {epoch}: Average Loss of Epoch: {avg_loss:.4f}, Average RMSE of Epoch: {avg_rmse:.4f}')
                
                avg_test_h1loss = total_test_h1loss / dataset_id
                avg_test_l2loss = total_test_l2loss / dataset_id
                avg_test_rmse = total_test_rmse / dataset_id
                print(f'Avg test H1 Loss: {avg_test_h1loss:.4f}, Avg test LP Loss: {avg_test_l2loss:.4f}, Avg test RMSE: {avg_test_rmse:.4f}\n')
                wandb.log({"avg_epoch_loss": avg_loss, "avg_epoch_rmse": avg_rmse, "avg_test_h1loss": avg_test_h1loss, "avg_test_l2loss": avg_test_l2loss, "avg_test_rmse": avg_test_rmse}, step = epoch)
                     
                if avg_loss < lowest_loss:
                    lowest_loss = avg_loss
                    torch.save(model, model_path)
                    print(f'Model with lowest training loss saved to {model_path}')

                if avg_test_rmse < lowest_test_rmse:
                    lowest_test_rmse = avg_test_rmse
                    save_path = f'Models/lowTest{prediction_length}.pth'
                    torch.save(model, save_path)
                    print(f'Model with lowest test_rmse loss saved to {save_path}\n')
                for pg in optimizer.param_groups:
                    lr = pg['lr']
                    wandb.log({"lr": lr}, step = epoch)
                print("-" * 100)  # Separator    
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(total_loss) #Reduce the learning rate if the loss is not decreasing
                else:
                    scheduler.step()

                epoch_train_time = default_timer() - t1
                wandb.log({"epoch_train_time": epoch_train_time}, step = epoch)
                del x, y #Delete the x and y tensors to free up memory
                        
    def evaluate(self, model, loss_dict, data_loader, output_encoder=None, log_prefix=''):
        """Evaluate the model on a dictionary of losses."""
        model.eval()
        is_logger = not self.use_distributed or comm.get_world_rank() == 0
        errors = {f'{log_prefix}_{loss_name}': 0 for loss_name in loss_dict.keys()}
        n_samples = 0
        rmse_loss = 0
        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample['x'], sample['y']
                n_samples += x.size(0)
                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                current_input = x.to(self.device)  # Initialize current_input with the initial input
                for t in range(y.size(1)):  # Assume y has shape [batch, timesteps, ...] and iterate over timesteps
                    #print("this is current input",current_input)
                    out = model(current_input)
                    current_input = out.detach()  # Use the output as the input for the next timestep
                    
                    # Now calculate the loss for this step
                    for loss_name, loss in loss_dict.items():
                        if y.size(1) > t:  # Make sure y has this timestep
                            #print("loss name",loss_name)
                            #print("this is out",out)
                            #print("tjos os y",y[:, t])
                            loss_value = loss(out, y[:, t])
                            errors[f'{log_prefix}_{loss_name}'] += loss_value.mean().item()
                        else:
                            print(f"Error: y does not have timestep {t}")
                            
                    rmse_loss += torch.sqrt(torch.mean((out - y[:, t]) ** 2)).item()
                    
                # After processing all timesteps, divide the accumulated losses by the number of timesteps to get the average
                for loss_name in errors.keys():
                    errors[loss_name] /= y.size(1)
                    
            errors[f'{log_prefix}_rmse'] = rmse_loss / (n_samples * y.size(1))
        return errors
