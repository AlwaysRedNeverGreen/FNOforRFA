"""
This script, adapted from the original at https://github.com/neuraloperator/neuraloperator 
implements a recursive training loop for a neural network model. 
The loop uses the predicted output as input for subsequent training iterations, enhancing the model's temporal prediction capabilities. 
Additionally, a custom evaluation function is integrated to assess model performance using this recursive training strategy. 
This script also includes detailed logging and experiment tracking via Weights & Biases.
Note: - The only remaining original code is the declaration of the Trainer class with its various intializing parameters.
      - The training and evaluation functions though have very little resemblance to the original code.
"""

import torch
import wandb
import sys 
import neuralop.mpu.comm as comm
import time
from neuralop.training.patching import MultigridPatching2D
from neuralop.training.losses import LpLoss
import regions as r

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
    def training(self, dataloaders, resolution, output_encoder,model, optimizer, scheduler, regularizer, training_loss=None, eval_losses=None, prediction_length=1, regions = False):
            if regions == True:
                self.training_two_regions(dataloaders, resolution, output_encoder,model, optimizer, scheduler, regularizer, training_loss, eval_losses, prediction_length, regions)

            else:
                self.training_single_region(dataloaders, resolution, output_encoder,model, optimizer, scheduler, regularizer, training_loss, eval_losses, prediction_length)

                            
    def training_single_region(self, dataloaders, resolution, output_encoder,model, optimizer, scheduler, regularizer, training_loss=None, eval_losses=None, prediction_length=1):

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
            lowest_test_rmse = float('inf')
            
            params_map_tissue = {
                0: (1, 2, 2),
                1: (2, 2, 2),
                2: (4, 2, 2),
                3: (5, 2, 2),
                4: (2, 1, 2),
                5: (2, 2, 2),
                6: (2, 4, 2),
                7: (2, 5, 2),
                8: (2, 2, 1),
                9: (2, 2, 2),
                10: (2, 2, 4),
                11: (2, 2, 5),
            }
  

            for epoch in range(self.n_epochs):
                epoch_start_time = time.time()
                
                if torch.cuda.is_available():
                    model = model.cuda() 
                    
                dataset_id=1 #To keep track of which dataset is being trained on
                avg_loss = 0
                avg_rmse = 0
                total_test_l2loss = 0
                total_test_rmse = 0
                tracker = 0
                
                for train_loaders, test_loaders in dataloaders:
                    total_loss = 0
                    total_rmse = 0
                    k, w, sig = params_map_tissue[tracker] #Get the parameters for the dataset
                    print(f'k{k}, w{w}, sig{sig}')
                    tracker += 1
                    params_tensor = torch.tensor([k, w, sig]).view(1, 3, 1, 1).expand(-1, -1, 101, 101) #Create a tensor of the parameters
                    params_tensor = params_tensor.to(self.device)
                    print("param tensor shape", params_tensor.shape)
                    model.train()  # Set model to training mode
                    avg_dataset_loss = 0 #To keep track of the average loss on the dataset
                    avg_dataset_rmse = 0 #To keep track of the average RMSE on the dataset
                    
                    for batch in train_loaders:
                        optimizer.zero_grad()  # Clear existing gradients
                        x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                        for t in range(y.size(1)):
                            x_with_params = torch.cat((x, params_tensor), dim=1) # Concatenate the input with the parameters
                            
                            output = model(x_with_params) # Forward pass
                            loss = training_loss(output, y[:, t]) # Compute the L2loss
                            
                            #Perform backpropagation after each prediction
                            loss.backward()  # Compute gradient of loss w.r.t model parameters
                            optimizer.step()  # Update model parameters
                            optimizer.zero_grad()  # Clear gradients for the next update
                            
                            # Calculate RMSE for logging/monitoring (Gives errors in original units of the data)
                            rmse = torch.sqrt(torch.mean((output - y[:, t]) ** 2))
                            
                            total_loss += loss.item() # Accumulate loss
                            total_rmse += rmse.item() # Accumulate RMSE
                            x = output.detach()  #use the output as input for the next step     
                    
                    scheduler.step() # Step the scheduler after each dataset
                        
                    
                    avg_dataset_loss = total_loss / y.size(1) #Calculate the average loss per prediction
                    avg_loss += avg_dataset_loss
                    avg_dataset_rmse = total_rmse / y.size(1)
                    avg_rmse += avg_dataset_rmse
                    
                    print(f'[Dataset {dataset_id}] Average dataset training Loss: {avg_dataset_loss:.4f}, Average dataset training RMSE: {avg_dataset_rmse:.4f}')
                    
                    if not isinstance(test_loaders, dict): 
                        test_loaders_dict = dict(test=test_loaders) 
                
                    for loader_name, loader in test_loaders_dict.items():
                        errors = self.evaluate(model,k,w,sig, eval_losses, loader, output_encoder, log_prefix=loader_name)
                        print(f'[Dataset {dataset_id} Eval]', end=' ')
                        
                        for loss_name, loss_value in errors.items():
                            print(f'{loss_name}={loss_value:.4f}', end=' ')  # Print each loss on the same line
                            if loss_name == 'test_l2':
                                total_test_l2loss += errors['test_l2']
                            if loss_name == 'test_rmse':
                                total_test_rmse += errors['test_rmse'] 
                        print()  # Move to a new line after printing all losses
                        print()        
                        
                    dataset_id += 1 #Increment the dataset id 
                # Log average loss and RMSE for the epoch
                avg_loss = avg_loss / ntrain_total #Calculate the average loss for the epoch
                avg_rmse = avg_rmse / ntrain_total #Calculate the average RMSE for the epoch
                print(f'Epoch {epoch}: Average Loss of Epoch: {avg_loss:.4f}, Average RMSE of Epoch: {avg_rmse:.4f}')
                
                
                avg_test_l2loss = total_test_l2loss / ntrain_total #Calculate the average test L2 loss for the epoch
                avg_test_rmse = total_test_rmse / ntrain_total #Calculate the average test RMSE for the epoch
                print(f'Avg test L2 Loss: {avg_test_l2loss:.4f}, Avg test RMSE: {avg_test_rmse:.4f}\n')
                wandb.log({"avg_epoch_loss": avg_loss, "avg_epoch_rmse": avg_rmse, "avg_test_l2loss": avg_test_l2loss, "avg_test_rmse": avg_test_rmse}, step = epoch)
                     
                if avg_loss < lowest_loss:
                    lowest_loss = avg_loss
                    torch.save(model, f'Models/model{resolution}res lowTrain{prediction_length}.pth')
                    print(f'{resolution} res Model with lowest training loss saved')

                if avg_test_rmse < lowest_test_rmse:
                    lowest_test_rmse = avg_test_rmse
                    save_path = f'Models/model{resolution}res lowTest{prediction_length}.pth'
                    torch.save(model, save_path)
                    print(f'Model with lowest test_rmse loss saved to {save_path}\n')

                for pg in optimizer.param_groups:
                    lr = pg['lr']
                    wandb.log({"lr": lr}, step = epoch)
                print("-" * 100)  # Separator    
                

                epoch_train_time = time.time()-epoch_start_time
                wandb.log({"epoch_train_time": epoch_train_time}, step = epoch)
                del x, y #Delete the x and y tensors to free up memory              
    
    def training_two_regions(self, dataloaders, resolution, output_encoder,model, optimizer, scheduler, regularizer, training_loss=None, eval_losses=None, prediction_length=1, regions = False):
            ntrain_total = 0
            ntest_total = 0
            for train_loaders,test_loaders in dataloaders:
                n_train = len(train_loaders.dataset)
                ntrain_total += n_train
                n_test = len(test_loaders.dataset)
                ntest_total += n_test
            
            if self.verbose:
                print(f'Training on {ntrain_total} samples')
                print(f'Testing on {ntest_total} samples\n')
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
            lowest_test_rmse = float('inf')
            
            params_map = {
                0: (3.3, 5.3),
                1: (4.3, 5.3),
                2: (6.3, 5.3),
                3: (7.3, 5.3),
                4: (5.3, 3.3),
                5: (5.3, 4.3),
                6: (5.3, 6.3),
                7: (5.3, 7.3),
                8: (3.1, 12.2),
                9: (6.1, 12.2),
                10: (24.5, 12.2),
                11: (49.1, 12.2),
                12: (12.2, 1),
                13: (12.2, 8),
                14: (12.2, 16),
                }

            w = r.parameterRegions(12.2, 4) #Initializing for first 8 datasets
            sig = r.parameterRegions(2,5.7) #This is the same for all datasets
            
            model = model.float()
            if torch.cuda.is_available():
                    model = model.cuda() 
                    
            for epoch in range(self.n_epochs):
                epoch_start_time = time.time()
                
                dataset_id=1 #To keep track of which dataset is being trained on
                avg_loss = 0
                avg_rmse = 0
                total_test_l2loss = 0
                total_test_rmse = 0
                tracker = 0
                
                for train_loaders, test_loaders in dataloaders:
                    total_loss = 0
                    total_rmse = 0
                    
                    if tracker < 8:
                        k = r.parameterRegions(params_map[tracker][0], params_map[tracker][1])
                        print(f'Parameter Values: k{params_map[tracker]}, w: (12.2,4) sig:(2,5.7)')
                    if tracker >= 8:
                        k = r.parameterRegions(5.3,5.3)
                        w = r.parameterRegions(params_map[tracker][0], params_map[tracker][1])
                        print(f'Parameter Values: k(5.3,5.3), w{params_map[tracker]}, sig:(2,5.7)')
                        
                    #print(f'Parameter Values: k{k}, w{w}, sig{sig}')
                    
                    tracker += 1
                    params_tensor = torch.stack([k, w, sig], dim=0)
                    params_tensor = params_tensor.unsqueeze(0)  # Add batch dimension: [1, 3, 101, 101]
                    params_tensor = params_tensor.to(self.device)
                    
                    #print("Param tensor shape\n", params_tensor.shape)
                    model.train()  # Set model to training mode
                    avg_dataset_loss = 0 #To keep track of the average loss on the dataset
                    avg_dataset_rmse = 0 #To keep track of the average RMSE on the dataset
                    
                    for batch in train_loaders:
                        optimizer.zero_grad()  # Clear existing gradients
                        x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                        for t in range(y.size(1)):
                            x_with_params = torch.cat((x, params_tensor), dim=1) # Concatenate the input with the parameters
                            x_with_params = x_with_params.float()  # Convert to float

                            output = model(x_with_params) # Forward pass
                            loss = training_loss(output, y[:, t]) # Compute the L2loss
                            
                            #Perform backpropagation after each prediction
                            loss.backward()  # Compute gradient of loss w.r.t model parameters
                            optimizer.step()  # Update model parameters
                            optimizer.zero_grad()  # Clear gradients for the next update
                            
                            # Calculate RMSE for logging/monitoring (Gives errors in original units of the data)
                            rmse = torch.sqrt(torch.mean((output - y[:, t]) ** 2))
                            
                            total_loss += loss.item() # Accumulate loss
                            total_rmse += rmse.item() # Accumulate RMSE
                            x = output.detach()  #use the output as input for the next step     
                    
                    scheduler.step() # Step the scheduler after each dataset
                        
                    
                    avg_dataset_loss = total_loss / y.size(1) #Calculate the average loss per prediction
                    avg_loss += avg_dataset_loss
                    avg_dataset_rmse = total_rmse / y.size(1)
                    avg_rmse += avg_dataset_rmse
                    
                    print(f'[Dataset {dataset_id}] Average dataset training Loss: {avg_dataset_loss:.4f}, Average dataset training RMSE: {avg_dataset_rmse:.4f}')
                    
                    if not isinstance(test_loaders, dict): 
                        test_loaders_dict = dict(test=test_loaders) 
                
                    for loader_name, loader in test_loaders_dict.items():
                        errors = self.evaluateRegions(model,params_tensor, eval_losses, loader, output_encoder, log_prefix=loader_name)
                        print(f'[Dataset {dataset_id} Eval]', end=' ')
                        
                        for loss_name, loss_value in errors.items():
                            print(f'{loss_name}={loss_value:.4f}', end=' ')  # Print each loss on the same line
                            if loss_name == 'test_l2':
                                total_test_l2loss += errors['test_l2']
                            if loss_name == 'test_rmse':
                                total_test_rmse += errors['test_rmse'] 
                        print()  # Move to a new line after printing all losses
                        print()        
                        
                    dataset_id += 1 #Increment the dataset id 
                # Log average loss and RMSE for the epoch
                avg_loss = avg_loss / ntrain_total #Calculate the average loss for the epoch
                avg_rmse = avg_rmse / ntrain_total #Calculate the average RMSE for the epoch
                print(f'Epoch {epoch}: Average Loss of Epoch: {avg_loss:.4f}, Average RMSE of Epoch: {avg_rmse:.4f}')
                
                
                avg_test_l2loss = total_test_l2loss / ntrain_total #Calculate the average test L2 loss for the epoch
                avg_test_rmse = total_test_rmse / ntrain_total #Calculate the average test RMSE for the epoch
                print(f'Avg test L2 Loss: {avg_test_l2loss:.4f}, Avg test RMSE: {avg_test_rmse:.4f}\n')
                wandb.log({"avg_epoch_loss": avg_loss, "avg_epoch_rmse": avg_rmse, "avg_test_l2loss": avg_test_l2loss, "avg_test_rmse": avg_test_rmse}, step = epoch)
                     
                if avg_loss < lowest_loss:
                    lowest_loss = avg_loss
                    torch.save(model, f'Models/model{resolution}res lowTrain{prediction_length}.pth')
                    print(f'{resolution} res Model with lowest training loss saved')

                if avg_test_rmse < lowest_test_rmse:
                    lowest_test_rmse = avg_test_rmse
                    save_path = f'Models/model{resolution}res lowTest{prediction_length}.pth'
                    torch.save(model, save_path)
                    print(f'Model with lowest test_rmse loss saved to {save_path}\n')

                for pg in optimizer.param_groups:
                    lr = pg['lr']
                    wandb.log({"lr": lr}, step = epoch)
                print("-" * 100)  # Separator    
                

                epoch_train_time = time.time()-epoch_start_time
                wandb.log({"epoch_train_time": epoch_train_time}, step = epoch)
                del x, y #Delete the x and y tensors to free up memory
    
    def evaluate(self, model,k,w,sig, loss_dict, data_loader, output_encoder=None, log_prefix=''):
        """Evaluate the model on a dictionary of losses."""
        model.eval()
        is_logger = not self.use_distributed or comm.get_world_rank() == 0
        errors = {f'{log_prefix}_{loss_name}': 0 for loss_name in loss_dict.keys()}
        n_samples = 0
        rmse_loss = 0
        params_tensor = torch.tensor([k, w, sig]).view(1, 3, 1, 1).expand(-1, -1, 101, 101)
        params_tensor = params_tensor.to(self.device)
        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample['x'], sample['y']
                n_samples += x.size(0)
                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                current_input = x.to(self.device)  # Initialize current_input with the initial input

                for t in range(y.size(1)):  # Assume y has shape [batch, timesteps, ...] and iterate over timesteps
                    current_with_params = torch.cat((current_input, params_tensor), dim=1)
                    out = model(current_with_params)
                    current_input = out.detach()  # Use the output as the input for the next timestep
                    
                    # Now calculate the loss for this step
                    for loss_name, loss in loss_dict.items():
                        if y.size(1) > t:  # Make sure y has this timestep
                            loss_value = loss(out, y[:, t])
                            errors[f'{log_prefix}_{loss_name}'] += loss_value.mean().item()
                        else:
                            print(f"Error: y does not have timestep {t}")
                            
                    rmse_loss += torch.sqrt(torch.mean((out - y[:, t]) ** 2)).item()
                    
                # After processing all timesteps, divide the accumulated losses by the number of timesteps to get the average
                for loss_name in errors.keys():
                    errors[loss_name] /= y.size(1)
            errors[f'{log_prefix}_rmse'] = (rmse_loss / y.size(1))
        return errors
    
    def evaluateRegions(self, model,params_tensor, loss_dict, data_loader, output_encoder=None, log_prefix=''):
        """Evaluate the model on a dictionary of losses."""
        model.eval()
        is_logger = not self.use_distributed or comm.get_world_rank() == 0
        errors = {f'{log_prefix}_{loss_name}': 0 for loss_name in loss_dict.keys()}
        n_samples = 0
        rmse_loss = 0
        
        params_tensor = params_tensor.to(self.device)
        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample['x'], sample['y']
                n_samples += x.size(0)
                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                current_input = x.to(self.device)  # Initialize current_input with the initial input

                for t in range(y.size(1)):  # Assume y has shape [batch, timesteps, ...] and iterate over timesteps
                    current_with_params = torch.cat((current_input, params_tensor), dim=1)
                    current_with_params = current_with_params.float()

                    out = model(current_with_params)
                    current_input = out.detach()  # Use the output as the input for the next timestep
                    
                    # Now calculate the loss for this step
                    for loss_name, loss in loss_dict.items():
                        if y.size(1) > t:  # Make sure y has this timestep
                            loss_value = loss(out, y[:, t])
                            errors[f'{log_prefix}_{loss_name}'] += loss_value.mean().item()
                        else:
                            print(f"Error: y does not have timestep {t}")
                            
                    rmse_loss += torch.sqrt(torch.mean((out - y[:, t]) ** 2)).item()
                    
                # After processing all timesteps, divide the accumulated losses by the number of timesteps to get the average
                for loss_name in errors.keys():
                    errors[loss_name] /= y.size(1)
            errors[f'{log_prefix}_rmse'] = (rmse_loss / y.size(1))
        return errors