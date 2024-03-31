"""Designed to evaluate the model on the full dataset.""" 
                                                            
import torch
from torch import nn
import dataVisualization as dv
import loadData as ld
import continuousDataLoaders as cdl
import viewData as vd

def eval(combined_loader, loaded_model, input_seq_len, last_timestep,model_name,data_name, prediction_len=1, interval=5):
    predictions = [] # List to store the model's predictions
    ground_truths = [] # List to store the ground truth temperature distributions
    timesteps = [] # List to store the timesteps for each prediction
    results_dict = {} # Dictionary to store the results for each timestep
    total_loss = [] # Keep track of the loss at each time step
    
    current_timestep = last_timestep

    with torch.no_grad():
        for batch in combined_loader:
            i=0
            x_test, y_true = batch['x'], batch['y']
            if i == 0:
                y_pred = loaded_model(x_test)
                print(f'xtest {current_timestep}')# \n {x_test}')
                results_dict[f'Original input T{current_timestep}'] = x_test
                current_timestep +=5
            for t in range(y_true.size(1)):
                #print("ypred\n:",y_pred)
                #print("ytrue\n:",y_true[:, t, :, :])
                results_dict[f'Prediction T{current_timestep}'] = y_pred
                loss = (y_true[:, t, :, :] - y_pred) ** 2
                
                mse = torch.mean(loss)
                rmse = torch.sqrt(mse)
                total_loss.append(rmse.item())
                print(f"RMSE at time step {current_timestep}: {rmse.item()}")
                current_timestep += 5
                #print("input ypred\n:",y_pred)
                y_pred = loaded_model(y_pred)
                #print("output ypred\n:",y_pred)
                i+=1
                               
    print(f"Mean of the error list: {sum(total_loss) / len(total_loss)}")        
    #print("results_dict",results_dict)
    dv.createAnimation(results_dict, f'{data_name}',model_name) 

def evalParams(combined_loader, loaded_model, input_seq_len, last_timestep,model_name,data_name, prediction_len=1, interval=5):
    predictions = [] # List to store the model's predictions
    ground_truths = [] # List to store the ground truth temperature distributions
    timesteps = [] # List to store the timesteps for each prediction
    results_dict = {} # Dictionary to store the results for each timestep
    total_loss = [] # Keep track of the loss at each time step
    k,w,sig = 3,3,5
    current_timestep = last_timestep
    params_tensor = torch.tensor([k, w, sig]).view(1, 3, 1, 1).expand(-1, -1, 101, 101)
    params_tensor = params_tensor.to('cpu')
    
    with torch.no_grad():
        for batch in combined_loader:
            i=0
            x_test, y_true = batch['x'], batch['y']
            if i == 0:
                x_with_params = torch.cat((x_test, params_tensor), dim=1)
                y_pred = loaded_model(x_with_params)
                results_dict[f'Original input T{current_timestep}'] = x_test
                
                current_timestep +=5
            for t in range(y_true.size(1)):
                #print("ypred\n:",y_pred)
                #print("ytrue\n:",y_true[:, t, :, :])
                results_dict[f'Prediction T{current_timestep}'] = y_pred
                loss = (y_true[:, t, :, :] - y_pred) ** 2
                
                mse = torch.mean(loss)
                rmse = torch.sqrt(mse)
                total_loss.append(rmse.item())
                print(f"RMSE at time step {current_timestep}: {rmse.item()}")
                current_timestep += 5

                y_with_params = torch.cat((y_pred, params_tensor), dim=1) # Concatenate the predicted output with the parameters
                y_pred = loaded_model(y_with_params)
                i+=1
                               
    print(f"Mean of the error list: {sum(total_loss) / len(total_loss)}")        
    #print("results_dict",results_dict)
    dv.createAnimation(results_dict, f'{data_name}',model_name) 

input_seq_len=1
prediction_length = 100

#file_path = 'Data/Original/CASE01_DATA.mat'
#file_path = 'Data/K/k3.mat'
file_path = 'Data/SIG/sig5.mat'
#file_path = 'Data/W/w3.mat'
model_path = f'Models/k1245params_64.pth'

data_variables = ld.loadData(file_path) #Load the data from the file
loader = cdl.create_single_dataset(data_variables,1)
#vd.print_test_loader_contents(loader)
loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
evalParams(loader,loaded_model,input_seq_len,5,"k1245params_64 ","w3",prediction_len=prediction_length)
#eval(loader,loaded_model,input_seq_len,5,"lowTest100k1245noparams","w3",prediction_len=prediction_length)