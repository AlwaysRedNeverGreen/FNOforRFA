"""Designed to evaluate the model on the full dataset.""" 
                                                            
import torch
from torch import nn
import dataVisualization as dv
import loadData as ld
import continuousDataLoaders as cdl
import viewData as vd
import Graphing as gp

def eval(device,combined_loader, loaded_model, input_seq_len, last_timestep,model_name,case_name, prediction_len=1, interval=5):
    ground_truths = {} # List to store the ground truth temperature distributions
    differences = {} # List to store the differences between the ground truth and the model's predictions
    predictions = {} # Dictionary to store the results for each timestep
    total_loss = [] # Keep track of the loss at each time step
    rmse_dict = {}
    current_timestep = last_timestep

    with torch.no_grad():
        for batch in combined_loader:
            i=0
            x_test, y_true = batch['x'].to(device), batch['y'].to(device)
            if i == 0:
                y_pred = loaded_model(x_test)
                
                predictions[f'Original input T{current_timestep}'] = x_test
                ground_truths[f'Ground Truth T{current_timestep}'] = x_test
                differences[f'Difference T{current_timestep}'] = x_test - x_test
                
                current_timestep +=5
                
            for t in range(y_true.size(1)):
                predictions[f'Prediction T{current_timestep}'] = y_pred.squeeze()
                ground_truths[f'Ground Truth T{current_timestep}'] = y_true[:, t, :, :].squeeze()
                
                loss = (y_true[:, t, :, :] - y_pred) ** 2
                differences[f'Difference T{current_timestep}'] = loss.squeeze()
                
                mse = torch.mean(loss)
                rmse = torch.sqrt(mse)
                rmse_dict[f'T{current_timestep}'] = rmse.item()
                total_loss.append(rmse.item())
                print(f"RMSE at time step {current_timestep}: {rmse.item()}")
                
                current_timestep += 5
                y_pred = loaded_model(y_pred) # Use the model's output as the input for the next timestep
                i+=1
                               
    print(f"Mean of the error list: {sum(total_loss) / len(total_loss)}")
    gp.append_results_to_csv(model_name, rmse_dict, f'eval_results_{case_name}.csv')
    #dv.createAnimationComparison(ground_truths, predictions, differences, case_name, model_name)

def evalParams(device,combined_loader, loaded_model, input_seq_len, last_timestep,model_name,case_name, prediction_len=1, interval=5):
    ground_truths = {} # List to store the ground truth temperature distributions
    differences = {} # List to store the differences between the ground truth and the model's predictions
    predictions = {} # Dictionary to store the results for each timestep
    total_loss = [] # Keep track of the loss at each time ste
    rmse_dict = {}
    current_timestep = last_timestep
    
    k,w,sig = 3,3,2
    params_tensor = torch.tensor([k, w, sig]).view(1, 3, 1, 1).expand(-1, -1, 101, 101)
    params_tensor = params_tensor.to(device)
    
    with torch.no_grad():
        for batch in combined_loader:
            i=0
            x_test, y_true = batch['x'], batch['y']
            if i == 0:
                x_with_params = torch.cat((x_test, params_tensor), dim=1)
                y_pred = loaded_model(x_with_params)
                predictions[f'Original input at T{current_timestep}'] = x_test
                ground_truths[f'Ground Truth at T{current_timestep}'] = x_test
                differences[f'Error at T{current_timestep}'] = x_test - x_test
                current_timestep +=5
            for t in range(y_true.size(1)):
                predictions[f'Prediction at T{current_timestep}'] = y_pred.squeeze()
                ground_truths[f'Ground Truth at T{current_timestep}'] = y_true[:, t, :, :].squeeze()
                
                loss = (y_true[:, t, :, :] - y_pred) ** 2
                differences[f'Error at T{current_timestep}'] = loss.squeeze()
                
                mse = torch.mean(loss)
                rmse = torch.sqrt(mse)
                rmse_dict[f'T{current_timestep}'] = rmse.item()
                total_loss.append(rmse.item())
                print(f"RMSE at time step {current_timestep}: {rmse.item()}")
                current_timestep += 5

                y_with_params = torch.cat((y_pred, params_tensor), dim=1) # Concatenate the predicted output with the parameters
                y_pred = loaded_model(y_with_params)
                i+=1
                               
    print(f"Mean of the error list: {sum(total_loss) / len(total_loss)}")
    gp.append_results_to_csv(model_name, rmse_dict, f'eval_results_{case_name}.csv')
    #dv.createAnimationComparison(ground_truths, predictions, differences, case_name, model_name)
        
input_seq_len=1
prediction_length = 100
#file_path = 'Data/Original/CASE01_DATA.mat'
#file_path = 'Data/K/k3.mat'
file_path = 'Data/SIG/sig4.mat'
#file_path = 'Data/W/w3.mat'
case_name = file_path.split('/')[-1].split('.mat')[0]

#model_path = f'Models/TrainedOn_K1,2,4,5_Params_64Res.pth'
model_path = f'Models/TrainedOn_K1,2,4,5_NoParams_64Res.pth'
model_name = model_path.split('/')[-1].split('.')[0]

data_variables = ld.loadData(file_path) #Load the data from the file
loader = cdl.create_single_dataset(data_variables,1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = torch.load(model_path, map_location=torch.device(device))

#evalParams(device,loader,loaded_model,input_seq_len,5, model_name,case_name,prediction_len=prediction_length)
eval(device,loader,loaded_model,input_seq_len,5,model_name,case_name,prediction_len=prediction_length)
