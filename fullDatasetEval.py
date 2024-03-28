"""Designed to evaluate the model on the full dataset.""" 
                                                            
import torch
from torch import nn
import dataVisualization as dv
import loadData as ld
import continusDataLoaders as ns
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

input_seq_len=1
prediction_length = 100

#file_path = 'Data/Original/CASE02_DATA.mat'
file_path = 'Data/K/k3.mat'
#file_path = 'Data/SIG/sig3.mat'
#file_path = 'Data/W/w3.mat'
model_path = f'Models/newdata100.pth'

data_variables = ld.loadData(file_path) #Load the data from the file
train_loader, test_loader, combined_loader,timestep = ns.create_datasets(data_variables, input_seq_len=input_seq_len, prediction_len=prediction_length, train_ratio=0.50, batch_size=1)
loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
eval(combined_loader,loaded_model,input_seq_len,5,"tk1tk2sig1sig2out50","k3",prediction_len=prediction_length)

            

