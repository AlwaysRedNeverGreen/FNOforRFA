"""
This script evaluates a trained model on a full dataset, specifically designed for time-series predictions where each subsequent input depends on the previous prediction.
It calculates the root mean square error (RMSE) for each timestep, logs errors, and visualizes results comparing predictions with ground truths. 
Parameters specific to the model's use case are incorporated into the input, allowing for detailed assessment of model performance under various conditions. 
Results are recorded and visualizations of predictions versus actual data are generated.
"""

import torch
import dataVisualization as dv
import loadData as ld
import dataReader as dr
import continuousDataLoaders as cdl
import time
import regions as r

def evalParams(device,combined_loader, loaded_model, last_timestep,model_name,case_name,k,w,sig):
    ground_truths = {} # List to store the ground truth temperature distributions
    differences = {} # List to store the differences between the ground truth and the model's predictions
    predictions = {} # Dictionary to store the results for each timestep
    total_loss = [] # Keep track of the loss at each time ste
    rmse_dict = {}
    current_timestep = last_timestep
    
    params_tensor = torch.tensor([k, w, sig]).view(1, 3, 1, 1).expand(-1, -1, 101, 101)
    params_tensor = params_tensor.to(device)
    
    with torch.no_grad():
        for batch in combined_loader:
            start_time = time.time()
            i=0
            x_test, y_true = batch['x'], batch['y']
            
            if i == 0: #Use input from the dataset
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
                current_timestep += 5 

                y_with_params = torch.cat((y_pred, params_tensor), dim=1) # Concatenate the predicted output with the parameters
                y_pred = loaded_model(y_with_params)
                i+=1
         
    end_time = time.time() - start_time
    
    print(f"Time taken to evaluate the model: {end_time}")                            
    print(f"Mean of the error list: {sum(total_loss) / len(total_loss)}")
    dv.createAnimationComparison(ground_truths, predictions, differences, case_name,k,w,sig, model_name)


def evalParamsRegions(device,combined_loader, loaded_model, last_timestep,model_name,case_name,k,w,sig):
    ground_truths = {} # List to store the ground truth temperature distributions
    differences = {} # List to store the differences between the ground truth and the model's predictions
    predictions = {} # Dictionary to store the results for each timestep
    total_loss = [] # Keep track of the loss at each time ste
    rmse_dict = {}
    current_timestep = last_timestep
    kn,wn,sign = k[0],w[0],sig[0]
    kt, wt, sigt = k[1],w[1],sig[1]
    k = r.parameterRegions(k[0],k[1])
    w = r.parameterRegions(w[0],w[1])
    sig = r.parameterRegions(sig[0],sig[1])
    
    params_tensor = torch.stack([k, w, sig], dim=0)
    params_tensor = params_tensor.unsqueeze(0)  # Add batch dimension: [1, 3, 101, 101]
    params_tensor = params_tensor.to(device)
    
    loaded_model.eval()
    with torch.no_grad():
        for batch in combined_loader:
            start_time = time.time()
            i=0
            x_test, y_true = batch['x'], batch['y']
            
            if i == 0: #Use input from the dataset
                x_with_params = torch.cat((x_test, params_tensor), dim=1)
                x_with_params = x_with_params.float()
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
                current_timestep += 5 

                y_with_params = torch.cat((y_pred, params_tensor), dim=1) # Concatenate the predicted output with the parameters
                y_with_params = y_with_params.float()
                y_pred = loaded_model(y_with_params)
                i+=1
         
    end_time = time.time() - start_time
    
    print(f"Time taken to evaluate the model: {end_time}")                            
    print(f"Mean of the error list: {sum(total_loss) / len(total_loss)}")
    dv.createAnimationComparison(ground_truths, predictions, differences, case_name,kn,wn,sign,kt,wt,sigt, model_name)        
input_seq_len=1
prediction_length = 100

file_path = 'Data/DATA 7 MAR/Param k_tissue = 0.53.csv'
case_name = file_path.split('/')[-1].split('.mat')[0] # Get the name of the file, used for saving the results

#data_variables = ld.loadData(file_path) #Load the data from the file
data_variables = dr.CSVReader(file_path)
loader = cdl.create_single_dataset(data_variables)

model_path = f'Models/March Models/Model86res lowTest100 correct param orientation.pth'
model_name = model_path.split('/')[-1].split('.')[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = torch.load(model_path, map_location=torch.device(device))

#k,w,sig = 5.3,12.2,5.7
k = (5.3,5.3)
w = (12.2,4)
sig = (2,4)
evalParamsRegions(device,loader,loaded_model,5, model_name,case_name,k,w,sig)