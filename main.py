"""
main.py
- Main entry point of the script
- This script orchestrates the process of loading data, creating train & test sets, training a model, and evaluating its performance. 
- It utilizes functions and classes defined in external modules (loadData, splits, trainer, modelEval) to perform data handling, model training, and evaluation tasks.

Usage:
    Run this script to train the model with data from a file and evaluate its performance.
"""
import loadData as ld
import splits as sp
import callTrain as ct
import modelEval as me
import viewData as vd
import dataVisualization as dv
import torch
from neuralop.models import FNO2d

input_seq_len=1
epochs = 3
prediction_length = 10

file_path = 'Data/Original/CASE02_DATA.mat'
#file_path = 'Data/K/k3.mat'
#file_path = 'Data/SIG/sig1.mat'
#file_path = 'Data/W/wq.mat'
model_path = f'Models/lowTest10.pth'

data_variables = ld.loadData(file_path) #Load the data from the file
#ld.printData(data_variables) #Print the data for verification and analysis
#dv.createAnimation(data_variables, 'test') #Create an animation of the data from the dataset

train_loader, test_loader,last_timestep = sp.create_datasets(data_variables, input_seq_len=input_seq_len, prediction_len=prediction_length, train_ratio=0.20, batch_size=1) #Create the datasets
#vd.print_test_loader_contents(train_loader) #Print the contents of the test loader for verification and analysis

loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
me.eval(test_loader,loaded_model,input_seq_len,last_timestep,prediction_len=prediction_length) #Evaluate the model"""