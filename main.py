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
import trainer as tr
import modelEval as me
import viewData as vd
import dataVisualization as dv
import trainer2 as tr2
import torch

input_seq_len=1
epochs = 5
prediction_length = 1
file_path = 'Data/CASE02_DATA.mat'
model_path = 'Models/DS2_Multi_10out.pth'

data_variables = ld.loadData(file_path) #Load the data from the file
#dv.createAnimation(data_variables, 'caseW5') #Create an animation of the data from the dataset

train_loader, test_loader,last_timestep = sp.create_datasets(data_variables, input_seq_len=input_seq_len, prediction_len=prediction_length, train_ratio=0.2, batch_size=1) #Create the datasets
tr.training(train_loader,test_loader,input_seq_len,epochs,model_path, prediction_length,last_timestep) #Train the model (this will save the model as well)

loaded_model = torch.load(model_path) #Load the model (this will be used for evaluation)
me.eval(test_loader,loaded_model,input_seq_len,last_timestep,prediction_len=prediction_length) #Evaluate the model