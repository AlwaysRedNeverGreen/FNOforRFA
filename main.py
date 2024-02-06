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
import torch

input_seq_len=5 
file_path = 'Data/CASE01_DATA.mat'
data_variables = ld.loadData(file_path) #Load the data from the file

train_loader, test_loader = sp.create_datasets(data_variables, input_seq_len=input_seq_len, prediction_len=1, train_ratio=0.5, batch_size=8) #Create the datasets

tr.training(train_loader,test_loader)
loaded_model = torch.load('trained_model1.pth')

me.eval(test_loader,loaded_model,input_seq_len)