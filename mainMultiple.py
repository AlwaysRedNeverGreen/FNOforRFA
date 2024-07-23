"""
This script configures and initiates the training process for a machine learning model. 
It specifies dataset paths, loads the data, splits it into training and testing sets, 
and conducts model training over a defined number of epochs. The trained model is then saved to a specified path.
"""

import loadData as ld
import callTrain as ct
import continuousDataLoaders as cdl
import viewData as vd

input_seq_len=1
epochs = 200
prediction_length =100
dataset_paths = ['Data/K/k1.mat']#, 'Data/K/k2.mat','Data/K/k4.mat','Data/K/k5.mat',
                 #'Data/W/w1.mat', 'Data/W/w2.mat','Data/W/w4.mat','Data/W/w5.mat',
                 #'Data/SIG/sig1.mat', 'Data/SIG/sig2.mat','Data/SIG/sig4.mat','Data/SIG/sig5.mat']
model_path = f'Models/testing.pth'

dataloaders = [] # A list to store the train and test loaders for each dataset
for path in dataset_paths:
    data_variables = ld.loadData(path)  # Load the data from the file
    print(f'Loaded data from {path}')
    train_loader, test_loader = cdl.create_datasets(
        data_variables,
        input_seq_len=input_seq_len,
        prediction_len=prediction_length,
        train_ratio=0.70,
        batch_size=1
    )
    dataloaders.append((train_loader, test_loader))
#vd.print_loader_contents(dataloaders) #View the data
ct.callTraining(dataloaders,epochs,prediction_length)#Train the model