import loadData as ld
import splits as sp
import callTrain as ct

input_seq_len=1
epochs = 3
prediction_length =2
dataset_paths = ['Data/K/k1.mat', 'Data/K/k2.mat', 'Data/K/k4.mat', 'Data/K/k5.mat']
model_path = f'Models/testing.pth'

dataloaders = []
for path in dataset_paths:
    data_variables = ld.loadData(path)  # Load the data from the file
    print(f'Loaded data from {path}')
    train_loader, test_loader, last_timestep = sp.create_datasets(
        data_variables,
        input_seq_len=input_seq_len,
        prediction_len=prediction_length,
        train_ratio=0.10,
        batch_size=1
    )
    dataloaders.append((train_loader, test_loader))
    
ct.callTrainingMultiple(dataloaders,input_seq_len,epochs,model_path, prediction_length)#Train the model (this will save the model as well)