import dataReader as dr
import continuousDataLoaders as cdl
import viewData as vd
import callTrain as ct

file_paths = ['Data/DATA 7 MAR/Param k_tissue = 0.33.csv'
'Data/DATA 7 MAR/Param k_tissue = 0.43.csv',
'Data/DATA 7 MAR/Param k_tissue = 0.53.csv',
'Data/DATA 7 MAR/Param k_tissue = 0.63.csv',
'Data/DATA 7 MAR/Param k_tissue = 0.73.csv',
'Data/DATA 7 MAR/Param k_tumour = 0.33.csv',
'Data/DATA 7 MAR/Param k_tumour = 0.43.csv',
'Data/DATA 7 MAR/Param k_tumour = 0.63.csv',
'Data/DATA 7 MAR/Param k_tumour = 0.73.csv',
'Data/DATA 7 MAR/Param w_tissue = 0.001625.csv',
'Data/DATA 7 MAR/Param w_tissue = 0.003250.csv',
'Data/DATA 7 MAR/Param w_tissue = 0.013000.csv'
'Data/DATA 7 MAR/Param w_tissue = 0.026000.csv'
'Data/DATA 7 MAR/Param w_tumour = 0.00053.csv'
'Data/DATA 7 MAR/Param w_tumour = 0.00424.csv'
'Data/DATA 7 MAR/Param w_tumour = 0.00848.csv']
input_seq_len=1
epochs = 200
prediction_length =100
dataloaders = [] # A list to store the train and test loaders for each dataset
i = 0
print("Loading data from files...")

for path in file_paths:
    data_variables = dr.CSVReader(path)  # Load the data from the file
    print("-" * 100)  # Separator 
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