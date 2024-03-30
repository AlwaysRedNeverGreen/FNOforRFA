"""
This script creates datasets without repeating sequences, one continous flow of input and output batches
- CustomDataset: A subclass of torch.utils.data.Dataset that stores the input and output sequences for model training and evaluation. It overrides the __len__ and __getitem__ methods to allow iteration over the dataset.

Functions:
- create_datasets(variables, input_seq_len, prediction_len, train_ratio, batch_size): Processes the raw data from a dictionary of variables into structured input-output pairs, splits them into training and testing sets based on a specified ratio, and wraps them in DataLoader instances for batch processing during model training and evaluation.
- lastTimeStep(ntrain, input_seq_len, sorted_keys): Helper function to determine the last timestep in the training set, which is used for visualization in modelEval.py.
Dependencies:
- torch: The main library used for creating tensor objects from the raw data and for defining the dataset and dataloader functionalities.
- DataLoader, Dataset from torch.utils.data: Utilized for batching and iterating over the datasets during model training and evaluation.

Usage:
This module is intended to be used in scenarios where sequential data (e.g., time series) needs to be fed into a neural network model, facilitating the process of preparing the data for model training and evaluation.
"""

from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch

class CustomDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {'x': self.inputs[idx], 'y': self.outputs[idx]}
        return sample

def create_datasets(variables, input_seq_len, prediction_len, train_ratio, batch_size):
    sorted_keys = sorted(variables.keys())  # Ensure keys are sorted for sequential processing
    tensors = [torch.tensor(variables[key].copy(), dtype=torch.float32) for key in sorted_keys]
    input_sequences = []
    output_sequences = []
    prediction_counter = 0

    for i in range(len(tensors)):
        if i == 0 or prediction_counter == prediction_len+1 :
            input_seq = tensors[i:i + input_seq_len]
            input_sequences.append(torch.stack(input_seq))
            #print("input_seq",input_seq)
            
            output_seq = tensors[i + input_seq_len:i + input_seq_len + prediction_len]
            output_sequences.append(torch.stack(output_seq))
            #print("output_seq",output_seq)
            prediction_counter = 0
        prediction_counter += 1

    total_pairs = int(len(input_sequences)/input_seq_len)
    print("Total input-output pairs:", total_pairs)
    ntrain = int(total_pairs * train_ratio)
    print("Number of training pairs:", ntrain)
    ntest = (total_pairs - ntrain)
    print("Number of testing pairs:", ntest)
    
    x_train = torch.stack(input_sequences[:ntrain])
    y_train = torch.stack(output_sequences[:ntrain])
    x_test = torch.stack(input_sequences[-ntest:])
    y_test = torch.stack(output_sequences[-ntest:])
    
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    timestep = lastTimeStep(train_loader, input_seq_len, sorted_keys) # Get the last timestep of the training set, used for visualization in modelEval.py
    return train_loader, test_loader, combined_loader,timestep

def create_single_dataset(variables, batch_size):
    sorted_keys = sorted(variables.keys())  # Ensure keys are sorted for sequential processing
    tensors = [torch.tensor(variables[key].copy(), dtype=torch.float32) for key in sorted_keys]
    input_sequences = []
    output_sequences = []
    prediction_counter = 0

    for i in range(len(tensors)):
        if i == 0 or prediction_counter == 120 :
            input_seq = tensors[i:i + 1]
            input_sequences.append(torch.stack(input_seq))
            #print("input_seq",input_seq)
            
            output_seq = tensors[i + 1:i + 1 + 120]
            output_sequences.append(torch.stack(output_seq))
            #print("output_seq",output_seq)
            prediction_counter = 0
        prediction_counter += 1

    #total_pairs = int(len(input_sequences)/input_seq_len)
    #print("Total input-output pairs:", total_pairs)
    #ntrain = int(total_pairs * train_ratio)
    #print("Number of training pairs:", ntrain)
    #ntest = (total_pairs - ntrain)
    #print("Number of testing pairs:", ntest)
    
    x_train = torch.stack(input_sequences)
    y_train = torch.stack(output_sequences)
    
    train_dataset = CustomDataset(x_train, y_train)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
 
    timestep = lastTimeStep(train_loader, 1, sorted_keys) # Get the last timestep of the training set, used for visualization in modelEval.py   
    return train_loader


def lastTimeStep(dataLoader, input_seq_len, sorted_keys):
    total_size = 0
    for batch_idx, batch in enumerate(dataLoader):
            x, y = batch['x'], batch['y']
            
            total_size = x.size(1) + y.size(1)+total_size
            
            #print(f"Batch {batch_idx}:")
            #print("Input Tensor:")
            #print(x)  # Print only up to 'limit' elements if specified
            #print("Output Tensor:")
    return (total_size*5)

