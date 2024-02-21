"""
This module provides functionalities for converting sequential data into PyTorch datasets suitable for machine learning models, particularly for training and testing purposes. It includes the definition of a custom Dataset class to handle input-output pairs and a function to prepare and split the data into training and testing sets according to specified criteria.

Classes:
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

from torch.utils.data import DataLoader, Dataset
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
    last_timesteps = []
    
    for i in range(len(tensors) - input_seq_len - prediction_len + 1):
        input_seq = tensors[i:i + input_seq_len]
        output_seq = tensors[i + input_seq_len:i + input_seq_len + prediction_len]
        input_sequences.append(torch.stack(input_seq))
        output_sequences.append(torch.stack(output_seq))
        last_timesteps.append(sorted_keys[i + input_seq_len - 1])  # Store the last timestep of the input sequence

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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    timestep = lastTimeStep(ntrain, input_seq_len, sorted_keys) # Get the last timestep of the training set, used for visualization in modelEval.py
    return train_loader, test_loader,timestep

def lastTimeStep(ntrain, input_seq_len, sorted_keys):
    if ntrain > 0:  # Ensure there is at least one sequence in the training set
        # -1 to adjust for zero indexing, and another -1 to ensure we're looking at the start of the last sequence
        last_train_sequence_start_index = (ntrain - 1) * input_seq_len
        last_train_key = sorted_keys[min(last_train_sequence_start_index + input_seq_len - 1, len(sorted_keys) - 1)]
        print("Last timestep key in training set:", last_train_key)
        return last_train_key
    else:
        print("Training set is empty.")



