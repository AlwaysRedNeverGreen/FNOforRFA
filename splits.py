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
    tensors = [torch.tensor(variables[key].copy(), dtype=torch.float32) for key in sorted(variables.keys()) if key in variables]
    
    # Prepare input-output pairs for sequential training
    input_sequences = []
    output_sequences = []
    for i in range(len(tensors) - input_seq_len - prediction_len + 1):
        input_seq = tensors[i:i + input_seq_len]
        output_seq = tensors[i + input_seq_len:i + input_seq_len + prediction_len]
        input_sequences.append(torch.stack(input_seq))
        output_sequences.append(torch.stack(output_seq))

    # Determine train and test set sizes based on the ratio
    total_pairs = len(input_sequences)
    ntrain = int(total_pairs * train_ratio)
    ntest = total_pairs - ntrain

    # Split into train and test sets
    x_train = torch.stack(input_sequences[:ntrain])
    y_train = torch.stack(output_sequences[:ntrain])
    x_test = torch.stack(input_sequences[-ntest:])
    y_test = torch.stack(output_sequences[-ntest:])

    # Create custom datasets
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
