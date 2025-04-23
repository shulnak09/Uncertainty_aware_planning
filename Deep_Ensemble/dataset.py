import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

class TrajDataset(Dataset):
    def __init__(self, input_data, output_data, min_val=-1, max_val=1):
        self.input_data = input_data
        self.output_data  = output_data
        self.min_val = min_val
        self.max_val = max_val
        
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self,idx):
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        sample = [input_sample, output_sample]
        
        input_sample = self.min_max_normalize(input_sample, min_val, max_val)
        output_sample = self.min_max_normalize(output_sample, min_val, max_val)
        
        return input_sample, output_sample
        
    
    def min_max_mormalize(self, data, min_val, max_val):

        
        min_data,_ = torch.min(flattened_data, dim=0)
        max_data,_ = torch.max(flattened_data, dim=0)
        normalized_data = min_val + (max_val - min_val)* (flattened_data - min_data) / (max_data - min_data)
        return normalized_data