import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = torch.tensor(self.data[idx], dtype=torch.long)
        return ids[:-1], ids[1:]