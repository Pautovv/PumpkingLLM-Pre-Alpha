import torch

def build_mask(self, seq):
    pad_mask = (seq == 0)
    attention_mask = torch.triu(torch.ones((seq.size(1), seq.size(1)), device=seq.device), diagonal=1).bool()
    return pad_mask, attention_mask