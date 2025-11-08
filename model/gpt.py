import torch
from layers import DecoderBlock, PositionalEncoding
from utils import build_mask

class GPT(torch.nn.Module):
    def __init__(self, layers_count, embedding_dim, head_count, ffl_dim, dropout_rate, vocab_size, max_len):
        super(GPT, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(max_len, embedding_dim)
        self.final = torch.nn.Linear(embedding_dim, vocab_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        self.decoder = torch.nn.ModuleList(
            [DecoderBlock(embedding_dim, head_count, ffl_dim, dropout_rate) for _ in range(layers_count)]
        )
    
    def forward(self, X):
        padding_mask, attention_mask = build_mask(X)
        
        X = self.embedding(X)
        X = self.positional_encoding(X)
        X = self.dropout(X)
        for dec in self.decoder:
            X = dec(X, padding_mask, attention_mask)
        
        return self.final(X)