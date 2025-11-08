import torch

class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_dim, head_count, ffl_dim, dropout_rate):
        super(DecoderBlock, self).__init__()
        
        self.MultiHeadAttention = torch.nn.MultiheadAttention(embedding_dim,head_count, dropout_rate, batch_first=True)
        self.ffl = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffl_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffl_dim, embedding_dim),
        )
        self.normal_1 = torch.nn.LayerNorm(embedding_dim)
        self.normal_2 = torch.nn.LayerNorm(embedding_dim)
        
        self.dropout_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_2 = torch.nn.Dropout(dropout_rate)
    
    def forward(self, X, padding_mask, attention_mask):
        mha, _ = self.MultiHeadAttention(X, X, X, attn_mask=attention_mask, key_padding_mask=padding_mask)
        X += self.dropout_1(mha)
        X = self.normal_1(X)
        
        ffl = self.ffl(X)
        X += self.dropout_2(ffl)
        X = self.normal_2(X)
        
        return X

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pos_encoding", self.positional_encoding(max_len, embedding_dim))
        
    def positional_encoding(self, max_len, embedding_dim):
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        index = torch.arange(embedding_dim, dtype=torch.float32).unsqueeze(0)
        
        args = positions / torch.pow(10_000, (2 * torch.floor(index / 2)) / embedding_dim)
        
        args[:, 0::2] = torch.sin(args[:, 0::2])
        args[:, 1::2] = torch.cos(args[:, 1::2])
        
        return args.unsqueeze(0)
    
    def forward(self, X):
        return X + self.pos_encoding[:, :X.size(1), :].to(X)