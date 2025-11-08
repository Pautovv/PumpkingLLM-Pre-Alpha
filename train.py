import torch, os
from tqdm import tqdm
from model.gpt import GPT
from model.tokenizer import Tokenizer
from data.dataset import TextDataset

def train_model(epochs, seq_len, batch_size, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = Tokenizer.load('data/tokenizer.json')
    data = torch.load('data/dataset.pt')
    
    dataset = TextDataset(data, seq_len=seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GPT(
        layers_count=4,
        embedding_dim=265,
        head_count=8,
        ffl_dim=512,
        dropout_rate=0.1,
        vocab_size=len(tokenizer.vocab),
        max_len=seq_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x, training_flag=True)
            
            loss = criterion(logits.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx%100 == 0:
                print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}')
                
        print(f'Epoch {epoch} finished. Average loss : {total_loss / len(loader):.4f}')
    
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, f'gpt_epoch{epoch}.pt'))
            