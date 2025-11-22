import torch, os
from tqdm import tqdm
from model.gpt import GPT
from model.tokenizer import Tokenizer
from data.dataset import TextDataset
from config.paths import (tokenizer_path, data_path)
from config.hyperparams import hyperparams

def train_model(epochs, seq_len, batch_size, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = Tokenizer.load(tokenizer_path)
    data = torch.load(data_path)

    dataset = TextDataset(data, seq_len=seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GPT(
        layers_count=hyperparams.layers_count,
        embedding_dim=hyperparams.embedding_dim,
        head_count=hyperparams.head_count,
        ffl_dim=hyperparams.ffl_dim,
        dropout_rate=hyperparams.dropout_rate,
        vocab_size=len(tokenizer.vocab),
        max_len=seq_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.transpose(1, 2), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} finished. Average loss: {avg_loss:.4f}")

        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, f'gpt_epoch{epoch}.pt'))
            