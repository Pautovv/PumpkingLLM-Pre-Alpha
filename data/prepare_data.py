import os, torch, json
from model.tokenizer import Tokenizer

def prepare_data(input_path, save_dir, vocab_size, seq_len):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read().splitlines()
    
    tokenizer = Tokenizer()
    tokenizer.train(text)
    tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))
    
    data_ids = [tokenizer.encode(t, max_len=seq_len, add_bos=True, add_eos=True) for t in text]
    torch.save(data_ids, os.path.join(save_dir, 'dataset.pt'))
    print('ready!')