import os, torch, json
from model.tokenizer import Tokenizer
from config.paths import (tokenizer_path, data_path)


def prepare_data(input_path, save_dir, vocab_size, seq_len):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read().splitlines()
    
    tokenizer = Tokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    tokenizer.save(os.path.join(save_dir, tokenizer_path))
    
    data_ids = [tokenizer.encode(t, max_len=seq_len, add_bos=True, add_eos=True) for t in text]
    torch.save(data_ids, os.path.join(save_dir, data_path))
    print('ready!')