import torch
from model.gpt import GPT
from model.tokenizer import Tokenizer

def generate(prompt, model_path, tokenizer_path, max_len, temperature, top_k):
    tokenizer = Tokenizer.load(tokenizer_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GPT(
        layers_count=4,
        embedding_dim=265,
        head_count=8,
        ffl_dim=512,
        dropout_rate=0.1,
        vocab_size=len(tokenizer.vocab),
        max_len=max_len
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    ids = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            out = model(input_ids, training_flag=False)
            next_logits = out[:, -1, :] / temperature
            
            values, indices = torch.topk(next_logits, k=top_k)
            probs = torch.softmax(values, dim=-1)
            next_id = indices.gather(-1, torch.multinomial(probs, 1))
        
            input_ids = torch.cat([input_ids, next_id], dim=1)
            
            if next_id.item() == tokenizer.token2id.get('<EOS>', -1):
                break

    output = tokenizer.decode(input_ids[0].tolist())
    output = output.replace('<BOS>', '').replace('<EOS>', '')
    print(output)