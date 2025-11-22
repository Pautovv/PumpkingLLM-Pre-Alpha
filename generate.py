import torch
from model.gpt import GPT
from model.tokenizer import Tokenizer
from config.hyperparams import hyperparams

def generate(prompt, model_path, tokenizer_path, max_len=64, temperature=1.0):
    tokenizer = Tokenizer.load(tokenizer_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT(
        layers_count=hyperparams.layers_count,
        embedding_dim=hyperparams.embedding_dim,
        head_count=hyperparams.head_count,
        ffl_dim=hyperparams.ffl_dim,
        dropout_rate=hyperparams.dropout_rate,
        vocab_size=len(tokenizer.vocab),
        max_len=max_len
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_ids = tokenizer.encode(prompt, max_len=max_len, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):

            input_ids_truncated = input_ids[:, -max_len:]

            logits = model(input_ids_truncated)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.token2id.get(tokenizer.eos_token, -1):
                break
    output = tokenizer.decode(input_ids[0].tolist(), skip_special=True)
    print(f"You: {prompt}\nBot: {output}")