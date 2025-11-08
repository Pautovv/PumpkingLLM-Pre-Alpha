import json, torch
from collections import Counter

class Tokenizer():
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.token2id = dict()
        self.id2token = dict()
        
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
    
    def pairs_frequencies(self, corpus):
        pairs = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] += 1
        return pairs

    def merge_pairs(self, pair, corpus):
        new_token = ''.join(pair)
        new_corpus = list()
        for word in corpus:
            new_word = list()
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(new_token)
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_corpus.append(new_word)
        return new_corpus
    
    def train(self, corpus):
        for word in corpus:
            for c in word:
                self.vocab.add(c)
        
        self.vocab.update([self.pad_token, self.unk_token, self.bos_token, self.eos_token])
                
        corpus = [list(word) + ['</w>'] for word in corpus]
        
        while True:
            pairs = self.pairs_frequencies(corpus)
            
            if not pairs:
                break
            
            max_pair = pairs.most_common(1)[0][0]
            corpus = self.merge_pairs(max_pair, corpus)
            self.vocab.add(''.join(max_pair))
            
            if len(self.vocab) >= self.vocab_size:
                break
        
        for i, token in enumerate(sorted(self.vocab)):
            self.token2id[token] = i
            self.id2token[i] = token
            
        return corpus, self.vocab
    
    def encode(self, text, max_len, add_bos, add_eos):
        word = list(text) + ['</w>']
        tokens = list()
        i = 0
        while i < len(word):
            flag=False
            for j in range(len(word), i, -1):
                piece = ''.join(word[i:j])
                if piece in self.vocab:
                    tokens.append(piece)
                    i += len(piece)
                    flag=True
                    break
            if not flag:
                tokens.append(self.unk_token)
                i += 1
        if add_bos:
            tokens = [self.bos_token] + tokens
        if add_eos:
            tokens += [self.eos_token]
        
        ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]
        
        if max_len is not None:
            if len(ids) < max_len:
                ids += [self.token2id[self.pad_token]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        
        return ids
        
    def decode(self, token_ids, skip_special):
        tokens = [self.id2token[id] for id in token_ids]
        text = ''
        for t in tokens:
            if skip_special and t in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            text+=t                
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def batch_encode(self, texts, max_len, add_bos, add_eos):
        batch_ids = [self.encode(t, max_len, add_bos, add_eos) for t in texts]
        return torch.tensor(batch_ids, dtype=torch.long)
    
    def save(self, path):
        json.dump({'token2id' : self.token2id, 'id2token' : self.id2token}, open(path, 'w'))
    
    @staticmethod
    def load(self, path):
        data = json.load(open(path))
        self.token2id = data['token2id']
        self.id2token = {int(k):v for k,v in data['id2token'].items()}
        
    