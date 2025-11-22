import json
from collections import Counter
from tqdm import tqdm
import torch

class Tokenizer:
    def __init__(self, vocab_size, end_flag=True):
        self.vocab_size = vocab_size
        self.end_flag = end_flag

        self.vocab = set()
        self.token2id = dict()
        self.id2token = dict()
        self.merges = list()

        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        self.end_marker = '</w>' if self.end_flag else None
        self._trained = False

    def _prepare_corpus(self, corpus):
        prepared_corpus = list()
        for w in corpus:
            if self.end_flag: prepared_corpus.append(list(w) + [self.end_marker])
            else: prepared_corpus.append(list(w))
        return prepared_corpus

    def _get_pair_stats(self, corpus):
        pairs = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_corpus(self, corpus, pair):
        merged_token = pair[0] + pair[1]
        new_corpus = list()

        for word in corpus:
            new_word = list()
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(new_word)
        return new_corpus

    def train(self, corpus):
        if len(corpus) == 0: raise ValueError("Corpus us empty")

        corpus_sample = corpus[:20000]

        for word in corpus_sample:
            for ch in word:
                self.vocab.add(ch)

        for st in self.special_tokens: self.vocab.add(st)

        if self.end_flag: self.vocab.add(self.end_marker)

        corpus = self._prepare_corpus(corpus_sample)
        initial_size = len(self.vocab)
        remaining = self.vocab_size - initial_size

        if remaining <= 0:
            self._finalize_vocab()
            self._trained = True
            return corpus_sample, self.vocab

        bar = tqdm(total=remaining, desc='Training Tokenizer')

        while len(self.vocab) < self.vocab_size:
            pair_stats = self._get_pair_stats(corpus)
            if not pair_stats: break

            best_pair = pair_stats.most_common(1)[0]
            new_token = ''.join(best_pair)

            if new_token in self.vocab:
                del pair_stats[best_pair]
                if not pair_stats:
                    break
                best_pair = pair_stats.most_common(1)[0]
                new_token = ''.join(best_pair)
                if new_token in self.vocab: break

            corpus = self._merge_corpus(corpus, best_pair)
            self.vocab.add(new_token)
            self.merges.append(best_pair)

            bar.update(1)
            if bar.n >= bar.total: break

        bar.close()
        self._finalize_vocab()
        self._trained = True

        return corpus_sample, self.vocab

    def _finalize_vocab(self):
        sorted_vocab = sorted(self.vocab)
        self.token2id = {tok: i for i, tok in enumerate(sorted_vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        if self.unk_token not in self.token2id:
            unk_id = len(self.token2id)
            self.token2id[self.unk_token] = unk_id
            self.id2token[unk_id] = self.unk_token

    def encode(self, text, max_len=None, add_bos=True, add_eos=True):
        if not self._trained: raise RuntimeError("Tokenizer is not trained")

        words = text.split()
        tokens = list()

        for w in words:
            chars = list(w)
            if self.use_end_marker: chars.append(self.end_marker)

            changed_flag = True
            while changed_flag:
                changed_flag = False
                i = 0
                while i < len(chars) - 1:
                    pair = (chars[i], chars[i + 1])
                    if pair in self.merges:
                        merged = ''.join(pair)
                        chars[i:i + 2] = [merged]
                        changed_flag = True
                    else: i += 1

            tokens.extend(chars)

        if add_bos: tokens = [self.bos_token] + tokens
        if add_eos: tokens += [self.eos_token]

        ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]

        if max_len is not None:
            if len(ids) > max_len: ids = ids[:max_len]
            else:
                pad_id = self.token2id[self.pad_token]
                ids += [pad_id] * (max_len - len(ids))

        return ids

    def decode(self, token_ids, skip_special=True):
        if not self.id2token: raise RuntimeError("Vocab is empty")

        tokens = [self.id2token.get(i, self.unk_token) for i in token_ids]
        out_tokens = list()

        for t in tokens:
            if skip_special and (t in self.special_tokens): continue
            if self.use_end_marker and t == self.end_marker: continue
            if self.use_end_marker and self.end_marker in t: t = t.replace(self.end_marker, ' ')
            out_tokens.append(t)

        text = ' '.join(out_tokens)
        return ' '.join(text.split())

    def batch_encode(self, texts, max_len=None, add_bos=True, add_eos=True):
        batch = [
            self.encode(t, max_len=max_len, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]
        return torch.tensor(batch, dtype=torch.long)

    def save(self, path):
        data = {
            'vocab_size': self.vocab_size,
            'use_end_marker': self.use_end_marker,
            'token2id': self.token2id,
            'id2token': self.id2token,
            'merges': self.merges,
            'special_tokens': {
                'pad': self.pad_token,
                'unk': self.unk_token,
                'bos': self.bos_token,
                'eos': self.eos_token,
                'end_marker': self.end_marker
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)

        tokenizer = cls(
            vocab_size=data['vocab_size'],
            use_end_marker=data['use_end_marker']
        )

        tokenizer.token2id = data['token2id']
        tokenizer.id2token = {int(k): v for k, v in data['id2token'].items()}
        tokenizer.merges = [tuple(x) for x in data['merges']]

        tokenizer.pad_token = data['special_tokens']['pad']
        tokenizer.unk_token = data['special_tokens']['unk']
        tokenizer.bos_token = data['special_tokens']['bos']
        tokenizer.eos_token = data['special_tokens']['eos']
        tokenizer.end_marker = data['special_tokens']['end_marker']

        tokenizer.vocab = set(tokenizer.token2id.keys())
        tokenizer._trained = True

        return tokenizer
