import torch


import torch.nn as nn
from torch.nn import functional as F


def load_shakespear_data():
    with open("shakespeare.txt", "r") as f:
        text = f.read()
    return text

text = load_shakespear_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(chars)

stoi = {chars[i]:i for i in range(len(chars))}
itos = {i:chars[i] for i in range(len(chars))}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9 * len(text))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print(xb.shape, yb.shape)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        B,T,C = logits.shape
        print(B,T,C)
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits)
print(loss)