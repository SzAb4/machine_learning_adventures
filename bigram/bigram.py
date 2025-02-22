import random
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def sample_from_probabilities(prob_dict):
    """
    Sample a key from a dictionary based on probability weights.
    
    Args:
        prob_dict (dict): Dictionary with keys and their probabilities
        
    Returns:
        str: Randomly selected key based on probabilities
    """
    # Extract keys and probabilities into separate lists
    keys = list(prob_dict.keys())
    probabilities = list(prob_dict.values())
    
    # Use random.choices() to sample one key based on probabilities
    # choices() returns a list, so we take the first element with [0]
    return random.choices(keys, weights=probabilities, k=1)[0]


with open("bigram/names.txt", "r") as f:
    words = f.read().splitlines()

bigrams = {}
for w in words:
    for ch1, ch2 in zip(["<S>"] + list(w), list(w) + ["<E>"]):
        bigram = (ch1,ch2)
        bigrams[bigram] = bigrams.get(bigram, 0) + 1

bigram_counts = {}
for bigram, count in bigrams.items():
    ch1, ch2 = bigram
    if ch1 in bigram_counts.keys():
        bigram_counts[ch1][ch2] = count
    else:
        bigram_counts[ch1] = {ch2:count}
def get_probs(counts):
    countsum = sum(counts.values())
    return {ch:count/countsum for ch, count in counts.items()}

#bigram_probs = {ch1:get_probs(counts) for ch1, counts in bigram_counts.items()}
# print(bigram_probs["e"])
# print(sum(bigram_probs["s"].values()))
def produce_word():
    chars = ["<S>"]
    while True:
        newch = sample_from_probabilities(bigram_probs[chars[-1]])
        if newch == "<E>":
            return "".join(chars[1:])
        chars.append(newch)

# for _ in range(5):
#     print(produce_word())

N = torch.zeros((28,28), dtype = torch.int32)
chars = sorted(list(set("".join(words))))

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}



# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off')
# plt.show()

# p = N[0].float()
# p = p / p.sum()
# g = torch.Generator().manual_seed(2147483647)
# ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# print(itos[ix])


xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)

W = torch.randn((27,27), requires_grad=True)

batchsize = 10
for k in range(2000):
    idxes = torch.randint(0, len(xs), (batchsize,))
    xsbatch = xs[idxes]
    xenc = F.one_hot(xsbatch, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(batchsize), ys[idxes]].log().mean()
    print(f"Loss at round {k}: {loss:.4f}")
    W.grad = None
    loss.backward()

    W.data += -10 * W.grad