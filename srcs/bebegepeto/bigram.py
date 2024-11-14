import torch
from icecream import ic
import typing as tp

DATASET = "../data/input.txt"
TRAIN_TEST_SPLIT = 0.9
BLOCK_SIZE = 8
batch_size = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

with open(DATASET, "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(len(data) * TRAIN_TEST_SPLIT)
train_data = data[:n_train]
val_data = data[n_train:]


def get_batch(split: tp.Literal["train", "val"]):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x.to(device=device), y.to(device=device)
