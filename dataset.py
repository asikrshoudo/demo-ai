"""Turns train.txt into (input, target) pairs for next-character prediction.
Each target is just the input shifted one position to the right."""

import random

from tokenizer import CharTokenizer
from config import CONTEXT_LENGTH, BATCH_SIZE, SEED

random.seed(SEED)


def load_dataset(path="data/train.txt", val_fraction=0.1):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer()
    tok.build_vocab(text)
    tok.save("tokenizer_vocab.json")

    data = tok.encode(text)
    split = int(len(data) * (1 - val_fraction))

    return tok, data[:split], data[split:]


def get_batch(data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH):
    if len(data) <= context_length:
        raise ValueError(
            f"Dataset only has {len(data)} tokens, need more than "
            f"context_length={context_length}."
        )

    x_batch, y_batch = [], []
    for _ in range(batch_size):
        start = random.randint(0, len(data) - context_length - 1)
        chunk = data[start : start + context_length + 1]
        x_batch.append(chunk[:-1])
        y_batch.append(chunk[1:])

    return x_batch, y_batch
