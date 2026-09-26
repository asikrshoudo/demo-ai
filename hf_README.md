---
license: mit
language:
  - en
tags:
  - text-generation
  - from-scratch
  - educational
  - tiny-transformer
  - pytorch
pipeline_tag: text-generation
---

# demo-ai

A tiny GPT-style language model, built completely from scratch — no pretrained weights, and no external AI API involved anywhere in the model itself. This is a personal project to understand how a language model actually works under the hood: tokenization, attention, backpropagation, training loops, checkpointing, all of it implemented and trained from zero.

Full source code, training pipeline, and GitHub Actions training workflow: [github.com/asikrshoudo/demo-ai](https://github.com/asikrshoudo/demo-ai)

## What this actually is

A decoder-only Transformer (a miniature GPT), currently around 600K parameters. For reference, that's roughly 200x smaller than GPT-2 small, and tens of thousands of times smaller than any model you'd typically chat with day to day.

**What it can do:** learn character-level patterns from a training text, generate short samples, hold a very basic back-and-forth in a terminal chat.

**What it can't do:** understand language, do real math, hold a coherent conversation, or generalize much beyond what it directly saw during training. The training dataset is still small, so expect the model to mostly echo things close to what it memorized rather than produce genuinely new text. That's an expected limitation of this scale, not a bug.

## Files

- `ckpt.pt` — trained model + optimizer state
- `tokenizer_vocab.json` — character-level vocabulary (required to decode the model's output correctly)

## Usage

Clone the [GitHub repo](https://github.com/asikrshoudo/demo-ai) for the model/tokenizer code, then:

```python
import torch
from model import MiniGPT
from tokenizer import CharTokenizer

tok = CharTokenizer()
tok.load("tokenizer_vocab.json")

model = MiniGPT(vocab_size=tok.vocab_size, n_embd=128, n_layer=3, n_head=4, context_length=128)
ckpt = torch.load("ckpt.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()
```

See `generate.py` / `chat.py` in the GitHub repo for full generation code.

## Training data

Short original conversational examples plus public-domain text from Project Gutenberg.

## License

MIT
