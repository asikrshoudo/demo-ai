# demo-ai

A tiny decoder-only Transformer language model, built and trained entirely
from scratch -- no pretrained weights, no external AI API. Runs fully
locally (trains and generates on-device).

## Pipeline

```
data/train.txt -> tokenizer -> tokens -> Transformer -> loss -> backprop -> checkpoint -> generate/chat
```

## Setup

```bash
pip install -r requirements.txt
# Termux users: pkg install python-torch  (instead of pip)
```

## Usage

```bash
python fetch_data.py      # downloads + cleans the public-domain sample text
python train.py           # trains the model, saves checkpoints/ckpt.pt
python generate.py "the fox"
python chat.py
```

## Config

All hyperparameters (model size, context length, training steps, etc.) live
in `config.py`.

## Notes

- Tokenizer is character-level: simple, no extra dependencies, handles any
  Unicode text.
- This is a learning project -- the model is intentionally small
  (currently ~600K parameters) and is not expected to produce
  fluent or reliable output. It's meant to demonstrate the full training
  pipeline (tokenizer, dataset batching, forward pass, loss, backprop,
  checkpointing, inference) working end to end.
