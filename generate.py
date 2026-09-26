"""Loads a trained checkpoint and generates text from a prompt.

Usage:
    python generate.py "hello"
    python generate.py "the fox" --max_tokens 100 --temperature 0.7
"""

import argparse
import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import CharTokenizer
from config import N_EMBD, N_LAYER, N_HEAD, CONTEXT_LENGTH, TEMPERATURE, TOP_K

CHECKPOINT_PATH = "checkpoints/ckpt.pt"
VOCAB_PATH = "tokenizer_vocab.json"


def load_model_and_tokenizer():
    tok = CharTokenizer()
    tok.load(VOCAB_PATH)

    model = MiniGPT(
        vocab_size=tok.vocab_size,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        context_length=CONTEXT_LENGTH,
    )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, tok


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, temperature, top_k):
    idx = tok.encode(prompt) or [0]
    idx = torch.tensor([idx])

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        logits = model(idx_cond)[:, -1, :] / temperature

        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tok.decode(idx[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from the trained model")
    parser.add_argument("prompt")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer()
    print(generate(model, tok, args.prompt, args.max_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
