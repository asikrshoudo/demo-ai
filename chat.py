"""Terminal chat interface, powered entirely by the locally trained model.
No external API calls -- everything runs on-device.

Usage:
    python chat.py
    /exit to quit
"""

import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import CharTokenizer
from config import N_EMBD, N_LAYER, N_HEAD, CONTEXT_LENGTH, TEMPERATURE, TOP_K

CHECKPOINT_PATH = "checkpoints/ckpt.pt"
VOCAB_PATH = "tokenizer_vocab.json"
MAX_REPLY_TOKENS = 80


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
def generate_reply(model, tok, prompt, max_new_tokens=MAX_REPLY_TOKENS,
                    temperature=TEMPERATURE, top_k=TOP_K):
    idx = tok.encode(prompt) or [0]
    idx = torch.tensor([idx])

    generated_ids = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        logits = model(idx_cond)[:, -1, :] / temperature

        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        generated_ids.append(next_id.item())

        if tok.id_to_char.get(next_id.item()) == "\n":
            break

    return tok.decode(generated_ids)


def main():
    print("TinyAI")
    print("------")
    print("(type /exit to quit)\n")

    model, tok = load_model_and_tokenizer()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input == "/exit":
            print("Exiting.")
            break
        if not user_input:
            continue

        reply = generate_reply(model, tok, user_input)
        print(f"AI: {reply.strip()}\n")


if __name__ == "__main__":
    main()
