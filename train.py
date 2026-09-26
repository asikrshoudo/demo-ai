"""Training loop: samples batches, computes loss, backprops, updates weights,
checkpoints periodically, and prints a generated sample so you can watch the
model's output improve over time."""

import os
import torch
import torch.nn.functional as F

from model import MiniGPT
from dataset import load_dataset, get_batch
from config import (
    N_EMBD, N_LAYER, N_HEAD, CONTEXT_LENGTH,
    BATCH_SIZE, LEARNING_RATE, MAX_STEPS,
    EVAL_INTERVAL, CHECKPOINT_INTERVAL,
    TEMPERATURE, TOP_K, SEED,
)

torch.manual_seed(SEED)

CHECKPOINT_PATH = "checkpoints/ckpt.pt"
EVAL_ITERS = 10


def get_eval_context_length(data, desired):
    """Shrinks the context length for tiny data splits so eval doesn't crash."""
    if len(data) <= desired:
        return max(len(data) - 2, 8)
    return desired


@torch.no_grad()
def estimate_loss(model, data, context_length):
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(data, batch_size=BATCH_SIZE, context_length=context_length)
        x, y = torch.tensor(x), torch.tensor(y)
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_sample(model, tok, prompt, max_new_tokens=80):
    model.eval()
    idx = tok.encode(prompt) or [0]
    idx = torch.tensor([idx])

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        logits = model(idx_cond)[:, -1, :] / TEMPERATURE

        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    model.train()
    return tok.decode(idx[0].tolist())


def main():
    os.makedirs("checkpoints", exist_ok=True)

    tok, train_data, val_data = load_dataset()
    print(f"Vocab size: {tok.vocab_size} | Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

    model = MiniGPT(
        vocab_size=tok.vocab_size,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        context_length=CONTEXT_LENGTH,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"]
        print(f"Resumed from checkpoint at step {start_step}")
    else:
        print("No checkpoint found, training from scratch")

    val_context_len = get_eval_context_length(val_data, CONTEXT_LENGTH)

    for step in range(start_step, MAX_STEPS):
        x, y = get_batch(train_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH)
        x, y = torch.tensor(x), torch.tensor(y)

        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            train_loss = estimate_loss(model, train_data, CONTEXT_LENGTH)
            val_loss = estimate_loss(model, val_data, val_context_len)
            print(f"Step {step:5d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
            print(f"  sample: {generate_sample(model, tok, prompt='the')!r}")

        if step % CHECKPOINT_INTERVAL == 0 and step > 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
            }, CHECKPOINT_PATH)
            print(f"  checkpoint saved at step {step}")

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": MAX_STEPS,
    }, CHECKPOINT_PATH)
    print("Training finished. Final checkpoint saved.")


if __name__ == "__main__":
    main()
