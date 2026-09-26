import torch
import torch.nn.functional as F
from model import MiniGPT
from tokenizer import CharTokenizer
from dataset import load_dataset, get_batch
from config import N_EMBD, N_LAYER, N_HEAD, CONTEXT_LENGTH, BATCH_SIZE

# 1. Load real data (not dummy random IDs this time)
tok, train_data, val_data = load_dataset()

model = MiniGPT(
    vocab_size=tok.vocab_size,
    n_embd=N_EMBD,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    context_length=CONTEXT_LENGTH,
)

# 2. Get one real batch
x, y = get_batch(train_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH)
x = torch.tensor(x)   # (B, T) input token IDs
y = torch.tensor(y)   # (B, T) target token IDs

# 3. Forward pass
logits = model(x)     # (B, T, vocab_size)

# 4. Cross-entropy loss expects (N, C) and (N,), so flatten batch+time dims
B, T, C = logits.shape
loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

print(f"Logits shape: {logits.shape}")
print(f"Loss (untrained model): {loss.item():.4f}")

# Sanity check: for a completely untrained model with `vocab_size` classes,
# random guessing gives loss ≈ ln(vocab_size). If our loss is close to that,
# it confirms the model starts "knowing nothing" — as expected before training.
import math
expected_random_loss = math.log(tok.vocab_size)
print(f"Expected random-guess loss: ~{expected_random_loss:.4f}")

# 5. Backward pass — this computes gradients for every parameter
loss.backward()

# 6. Check that gradients actually got created
total_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
total_params = sum(1 for p in model.parameters())
print(f"Parameters with gradients: {total_params_with_grad}/{total_params}")

# Peek at one gradient to prove it's not all zeros
sample_grad = model.head.weight.grad
print(f"Sample gradient (output layer) - mean abs value: {sample_grad.abs().mean().item():.6f}")
