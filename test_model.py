import torch
from model import MiniGPT
from tokenizer import CharTokenizer
from config import N_EMBD, N_LAYER, N_HEAD, CONTEXT_LENGTH

tok = CharTokenizer()
tok.load("tokenizer_vocab.json")

model = MiniGPT(
    vocab_size=tok.vocab_size,
    n_embd=N_EMBD,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    context_length=CONTEXT_LENGTH,
)

print(f"Vocab size:  {tok.vocab_size}")
print(f"Parameters:  {model.num_params():,}")

# dummy batch: 2 sequences, 8 tokens each, random token IDs
dummy_input = torch.randint(0, tok.vocab_size, (2, 8))
logits = model(dummy_input)

print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {logits.shape}   (expect: batch=2, seq=8, vocab={tok.vocab_size})")
