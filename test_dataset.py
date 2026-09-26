from dataset import load_dataset, get_batch
from config import CONTEXT_LENGTH, BATCH_SIZE

tok, train_data, val_data = load_dataset()

print(f"Vocab size:   {tok.vocab_size}")
print(f"Total tokens: {len(train_data) + len(val_data)}")
print(f"Train tokens: {len(train_data)}")
print(f"Val tokens:   {len(val_data)}")

x, y = get_batch(train_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH)
print(f"\nBatch shape: {len(x)} sequences x {len(x[0])} tokens each")

print("\n--- Example 1 ---")
print("Input :", repr(tok.decode(x[0])))
print("Target:", repr(tok.decode(y[0])))

print("\n--- Example 2 ---")
print("Input :", repr(tok.decode(x[1])))
print("Target:", repr(tok.decode(y[1])))
