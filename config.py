"""Central config for the tiny language model."""

# Tokenizer / data
CONTEXT_LENGTH = 128

# Model architecture (~1M params at these settings)
N_EMBD = 160
N_LAYER = 3
N_HEAD = 4

# Training
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 5000
EVAL_INTERVAL = 250
CHECKPOINT_INTERVAL = 1000

# Generation
TEMPERATURE = 0.8
TOP_K = 20

# Reproducibility
SEED = 1337
