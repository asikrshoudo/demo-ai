"""Central config for the tiny language model. Change these to experiment
with model size, training length, and generation behavior."""

# Tokenizer / data
CONTEXT_LENGTH = 128       # how many previous tokens the model can see at once

# Model architecture (~600K params at these settings)
N_EMBD = 128                # dimensionality of each token's vector
N_LAYER = 3                 # number of transformer blocks stacked
N_HEAD = 4                  # attention heads per block

# Training
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
MAX_STEPS = 3000
EVAL_INTERVAL = 200
CHECKPOINT_INTERVAL = 500

# Generation
TEMPERATURE = 0.8
TOP_K = 20

# Reproducibility
SEED = 1337
