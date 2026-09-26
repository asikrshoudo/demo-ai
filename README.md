# demo-ai

A tiny GPT-style language model, built completely from scratch — no pretrained weights, and no external AI API involved anywhere in the model itself. This is a personal project to understand how a language model actually works under the hood: tokenization, attention, backpropagation, training loops, checkpointing, all of it implemented and trained from zero.

## What this actually is

A decoder-only Transformer (a miniature GPT), currently around 600K parameters. For reference, that's roughly 200x smaller than GPT-2 small, and tens of thousands of times smaller than any model you'd typically chat with day to day.

**What it can do:** learn character-level patterns from a training text, generate short samples, hold a very basic back-and-forth in a terminal chat.

**What it can't do:** understand language, do real math, hold a coherent conversation, or generalize much beyond what it directly saw during training. The training dataset is still small, so expect the model to mostly echo things close to what it memorized rather than produce genuinely new text. That's an expected limitation of this scale, not a bug.

## Pipeline

```

data/train.txt → tokenizer (char-level) → tokens → Transformer → loss → backprop → checkpoint → generate.py / chat.py
```

Every step in that pipeline is plain, readable code in this repo — nothing here is a black box.

## Project structure

```
config.py       - all hyperparameters in one place (model size, training steps, etc.)
tokenizer.py    - character-level tokenizer
dataset.py      - turns train.txt into training batches
model.py        - the Transformer architecture itself
train.py        - training loop, where the actual learning happens
generate.py     - generate text from a trained checkpoint
chat.py         - terminal chat interface
fetch_data.py   - downloads a public-domain sample text to train on
data/           - training data
checkpoints/    - trained model weights get saved here

```

## Running it

```bash
pip install -r requirements.txt

python fetch_data.py     # downloads a small public-domain text to train on
python train.py          # trains from scratch, saves checkpoints/ckpt.pt
python generate.py "the fox"
python chat.py
```

## Training on GitHub Actions

Training on limited local hardware is slow, so this repo is also set up to train on GitHub's own runners:

```bash
gh workflow run train.yml
gh run watch
```

The trained checkpoint is uploaded as a build artifact, downloadable from the Actions run page.

## Current limitations

This project started as a ~110K parameter model that memorized a 10-line dataset almost perfectly, and could recite it back but do little else. It's since been scaled up to ~600K parameters with a larger dataset, but is still firmly in "the training pipeline works end to end" territory rather than "useful chatbot" territory. Scaling the model further and growing the dataset are the natural next steps.

Multilingual support (originally planned for English, Bangla, and Banglish together) has been scaled back to English-only for now, to focus on getting a solid single-language pipeline working well before adding that complexity back.

## Data sources

Training data currently includes short original conversational examples plus public-domain text from Project Gutenberg. No copyrighted or scraped-without-permission material is used.

## License

MIT — see [LICENSE](LICENSE).
