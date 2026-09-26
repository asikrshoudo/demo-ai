"""
Downloads a chunk of the TinyStories dataset (Eldan & Li, Microsoft Research) --
short, simple English stories written specifically to be learnable by small
language models. Source: https://huggingface.co/datasets/roneneldan/TinyStories

We use the smaller "valid" split (not the full 1.9GB train file) and cut it
down to a target character count, since our model is tiny and doesn't need
(or benefit from) the entire multi-GB dataset.
"""

import urllib.request

SOURCE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
RAW_PATH = "data/tinystories_raw.txt"
OUTPUT_PATH = "data/train.txt"

# ~8 million characters ≈ 5-10 million tokens once combined with our existing
# small conversational dataset. Adjust this if you want more/less.
TARGET_CHARS = 8_000_000


def download():
    print(f"Downloading from {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, RAW_PATH)
    print(f"Saved to {RAW_PATH}")


def build_train_file():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read(TARGET_CHARS)

    # keep our existing small conversational examples too, so the model
    # still sees dialogue-style text, not just narrative stories
    with open("data/conversations_en.txt", "r", encoding="utf-8") as f:
        conversations = f.read()

    combined = conversations + "\n\n" + text

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"Final train.txt size: {len(combined):,} characters")


if __name__ == "__main__":
    download()
    build_train_file()
