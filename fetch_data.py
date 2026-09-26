"""
Downloads a public-domain text (Aesop's Fables, via Project Gutenberg) and
strips Gutenberg's license header/footer, leaving just the story text.

Run this once before training if data/gutenberg_clean.txt doesn't exist yet.
"""

import re
import urllib.request

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/21/pg21.txt"
RAW_PATH = "data/raw_gutenberg.txt"
CLEAN_PATH = "data/gutenberg_clean.txt"

START_MARKER = "*** START OF THE PROJECT GUTENBERG"
END_MARKER = "*** END OF THE PROJECT GUTENBERG"


def download():
    print(f"Downloading {GUTENBERG_URL} ...")
    urllib.request.urlretrieve(GUTENBERG_URL, RAW_PATH)
    print(f"Saved to {RAW_PATH}")


def clean():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    start = text.find(START_MARKER)
    start = text.find("\n", start) + 1 if start != -1 else 0

    end = text.find(END_MARKER)
    end = end if end != -1 else len(text)

    body = text[start:end].strip()
    body = re.sub(r"\n{3,}", "\n\n", body)  # collapse extra blank lines

    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        f.write(body)

    print(f"Cleaned text written to {CLEAN_PATH} ({len(body)} chars)")


if __name__ == "__main__":
    download()
    clean()
