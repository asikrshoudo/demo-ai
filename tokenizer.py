"""Character-level tokenizer. Maps every unique character in the training
text to an integer ID and back. Simple, has no external dependencies, and
handles any Unicode text without special casing."""

import json


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def build_vocab(self, text):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids):
        return "".join(self.id_to_char[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.char_to_id = json.load(f)
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
