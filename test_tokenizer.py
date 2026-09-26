from tokenizer import CharTokenizer

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tok = CharTokenizer()
tok.build_vocab(text)

print(f"Vocab size: {tok.vocab_size}")

samples = [
    "Hello world",
    "হ্যালো পৃথিবী",
    "ami ajke bhalo achi",
]

for s in samples:
    ids = tok.encode(s)
    decoded = tok.decode(ids)
    match = "OK" if decoded == s else "MISMATCH"
    print(f"\nOriginal: {s}")
    print(f"Encoded : {ids}")
    print(f"Decoded : {decoded}  [{match}]")

tok.save("tokenizer_vocab.json")
print("\nSaved vocab to tokenizer_vocab.json")
