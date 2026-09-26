"""Strip Project Gutenberg's license header/footer, keep only the book text."""

with open("data/raw_gutenberg.txt", "r", encoding="utf-8") as f:
    text = f.read()

start_marker = "*** START OF THE PROJECT GUTENBERG"
end_marker = "*** END OF THE PROJECT GUTENBERG"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx != -1:
    # skip past the marker line itself
    start_idx = text.find("\n", start_idx) + 1
else:
    start_idx = 0

if end_idx == -1:
    end_idx = len(text)

body = text[start_idx:end_idx].strip()

# collapse excessive blank lines
import re
body = re.sub(r"\n{3,}", "\n\n", body)

with open("data/gutenberg_clean.txt", "w", encoding="utf-8") as f:
    f.write(body)

print(f"Cleaned text length: {len(body)} characters")
