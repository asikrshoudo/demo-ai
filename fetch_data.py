"""
Builds train.txt from two sources:

1. daily_dialog (ConvLab's HuggingFace mirror) -- real, human-written
   everyday conversations. Formatted with USER:/A: tags.
2. TinyStories -- simple narrative stories, tagged with STORY:.

Keeping the tags distinct is what lets the model tell "reply mode" apart
from "narrative mode", instead of always drifting into story-continuation.
"""

import urllib.request
import zipfile
import json

DAILYDIALOG_ZIP_URL = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip"
DAILYDIALOG_ZIP_PATH = "data/dailydialog.zip"

STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
STORIES_RAW_PATH = "data/tinystories_raw.txt"

OUTPUT_PATH = "data/train.txt"
STORY_TARGET_CHARS = 1_500_000


def fetch_conversations():
    print("Downloading daily_dialog...")
    urllib.request.urlretrieve(DAILYDIALOG_ZIP_URL, DAILYDIALOG_ZIP_PATH)

    with zipfile.ZipFile(DAILYDIALOG_ZIP_PATH, "r") as z:
        with z.open("data/dialogues.json") as f:
            dialogues = json.load(f)

    lines = []
    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            # map their "user"/"system" labels onto our USER:/A: tags
            speaker = "USER" if turn["speaker"] == "user" else "A"
            utterance = turn["utterance"].strip()
            if utterance:
                lines.append(f"{speaker}:\n{utterance}\n")

    return "\n".join(lines)


def fetch_stories():
    print("Downloading TinyStories sample...")
    urllib.request.urlretrieve(STORIES_URL, STORIES_RAW_PATH)
    with open(STORIES_RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read(STORY_TARGET_CHARS)
    return f"STORY:\n{text}\n"


def main():
    conversations = fetch_conversations()
    stories = fetch_stories()

    combined = conversations + "\n\n" + stories

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"Conversation text: {len(conversations):,} characters")
    print(f"Story text:        {len(stories):,} characters")
    print(f"Total train.txt:   {len(combined):,} characters")


if __name__ == "__main__":
    main()
