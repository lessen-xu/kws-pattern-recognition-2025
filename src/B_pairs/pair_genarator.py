import json
import random
from pathlib import Path

# directory structure
ROOT = Path(__file__).resolve().parents[1]  # src/
DATA_ROOT = ROOT / "KWS"

META_PATH = ROOT / "A_data" / "words_metadata.tsv"
TRANS_PATH = DATA_ROOT / "transcription.tsv"
KEYWORD_PATH = DATA_ROOT / "keywords.tsv"

OUT_PATH = ROOT / "B_pairs" / "pairs_train.json"


def load_metadata():
    metadata = {}
    with open(META_PATH, "r") as f:
        next(f)
        for line in f:
            word_id, page, xmin, ymin, xmax, ymax, rel_path = line.strip().split("\t")
            img_path = DATA_ROOT / "cropped_words" / f"{word_id}.png"
            metadata[word_id] = img_path.as_posix()
    return metadata


def load_transcriptions():
    trans = {}
    with open(TRANS_PATH, "r") as f:
        for line in f:
            wid, txt = line.strip().split("\t")
            trans[wid] = txt.lower()
    return trans


def load_keywords():
    with open(KEYWORD_PATH, "r") as f:
        return [line.strip() for line in f]


def generate_positive_pairs(ids):
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((ids[i], ids[j], 1))
    return pairs


def generate_negative_pairs(all_ids, num_pos):
    pairs = []
    while len(pairs) < num_pos:
        a, b = random.sample(all_ids, 2)
        if a != b:
            pairs.append((a, b, 0))
    return pairs


def main():
    print("Loading data...")

    metadata = load_metadata()
    trans = load_transcriptions()
    keywords = load_keywords()

    trans2ids = {}
    for wid, t in trans.items():
        trans2ids.setdefault(t, []).append(wid)

    positive_pairs = []
    for kw in keywords:
        target_t = kw.lower()
        if target_t in trans2ids:
            ids = trans2ids[target_t]
            if len(ids) >= 2:
                positive_pairs.extend(generate_positive_pairs(ids))

    print(f"Positive pairs: {len(positive_pairs)}")

    all_ids = list(metadata.keys())
    negative_pairs = generate_negative_pairs(all_ids, len(positive_pairs))
    print(f"Negative pairs: {len(negative_pairs)}")

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    result = []
    for a, b, label in all_pairs:
        result.append({
            "img1": metadata[a],
            "img2": metadata[b],
            "label": label
        })

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
