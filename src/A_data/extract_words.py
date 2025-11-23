import os
import xml.etree.ElementTree as ET
from PIL import Image
import csv
import re

# ---------------------------------------------------------
# Task A – Word Extraction (Auto Path Version)
# ---------------------------------------------------------
# Script location:     src/A_data/extract_words.py
# Raw dataset folder:  src/KWS/
# Outputs go to:       src/A_data/
# ---------------------------------------------------------

# Absolute path of THIS SCRIPT:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = one level above A_data
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input folder (raw data)
RAW_DIR = os.path.join(PROJECT_ROOT, "KWS")
IMG_DIR = os.path.join(RAW_DIR, "images")
LOC_DIR = os.path.join(RAW_DIR, "locations")

# Output folder (Task A results)
OUT_DIR = SCRIPT_DIR                      # A_data/
CROPPED_DIR = os.path.join(OUT_DIR, "cropped_words")
META_FILE = os.path.join(OUT_DIR, "words_metadata.tsv")

os.makedirs(CROPPED_DIR, exist_ok=True)


# ---------------------------------------------------------
# SVG parsing utilities
# ---------------------------------------------------------

def parse_path_d(d_string):
    """Extract coordinate pairs from SVG path 'd' attribute."""
    coords = re.findall(r"(\d+\.?\d*)[ ,]+(\d+\.?\d*)", d_string)
    return [(float(x), float(y)) for x, y in coords]


def parse_svg(svg_path):
    """Parse <path> polygons and compute bounding boxes."""
    ns = {"svg": "http://www.w3.org/2000/svg"}
    tree = ET.parse(svg_path)
    root = tree.getroot()

    words = []
    for node in root.findall(".//svg:path", ns):
        word_id = node.get("id")
        d = node.get("d")
        if not d or not word_id:
            continue

        pts = parse_path_d(d)
        if not pts:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        words.append((word_id, int(xmin), int(ymin), int(xmax), int(ymax)))

    return words


# ---------------------------------------------------------
# Main Extraction Function
# ---------------------------------------------------------

def extract_all_words():
    metadata = []

    svg_files = [f for f in os.listdir(LOC_DIR) if f.endswith(".svg")]
    svg_files.sort()

    for svg_file in svg_files:
        page_id = svg_file.replace(".svg", "")

        svg_path = os.path.join(LOC_DIR, svg_file)

        # Load matching image (PNG or JPG)
        img_path_png = os.path.join(IMG_DIR, page_id + ".png")
        img_path_jpg = os.path.join(IMG_DIR, page_id + ".jpg")

        if os.path.exists(img_path_png):
            img_path = img_path_png
        elif os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        else:
            print(f"[WARNING] No image found for page {page_id}")
            continue

        image = Image.open(img_path).convert("RGB")

        # Parse SVG for word bounding boxes
        words = parse_svg(svg_path)

        for word_id, xmin, ymin, xmax, ymax in words:
            crop = image.crop((xmin, ymin, xmax, ymax))

            out_path = os.path.join(CROPPED_DIR, f"{word_id}.png")
            rel_path = os.path.relpath(out_path, OUT_DIR).replace("\\", "/")

            crop.save(out_path)
            metadata.append([word_id, page_id, xmin, ymin, xmax, ymax, rel_path])

        print(f"[DONE] {page_id} – extracted {len(words)} words")

    # Save metadata
    with open(META_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["word_id", "page", "xmin", "ymin", "xmax", "ymax", "path"])
        writer.writerows(metadata)

    print("\n[FINISHED] Task A complete.")
    print(f"Metadata saved to {META_FILE}")
    print(f"Cropped words saved to {CROPPED_DIR}")


# ---------------------------------------------------------
if __name__ == "__main__":
    extract_all_words()
