Task A - Word Extraction

Input data:
src/KWS-competition/images/
src/KWS-competition/locations/

Output:
src/KWS-competition/cropped_words/
src/A_data/words_metadata.tsv

How to run:
cd src/A_data
python extract_words.py

Description:
This script reads each SVG file in locations/, extracts polygon coordinates,
computes bounding boxes, crops word images from the page images, and writes
all results into words_metadata.tsv with relative paths.

The output metadata is used by Task B, Task C, and Task D.
