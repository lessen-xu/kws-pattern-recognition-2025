Task A – Word Extraction

1.Input
The raw dataset is stored in:
src/KWS/
This folder contains:

images/ (full page document images)

locations/ (SVG files containing word-level polygon paths)

transcription.tsv

train.tsv

validation.tsv

keywords.tsv

2.Process
Task A is implemented in extract_words.py located in:
src/A_data/

The script reads all images and SVG files from src/KWS/, parses each SVG <path> element, extracts polygon coordinates, computes the bounding box for each word, and crops the corresponding region from the page image.

3.Output
All cropped word images are saved in:
src/A_data/cropped_words/

A metadata file named words_metadata.tsv is created in:
src/A_data/

Each line in words_metadata.tsv contains:
word_id page xmin ymin xmax ymax relative_image_path

4.How to run
Open a terminal and execute:

python src/A_data/extract_words.py

The script automatically reads input from src/KWS/ and writes all outputs to src/A_data/.

5.Usage
The outputs of Task A (cropped word images and metadata file) are required for Task B (pair generation), Task C (model training), and Task D (evaluation).