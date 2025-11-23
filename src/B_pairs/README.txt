1. Input

This task uses the outputs of Task A and several files from the raw dataset.

Required files:
Located in src/A_data/:
words_metadata.tsv
(contains word IDs and cropped image paths)
Located in src/KWS/:
transcription.tsv
(character-level transcription for each word ID)
keywords.tsv
(list of keywords already in character-level format)
images/ and locations/
(not directly used in Task B, but part of the dataset structure)

2. Process

Task B is implemented in pair_generator.py, located in:
src/B_pairs/

The script performs the following steps:
Read metadata from Task A
Load the transcriptions for all word IDs
Load the keyword list (already in c-h-a-r-a-c-t-e-r format)
For each keyword:
Find all word IDs in the dataset whose transcription matches the keyword
If the keyword appears at least twice, generate positive pairs
(two images corresponding to the same keyword)
Randomly sample an equal number of negative pairs
(pairs of images from different keywords)
Save all generated pairs in a JSON file for use in Task C

3. Output

All outputs of Task B are saved in:
src/B_pairs/
Generated files:
pair_generator.py
(the script performing the pair generation)
pairs_train.json
(contains all positive and negative image pairs)
Each pair in pairs_train.json has the structure:
{
  "img1": "path/to/image1.png",
  "img2": "path/to/image2.png",
  "label": 1
}
Where label = 1 for positive pairs and 0 for negative pairs.
4. How to Run
Open a terminal at the project root and execute:
python src/B_pairs/pair_generator.py
After execution, the output file pairs_train.json will be created in:
src/B_pairs/
5. Usage

The generated image pairs from Task B are the required input for:
Task C — training the embedding model
Task D — evaluating keyword retrieval performance
Task B must be completed successfully before proceeding to Tasks C and D.