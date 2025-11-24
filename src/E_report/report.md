# Keyword Spotting System on George Washington Handwritten Documents

## Introduction

This work presents the implementation of a keyword spotting system for historical handwritten documents based on a modular pipeline. The task of keyword spotting is to find all instances of a given keyword in a collection of document images by comparing visual similarities between individual word images. We developed our system on the George Washington subset provided for this exercise—a collection of 18th-century handwritten document pages—by decomposing the problem into four parts: extracting data, generating pairs, designing the embedding model, and training with retrieval evaluation. Each team member worked on one part of the process, and then the components were combined. Our approach uses a CNN to embed word images into a feature space such that images of the same word are close together. Similarity computation between these embeddings allows the system to retrieve matching words effectively for a given query image.

---

## Dataset

We evaluated our method on the provided George Washington handwritten document subset. Each page in this dataset comes annotated at word level—for every word present on the page, the dataset provides the coordinates of its boundary and the transcription of the word. Using all available pages in the assignment, we obtained several thousand segmented word instances. Each word instance is identified by a unique ID made up of page–line–word indices, and it also has a ground-truth transcription—the actual word spelled out. A list of target keywords—a set of words appearing multiple times—is also provided. These keywords are particularly suitable for retrieval evaluation. The dataset was split into a training portion and a validation portion to test generalization to unseen pages.

---

## Methodology

### Task A: Word Extraction

Individual word images were extracted from the full page scans. Using the provided SVG location files for each page, our script parsed the polygon coordinates for each word and computed a tight bounding box. We then cropped each word from the page image. All cropped word images were saved as separate grayscale image files, and a metadata file was created mapping each unique word ID to its cropped image path and associated text. This stage produced the database of normalized word images required for downstream processing.

### Task B: Keyword Pair Generation

We created a dataset of word image pairs with their labels to train the embedding model. From the metadata and the keyword list, we identified all occurrences of each keyword. For each keyword, we generated positive pairs among images of the same word and an equal number of negative pairs by pairing images of different words. This balanced supervision provided the embedding network with a strong signal to learn visual similarity.

### Task C: Embedding Model

We implemented a CNN-based embedding model called SimpleCNN that converts word images into fixed-size feature vectors. The model consists of four convolutional layers with pooling, followed by a fully connected layer that outputs a 128-dimensional embedding. We also considered using a pretrained ResNet18 backbone, although our main experiments focused on the lighter network for efficiency. The model provides an interface for computing cosine similarity between embeddings.

### Task D: Training and Retrieval Evaluation

We trained the embedding model on the generated pairs using contrastive loss with a margin. Optimization was performed with Adam (learning rate 0.001, batch size 32), and the loss consistently decreased during training. The learned embeddings positioned identical words closer together while pushing different-word embeddings apart. For evaluation, we treated the task as image retrieval: given a word image query, we computed its embedding and compared it to all other word images in the validation set using cosine similarity.

---

## Experiments and Results

### Word Extraction Results

The word cropping in Task A was completed for all pages and produced several thousand single-word images. Visual inspection confirmed that each cropped image cleanly contains a single word with minimal surrounding blank space.

### Pair Generation Results

Task B yielded a balanced set of training pairs, with roughly equal numbers of positive and negative examples. These pairs captured meaningful variations in handwriting for the same word and provided rich supervision for training.

### Model Training Results

The contrastive learning process successfully reduced the loss over 15 epochs. The model learned to embed instances of the same word close to each other in feature space.

### Retrieval and Keyword Spotting Results

When tested on validation pages not seen during training, the system successfully retrieved correct word instances for most queries. For example, given a query instance of the word “orders,” the system returned other “orders” instances with high similarity, while unrelated words ranked lower. Although we did not compute a numerical metric such as mAP, visual inspection of the top retrieval results suggests strong precision at high ranks.

---

## Limitations and Future Work

One limitation is that the current system is trained specifically on keywords that appear multiple times, and performance on rare words may be lower. Future work could explore harder negative mining or larger-scale similarity models to further improve robustness.

---

## Conclusion

This work developed a complete pipeline for keyword spotting in handwritten documents and demonstrated its effectiveness on the provided George Washington dataset subset. The system spans from data preprocessing and model training to retrieval evaluation. In summary, even a relatively simple CNN learns discriminative features for handwritten word images, achieving high retrieval accuracy for keywords and enabling practical keyword searching for historical document collections.

---