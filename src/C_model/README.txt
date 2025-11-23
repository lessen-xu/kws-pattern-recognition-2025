# CNN Embedding Models for Keyword Spotting

**Date:** November 2024  
**Project:** Keyword Spotting on George Washington Historical Documents Dataset

---

##  Overview

This module provides CNN-based embedding models for keyword spotting tasks. The models convert word images into fixed-size vector embeddings, enabling similarity-based retrieval and matching.

### Key Features

-  **Multiple Architectures**: SimpleCNN (lightweight) and ResNet18 (powerful)
-  **Flexible Embedding Dimensions**: 64, 128, 256, or custom sizes
-  **Similarity Functions**: Cosine similarity and Euclidean distance
-  **Easy Integration**: Clean API for training and inference
-  **Model Persistence**: Save and load trained models
-  **Fully Tested**: Comprehensive test suite included

---

##  Quick Start

### Installation
```bash
pip install torch torchvision
```

### Basic Usage
```python
from model import EmbeddingModel, cosine_similarity

# Create model
model = EmbeddingModel(backbone='simple_cnn', embedding_dim=128)

# Prepare word image (batch_size, channels, height, width)
word_image = torch.randn(1, 1, 64, 128)  # Grayscale 64x128 image

# Get embedding
embedding = model(word_image)  # Returns (1, 128)

# Compare two word images
emb1 = model(word_image_1)
emb2 = model(word_image_2)
similarity = cosine_similarity(emb1, emb2)

print(f"Similarity: {similarity.item():.3f}")  # Value between -1 and 1
```

---

##  Architecture Options

### SimpleCNN (Recommended for Most Cases)

**Specifications:**
* 4 convolutional blocks
* ~421K parameters (embedding_dim=128)
* Fast inference (~1000 images/second on GPU)
* Good balance of speed and accuracy

**When to use:**
* Limited computational resources
* Real-time applications
* Initial experiments and baseline

**Example:**
```python
model = EmbeddingModel(
    backbone='simple_cnn',
    embedding_dim=128
)
```

### ResNet18 (For Maximum Performance)

**Specifications:**
* 18-layer ResNet architecture
* ~11M parameters (embedding_dim=128)
* Slower inference (~300 images/second on GPU)
* Higher accuracy potential

**When to use:**
* Computational resources available
* Maximum accuracy required
* Complex/difficult dataset

**Example:**
```python
model = EmbeddingModel(
    backbone='resnet18',
    embedding_dim=256,
    pretrained=True  # Use ImageNet pretrained weights
)
```

---

##  API Reference

### EmbeddingModel

Main model class for generating embeddings.
```python
model = EmbeddingModel(
    backbone='simple_cnn',    # 'simple_cnn' or 'resnet18'
    embedding_dim=128,        # Output embedding size
    pretrained=False          # Use pretrained weights (ResNet18 only)
)
```

**Methods:**

* `forward(x)` or `__call__(x)`: Compute embeddings
* `get_embedding(x)`: Alias for forward pass

**Input Format:**
* Tensor of shape `(batch_size, 1, height, width)`
* Values normalized to `[0, 1]`
* Grayscale images

**Output:**
* Tensor of shape `(batch_size, embedding_dim)`

---

### Similarity Functions

#### cosine_similarity

Compute cosine similarity between embeddings.
```python
from model import cosine_similarity

similarity = cosine_similarity(emb1, emb2)
# Returns: Values in [-1, 1] where 1 = most similar
```

**Range:**
* `1.0`: Identical direction (most similar)
* `0.0`: Orthogonal (no similarity)
* `-1.0`: Opposite direction (most dissimilar)

#### euclidean_distance

Compute L2 distance between embeddings.
```python
from model import euclidean_distance

distance = euclidean_distance(emb1, emb2)
# Returns: Values in [0, ∞) where 0 = most similar
```

#### similarity_matrix

Compute pairwise similarity matrix for batch.
```python
from model import similarity_matrix

# Batch of embeddings
embeddings = torch.randn(100, 128)

# Compute all pairwise similarities
sim_matrix = similarity_matrix(embeddings)  # Shape: (100, 100)

# Find most similar to query
query_idx = 0
similarities = sim_matrix[query_idx]
top_k = similarities.argsort(descending=True)[:10]
```

---

### Model Persistence

#### save_model

Save trained model to disk.
```python
from model import save_model

save_model(
    model,
    'checkpoints/best_model.pth',
    history=training_history,
    final_map=0.78,
    num_epochs=50
)
```

#### load_model

Load model from checkpoint.
```python
from model import load_model

model, checkpoint = load_model(
    'checkpoints/best_model.pth',
    device='cuda'
)

print(f"Model mAP: {checkpoint['final_map']}")
print(f"Trained for: {checkpoint['num_epochs']} epochs")
```

---

### Utilities

#### count_parameters

Count trainable parameters in model.
```python
from model import count_parameters

num_params = count_parameters(model)
print(f"Parameters: {num_params:,}")
```

#### get_model_info

Get detailed model information.
```python
from model import get_model_info

info = get_model_info(model)
print(info)
# {'backbone': 'simple_cnn', 'embedding_dim': 128, 'num_parameters': 421696}
```

---

##  Complete Example
```python
import torch
from model import EmbeddingModel, cosine_similarity, save_model, load_model

# 1. Create model
model = EmbeddingModel(backbone='simple_cnn', embedding_dim=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model on {device}")

# 2. Prepare data (example with random images)
batch_size = 8
word_images = torch.randn(batch_size, 1, 64, 128).to(device)

# 3. Get embeddings
model.eval()
with torch.no_grad():
    embeddings = model(word_images)

print(f"Embeddings shape: {embeddings.shape}")  # (8, 128)

# 4. Compare images
query_emb = embeddings[0:1]
similarities = cosine_similarity(
    query_emb.expand(batch_size, -1),
    embeddings
)

print(f"Similarities: {similarities}")

# 5. Find most similar
most_similar_idx = similarities.argsort(descending=True)[1]  # Skip self
print(f"Most similar to image 0: image {most_similar_idx.item()}")

# 6. Save model
save_model(
    model.cpu(),
    'my_model.pth',
    embedding_dim=128,
    description='SimpleCNN trained model'
)

# 7. Load model later
loaded_model, checkpoint = load_model('my_model.pth', device='cuda')
print(f"Loaded: {checkpoint['description']}")
```

---

##  Testing

Run the built-in test suite:
```python
# Run model.py directly
python model.py
```

Expected output:
```
======================================================================
MODEL.PY - Test Script
======================================================================

Test 1: SimpleCNN Backbone
----------------------------------------------------------------------
 Model created
  Parameters: 421,696
 Input shape: torch.Size([4, 1, 64, 128])
 Output shape: torch.Size([4, 128])

Test 2: ResNet18 Backbone
----------------------------------------------------------------------
 Model created
  Parameters: 11,301,568
 Output shape: torch.Size([4, 256])

...

======================================================================
 ALL TESTS PASSED!
======================================================================
```

---

##  Data Format Requirements

### Input Images

**Required format:**
```python
# Shape: (batch_size, channels, height, width)
# - batch_size: Number of images
# - channels: 1 (grayscale)
# - height, width: Image dimensions (e.g., 64x128)

word_images = torch.FloatTensor  # Shape: (B, 1, H, W)
```

**Preprocessing steps:**
1. Convert to grayscale
2. Resize to consistent dimensions
3. Normalize pixel values to [0, 1]

**Example preprocessing:**
```python
from PIL import Image
import torchvision.transforms as transforms

# Define transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 128)),
    transforms.ToTensor(),  # Converts to [0, 1]
])

# Apply to image
image = Image.open('word.png')
tensor = transform(image)  # Shape: (1, 64, 128)
```

---

##  Integration with Data Pipeline

For the **Data Team**, here's how to integrate this model:

### 1. DataLoader Format

Your DataLoader should return:
```python
for img1, img2, labels in train_loader:
    # img1: torch.Tensor (batch_size, 1, 64, 128)
    # img2: torch.Tensor (batch_size, 1, 64, 128)
    # labels: torch.Tensor (batch_size,) with values 0 or 1
    #   - 1 = similar pair (same word)
    #   - 0 = dissimilar pair (different words)
    pass
```

### 2. Training Integration
```python
from model import EmbeddingModel

# Create model
model = EmbeddingModel(backbone='simple_cnn', embedding_dim=128)
model = model.to(device)

# Your training loop
for epoch in range(num_epochs):
    for img1, img2, labels in train_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)
        
        # Get embeddings
        emb1 = model(img1)
        emb2 = model(img2)
        
        # Compute loss (you provide this)
        loss = your_loss_function(emb1, emb2, labels)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. Evaluation Integration
```python
from model import similarity_matrix

# Get all embeddings
all_embeddings = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        embeddings = model(images.to(device))
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

all_embeddings = torch.cat(all_embeddings)
all_labels = torch.cat(all_labels)

# Compute similarity matrix
sim_mat = similarity_matrix(all_embeddings)

# Your evaluation metrics (mAP, Precision@K, etc.)
# ...
```

---

##  Frequently Asked Questions

### Which backbone should I use?

* **SimpleCNN**: Start here. It's fast, efficient, and often sufficient.
* **ResNet18**: Use if SimpleCNN doesn't achieve target performance.

### What embedding dimension should I choose?

* **64**: Fastest, smallest models. Good for initial experiments.
* **128**: **Recommended**. Best balance of performance and efficiency.
* **256**: More capacity, slower. Use for complex datasets.

### How do I normalize images?
```python
# If your images are in [0, 255]
normalized = image / 255.0

# If using PIL/torchvision
transform = transforms.ToTensor()  # Automatically normalizes to [0, 1]
```

### Can I use this for RGB images?

The current implementation expects grayscale. For RGB:
```python
# Convert RGB to grayscale
from torchvision.transforms import Grayscale
transform = Grayscale(num_output_channels=1)
```

### How do I handle different image sizes?

Resize all images to a consistent size:
```python
from torchvision.transforms import Resize

transform = Resize((64, 128))  # Height x Width
```

### Should I use pretrained ResNet18?

* **Yes** if you have limited data (<10K images)
* **No** if you have lots of data (>50K images)
* **Try both** if unsure

---

## 🔄 Version History

* **v1.0** (November 2024): Initial release
  - SimpleCNN implementation
  - ResNet18 backbone
  - Similarity functions
  - Model persistence
  - Complete documentation

---

## 📄 License

This module is part of the Keyword Spotting project for academic purposes.

---
