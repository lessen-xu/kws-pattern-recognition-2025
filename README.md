# Keyword Spotting — Pattern Recognition Exercise 3

This repository contains our implementation of a **Keyword Spotting (KWS)** system based on the George Washington handwritten document dataset.  
The goal is to locate all occurrences of a queried word image by computing visual similarity between word crops.

The project follows a modular, multi‑member workflow so that each component can be developed independently without interfering with others.

---

## 📁 Project Structure

```
src/
  ├── A_data/          # Data extraction (word cropping, SVG parsing)
  ├── B_pairs/         # Keyword indexing & pair generation
  ├── C_model/         # Embedding model (CNN / ResNet backbone)
  ├── D_train_eval/    # Training loop & retrieval evaluation
  └── E_report/        # Final plots / report materials

data/                  # Raw dataset (ignored by Git)
results/               # Generated retrieval examples & visualizations
```

---

## 👥 Team Roles

| Member | Responsibility |
|--------|----------------|
| **Yuting Zhu** | Data extraction (A) |
| **Bole Yi** | Pair generation (B) |
| **Jules** | Embedding model (C) |
| **Songzhi Liu** | Training & evaluation (D) |
| **Lishang Xu** | GitHub management & integration |

---

## 🔄 Task Dependencies 

```
A → B → D (training)
     ↘
       C (independent)
```

- **A** outputs cropped word images + metadata  
- **B** uses A’s output to create positive/negative pairs  
- **C** builds the embedding network (independent; can be done anytime)  
- **D** needs A + B + C to train and evaluate the model  

Even though dependencies exist, **all members can write their code structure in parallel**.

---

## 🧪 Pipeline Overview

### **1. Data Extraction (A)**  
Located in `src/A_data/`

Should implement:
- Load full-page document images  
- Read polygon locations from SVG files  
- Crop each word into a normalized image  
- Map each word crop to its transcription (character‑level)  
- Save:  
  - `processed_words/` containing cropped word images  
  - Optional metadata JSON with IDs + paths + text

**Expected outputs:**
```
src/A_data/data_loader.py
processed_words/
metadata.json   (optional)
```

---

### **2. Keyword Indexing & Pair Generation (B)**  
Located in `src/B_pairs/`

Tasks:
- Load `keywords.tsv`
- Find all occurrences of each keyword in A’s processed dataset  
- Generate:  
  - Positive pairs: same word  
  - Negative pairs: different words  
- Serialize pairs (JSON recommended)

**Expected outputs:**
```
src/B_pairs/pair_generator.py
pairs_train.json
```

---

### **3. Embedding Model (C)**  
Located in `src/C_model/`

Model options:
- Small custom CNN  
- ResNet18 backbone (recommended) → projection head → embedding vector (e.g., 128‑D)

Model outputs:
- `model.py` returning embeddings  
- Cosine similarity function  

**Expected outputs:**
```
src/C_model/model.py
(optional) test_forward.py
```

---

### **4. Training & Evaluation (D)**  
Located in `src/D_train_eval/`

Tasks:
- Build PyTorch dataloader for B’s pairs  
- Train embedding model for a few epochs  
- Implement keyword retrieval:  
  - Given a query word image → rank all other words by cosine similarity  
- Save results to `results/`

**Expected outputs:**
```
src/D_train_eval/train.py
src/D_train_eval/evaluate.py
results/top_k_examples/
```

---

## 🛠 Development Workflow (Git)

To avoid conflicts:
1. **Each person works only in their own folder under `src/`**
2. Create a feature branch once:
   ```
   git checkout -b feature/A_data
   ```
3. Push changes:
   ```
   git add .
   git commit -m "Add data loader"
   git push -u origin feature/A_data
   ```
4. Open a Pull Request  
5. LS reviews & merges into `main`

---

## ▶️ Running the Project (Example)

After everything is integrated:

### **Training**
```
python src/D_train_eval/train.py
```

### **Evaluation / Retrieval**
```
python src/D_train_eval/evaluate.py --query_id 270-05-07
```

Outputs (example):
```
results/
  topk_270-05-07.png
  embeddings.npy
  retrieval_scores.json
```

---

© 2025 · University of Fribourg · Pattern Recognition
