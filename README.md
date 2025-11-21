# Keyword Spotting — Pattern Recognition Group Exercise 3  
### Group 14

This repository contains our implementation for the Keyword Spotting task based on the George Washington dataset.  
The pipeline includes data extraction, keyword indexing, embedding model design, training, and retrieval evaluation.

---

## 👥 Team & Roles

| Member | Role | Responsibility |
|--------|------|----------------|
| **Yuting Zhu** | A | Data extraction (cropping words, reading SVG polygons, mapping transcriptions) |
| **Bole Yi** | B | Keyword indexing & positive/negative pair generation |
| **Jules** | C | Embedding model (CNN/ResNet-based) |
| **Songzhi Liu** | D | Training loop, evaluation pipeline, retrieval visualization |
| **Lishang Xu** | Integration | GitHub workflow, merging, repository structure, final report |

---

## 📁 Project Structure

```
src/
  ├── A_data/          # A: data extraction scripts
  ├── B_pairs/         # B: pair generation
  ├── C_model/         # C: embedding model
  ├── D_train_eval/    # D: training/evaluation
  └── E_report/        # Final report materials

data/                  # Raw dataset (ignored by Git)
results/               # Retrieval examples & plots
```

---

## 🔄 Task Dependencies

```
A → B → D (training)
     ↘
       C (model, independent)
```

- **A must finish first** (cropped words + metadata)  
- **B uses A’s output to generate pairs**  
- **C can be done anytime, independent**  
- **D needs A + B + C to train/evaluate**  
- All members work inside their own folders to avoid conflicts.

---

## 🛠 Git Workflow

1. Clone repo  
2. Work only inside your own folder  
3. Create your feature branch:  
   `git checkout -b feature/A_data` (example)  
4. Push your changes  
5. Open a Pull Request  
6. LS will review + merge into main  

---

© 2025 · University of Fribourg · Pattern Recognition  
