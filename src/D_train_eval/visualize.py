import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms

# === 1. Base Path Settings ===
current_dir = Path(__file__).resolve().parent
src_root = current_dir.parent
sys.path.append(str(src_root))


try:
    from evaluate import load_validation_data, MODEL_PATH, EmbeddingModel
except ImportError:
    sys.path.append(str(current_dir))
    from evaluate import load_validation_data, MODEL_PATH, EmbeddingModel

def visualize_simple():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualization device: {device}")

    # === 2. Load Model & Data ===
    if not MODEL_PATH.exists():
        print("Model not found. Run train.py first.")
        return

    model = EmbeddingModel(backbone='simple_cnn', embedding_dim=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Loading data (only the first 500 records are displayed for demonstration purposes)
    dataset = load_validation_data()[:500]
    if not dataset: return

    transform = transforms.Compose([transforms.Resize((64, 128)), transforms.ToTensor()])

    # === 3. Feature Extraction  ===
    print("Extracting features...")
    feats, images, labels = [], [], []

    with torch.no_grad():
        for item in dataset:
            try:
                pil_img = Image.open(item['path']).convert('L')
                tensor_img = transform(pil_img).unsqueeze(0).to(device)
                
                feats.append(model(tensor_img).cpu())
                images.append(pil_img)
                labels.append(item['text'])
            except: continue

    feats = torch.cat(feats) # (N, 128)
    # Normalization for cosine similarity
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    # === 4. Randomly select 5 queries and plot them. ===
    indices = np.random.choice(len(feats), 5, replace=False)
    sim_matrix = torch.mm(feats[indices], feats.t()) # (5, N)

    fig, axes = plt.subplots(5, 6, figsize=(12, 8))
    fig.suptitle("KWS Retrieval Results (Query vs Top-5 Matches)", fontsize=16)
    
    # Set column headers
    cols = ["Query Image"] + [f"Rank {i+1}" for i in range(5)]
    for ax, col in zip(axes[0], cols): ax.set_title(col, fontweight='bold')

    for i, row_idx in enumerate(indices):
        #  Query
        axes[i, 0].imshow(images[row_idx], cmap='gray')
        axes[i, 0].set_ylabel(f"Query: {labels[row_idx]}", rotation=0, labelpad=40, fontsize=10)
        
        # Fine Top-6 (Except self)
        scores, top_k = torch.topk(sim_matrix[i], k=6)
        
        col = 1
        for idx in top_k:
            idx = idx.item()
            if idx == row_idx: continue # 
            if col > 5: break
            
            # 
            axes[i, col].imshow(images[idx], cmap='gray')
            
            # Green indicates a correct match; red indicates an error.
            color = 'green' if labels[idx] == labels[row_idx] else 'red'
            axes[i, col].set_xlabel(f"{labels[idx]}\n{scores[col-1]:.3f}", color=color, fontsize=9)
            
            # Remove axis tick marks
            axes[i, col].set_xticks([])
            axes[i, col].set_yticks([])
            col += 1
            
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

    plt.tight_layout()
    save_path = src_root / "results" / "retrieval_viz.png"
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_simple()