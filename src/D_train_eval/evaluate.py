import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. Path Settings
# ==========================================
# 
current_dir = Path(__file__).resolve().parent
# Get the src directory (parent directory)
src_root = current_dir.parent
# Add src to Python's search path
sys.path.append(str(src_root))

# Attempt to import the model
try:
    from C_model.model import EmbeddingModel
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# ==========================================
# 2. Configuration Path
# ==========================================
IMG_DIR = src_root / "A_data" / "cropped_words"
METADATA_PATH = src_root / "A_data" / "words_metadata.tsv"
TRANSCRIPTION_PATH = src_root / "KWS" / "transcription.tsv"
MODEL_PATH = src_root / "results" / "embedding_model.pth"


if not MODEL_PATH.exists():
    MODEL_PATH = src_root.parent / "results" / "embedding_model.pth"

# ==========================================
# 3. helper function
# ==========================================
def load_validation_data():
    """Load the validation set data list"""
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file not found {METADATA_PATH}")
        return []

    df_meta = pd.read_csv(METADATA_PATH, sep='\t')
    
    trans_dict = {}
    if TRANSCRIPTION_PATH.exists():
        with open(TRANSCRIPTION_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    trans_dict[parts[0]] = parts[1].lower()

    data = []
    for _, row in df_meta.iterrows():
        wid = str(row['word_id'])
        # Ensure path concatenation is correct
        img_name = Path(row['path']).name
        img_path = IMG_DIR / img_name
        
        if img_path.exists():
            text = trans_dict.get(wid, "unknown")
            data.append({
                'path': img_path,
                'text': text,
                'id': wid
            })
    return data

# ==========================================
# 4. Main evaluation function
# ==========================================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    if not MODEL_PATH.exists():
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        print("请先运行 train.py 生成模型。")
        return

    # Loading Model
    print(f"Loading model from {MODEL_PATH}...")
    model = EmbeddingModel(backbone='simple_cnn', embedding_dim=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Perpare Data
    dataset = load_validation_data()
    if not dataset:
        print("没有找到验证数据。")
        return
    
    # Limit the number of demonstrations
    if len(dataset) > 1000:
        dataset = dataset[:1000]

    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor()
    ])

    print("Feature extraction...")
    embeddings = []
    labels = []

    with torch.no_grad():
        for item in dataset:
            try:
                img = Image.open(item['path']).convert('L')
                img = transform(img).unsqueeze(0).to(device)
                emb = model(img)
                embeddings.append(emb.cpu())
                labels.append(item['text'])
            except Exception:
                continue

    if not embeddings:
        print("Failed to Feature extraction ")
        return

    feats = torch.cat(embeddings, dim=0)
    feats = F.normalize(feats, p=2, dim=1)

    # Randomly select 5 query terms
    num_queries = 5
    indices = np.random.choice(len(feats), num_queries, replace=False)
    
    # Calculate similarity
    query_feats = feats[indices]
    sim_matrix = torch.mm(query_feats, feats.t())

    print("\n" + "="*50)
    print("Search Results Demo")
    print("="*50)

    for i in range(num_queries):
        q_idx = indices[i]
        query_text = labels[q_idx]
        print(f"\nQuery [{i+1}]: '{query_text}'")
        
        # Get the Top 6 (including yourself)
        scores, top_indices = torch.topk(sim_matrix[i], k=6)
        
        for rank, idx in enumerate(top_indices):
            idx = idx.item()
            if idx == q_idx: continue
            
            match_text = labels[idx]
            score = scores[rank].item()
            is_match = (match_text == query_text)
            marker = "✅" if is_match else "❌"
            
            print(f"  {marker} Rank {rank}: {match_text} ({score:.4f})")

    print("\Evaluation completed.")

if __name__ == "__main__":
    evaluate()