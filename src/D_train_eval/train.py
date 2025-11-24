import sys
import os
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

current_dir = Path(__file__).resolve().parent
src_root = current_dir.parent
sys.path.append(str(src_root))

try:
    from C_model.model import EmbeddingModel
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    exit()

IMG_DIR = src_root / "A_data" / "cropped_words"
PAIRS_FILE = src_root / "B_pairs" / "pairs_train.json"
RESULTS_DIR = current_dir.parent / "results"

# ============================================================================

class GWDataset(Dataset):
    def __init__(self,  pairs_file,imd_dir, transform=None):
        with open(pairs_file, 'r') as f:
            self.pairs = json.load(f)
        self.imd_dir = imd_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        name1 = Path(pair['img1']).name
        name2 = Path(pair['img2']).name
        
        path1 = self.imd_dir / name1
        path2 = self.imd_dir / name2

        if not path1.exists():
            raise FileNotFoundError(f"Image not found: {path1}")
        if not path2.exists():
            raise FileNotFoundError(f"Image not found: {path2}")
        

        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')

        label = torch.tensor(pair['label'], dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                    (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
def train():

    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 128


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    print(f"Using device: {device}")

    if not PAIRS_FILE.exists():
        raise FileNotFoundError(f"Pairs file not found: {PAIRS_FILE}")

    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor()
        ])

        #Loading data
    dataset = GWDataset(PAIRS_FILE, IMG_DIR, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} pairs.")
        
    model = EmbeddingModel(backbone='simple_cnn', embedding_dim=EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print("Start Training...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULTS_DIR / "embedding_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()




        
        