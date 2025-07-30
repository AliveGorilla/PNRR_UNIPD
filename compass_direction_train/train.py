import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import ResNet18Keypoints
from dataset import KeypointDataset

# Path and training configuration
DATA_DIR = 'data'
IMAGE_DIR = os.path.join(DATA_DIR, 'image')
JSON_FILE = os.path.join(DATA_DIR, 'annotations.json')
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
IMG_SIZE = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_losses(train_losses, val_losses, name):
    """
    Save a plot of training and validation losses per epoch.

    Args:
        train_losses (list of float): Training losses.
        val_losses (list of float): Validation losses.
        name (str): Output file prefix.
    """
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('MSE Loss')
    plt.savefig(f'{name}_loss.png')
    plt.close()

def plot_scatter(preds, targets, name):
    """
    Save a scatter plot comparing predicted and true keypoint values.

    Args:
        preds (list): Model predictions.
        targets (list): True keypoints.
        name (str): Output file prefix.
    """
    preds = np.array(preds)
    targets = np.array(targets)
    plt.figure(figsize=(8, 8))
    plt.scatter(targets[:,0], preds[:,0], alpha=0.5, label='Center X')
    plt.scatter(targets[:,1], preds[:,1], alpha=0.5, label='Center Y')
    plt.scatter(targets[:,2], preds[:,2], alpha=0.5, label='North X')
    plt.scatter(targets[:,3], preds[:,3], alpha=0.5, label='North Y')
    plt.plot([0,1],[0,1],'k--', alpha=0.4)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Predicted vs True Keypoints')
    plt.savefig(f'{name}_scatter.png')
    plt.close()

# Image transformation pipeline for training and validation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_val_split(dataset, val_ratio=0.15):
    """
    Split a dataset into training and validation subsets.

    Args:
        dataset: The dataset to split.
        val_ratio (float): Proportion to use for validation.

    Returns:
        train_ds, val_ds: Training and validation subsets.
    """
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    n_val = int(val_ratio * len(dataset))
    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]
    from torch.utils.data import Subset
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)

def main():
    """
    Main training loop for keypoint regression.

    Loads the dataset, splits into train/val, trains the model, evaluates,
    saves the best weights, and creates plots for loss and predictions.
    """
    dataset = KeypointDataset(JSON_FILE, IMAGE_DIR, transform)
    train_ds, val_ds = train_val_split(dataset)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model = ResNet18Keypoints().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, keyps, _ in tqdm(train_loader, desc=f'Train Epoch {epoch+1}'):
            imgs, keyps = imgs.to(device), keyps.to(device)
            out = model(imgs)
            loss = criterion(out, keyps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        preds_all, targets_all = [], []
        with torch.no_grad():
            for imgs, keyps, _ in val_loader:
                imgs, keyps = imgs.to(device), keyps.to(device)
                out = model(imgs)
                loss = criterion(out, keyps)
                val_loss += loss.item() * imgs.size(0)
                preds_all.extend(out.cpu().numpy())
                targets_all.extend(keyps.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Save loss and scatter plots
    plot_losses(train_losses, val_losses, 'training')
    plot_scatter(preds_all, targets_all, 'scatter')

if __name__ == '__main__':
    main()