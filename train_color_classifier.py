"""
Train a vehicle color classifier using MobileNetV3-Small.
Uses the VCoR dataset from HuggingFace (vehicle color recognition).

Usage:
    pip install datasets torchvision torch
    python train_color_classifier.py

Output:
    color_classifier.pth  — saved model (~10MB)
    color_classes.json    — class index mapping
"""

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATASET      = "tanganke/stanford_cars"   # swap for your own dataset
EPOCHS       = 10
BATCH_SIZE   = 32
LR           = 1e-3
IMG_SIZE     = 224
OUTPUT_MODEL = "color_classifier.pth"
OUTPUT_CLASSES = "color_classes.json"

# Colour labels — edit to match your dataset's label names
COLOR_CLASSES = [
    "Black", "White", "Silver", "Grey", "Red",
    "Blue", "Green", "Yellow", "Orange", "Brown",
    "Purple", "Gold", "Beige",
]

# ── Dataset ───────────────────────────────────────────────────────────────────
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ColorDataset(torch.utils.data.Dataset):
    """
    Wraps any image folder structure where subfolders = color class names.
    Replace this with your own dataset loader if needed.

    Expected folder structure:
        data/
          Black/  img1.jpg img2.jpg ...
          White/  img1.jpg ...
          ...
    """
    def __init__(self, root, transform=None):
        from torchvision.datasets import ImageFolder
        self.ds = ImageFolder(root, transform=transform)
        self.classes = self.ds.classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Replace final classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train(data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_ds = ColorDataset(data_root, transform=train_tfm)
    n_val = max(1, int(0.15 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = val_tfm

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    num_classes = len(full_ds.classes)
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS}  loss={running_loss/len(train_loader):.3f}  val_acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": full_ds.classes,
                "num_classes": num_classes,
            }, OUTPUT_MODEL)
            print(f"  Saved best model ({acc:.1f}%)")

    # Save class mapping
    with open(OUTPUT_CLASSES, "w") as f:
        json.dump(full_ds.classes, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {OUTPUT_MODEL}")
    print(f"Classes saved to: {OUTPUT_CLASSES}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_color_classifier.py <path/to/data_folder>")
        print()
        print("Data folder should have subfolders named by colour:")
        print("  data/Black/  data/White/  data/Silver/  ...")
        print()
        print("Recommended datasets:")
        print("  - VeRi-776:  https://github.com/JDAI-CV/VeRi")
        print("  - CompCars:  http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/")
        print("  - Your own:  50+ images per colour class is enough")
        sys.exit(0)
    train(sys.argv[1])
