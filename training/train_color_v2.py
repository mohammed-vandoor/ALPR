"""
Train vehicle colour classifier v2:
  - EfficientNet-B0 (better accuracy than MobileNetV3)
  - Merges VCoR + seebicb + dataclusterlabs datasets
  - CCTV-optimised augmentation (brightness/contrast/blur)
  - Outputs: color_classifier.pth + color_classes.json

Usage:
    source venv/bin/activate
    python train_color_v2.py
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# ── Config ────────────────────────────────────────────────────────────────────
MERGED_DIR    = "data/merged"
VCOR_DIR      = "data/vcor"
SEEBICB_DIR   = "data/seebicb"
DATACL_DIR    = "data/dataclusterlabs/Vehicle Color Detection Dataset"
OUTPUT_MODEL  = "color_classifier.pth"
OUTPUT_CLASSES = "color_classes.json"
EPOCHS        = 12
BATCH_SIZE    = 32
LR            = 5e-4
IMG_SIZE      = 224

# Canonical colour names — map all dataset variants to these
COLOUR_MAP = {
    "black": "black",   "black_auto": "black",
    "white": "white",   "white_auto": "white",
    "silver": "silver", "silver_auto": "silver",
    "grey": "grey",     "gray": "grey",     "grey_auto": "grey",
    "blue": "blue",     "blue_auto": "blue",
    "red": "red",       "red_auto": "red",
    "green": "green",   "green_auto": "green",
    "yellow": "yellow", "yellow_auto": "yellow",
    "orange": "orange", "orange_auto": "orange",
    "brown": "brown",   "brown_auto": "brown",
    "beige": "beige",   "beige_auto": "beige",
    "purple": "purple", "purple_auto": "purple",
    "gold": "gold",     "gold_auto": "gold",
    "pink": "pink",     "pink_auto": "pink",
    "tan": "tan",       "tan_auto": "tan",
    "cyan": "blue",     "cyan_auto": "blue",
}

# ── Step 1: Merge datasets ────────────────────────────────────────────────────
def merge_datasets():
    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)
    os.makedirs(MERGED_DIR)

    total = 0

    # — VCoR & seebicb: already folder-structured (train/val/test) —
    for ds_root in [VCOR_DIR, SEEBICB_DIR]:
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(ds_root, split)
            if not os.path.isdir(split_dir):
                continue
            for colour_folder in os.listdir(split_dir):
                src = os.path.join(split_dir, colour_folder)
                colour = COLOUR_MAP.get(colour_folder.lower())
                if not colour or not os.path.isdir(src):
                    continue
                dst = os.path.join(MERGED_DIR, colour)
                os.makedirs(dst, exist_ok=True)
                for img_file in os.listdir(src):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        src_path = os.path.join(src, img_file)
                        dst_name = f"{ds_root.split('/')[-1]}_{split}_{img_file}"
                        shutil.copy2(src_path, os.path.join(dst, dst_name))
                        total += 1

    # — dataclusterlabs: XML annotations with bounding boxes —
    xml_files = [f for f in os.listdir(DATACL_DIR) if f.endswith(".xml")]
    for xml_file in xml_files:
        try:
            tree = ET.parse(os.path.join(DATACL_DIR, xml_file))
            root = tree.getroot()
            img_name = os.path.basename(xml_file.replace(".xml", ".jpg"))
            img_path = os.path.join(DATACL_DIR, img_name)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            for i, obj in enumerate(root.findall("object")):
                raw_label = obj.find("name").text.strip().lower()
                colour = COLOUR_MAP.get(raw_label)
                if not colour:
                    continue
                bb = obj.find("bndbox")
                x1 = int(float(bb.find("xmin").text))
                y1 = int(float(bb.find("ymin").text))
                x2 = int(float(bb.find("xmax").text))
                y2 = int(float(bb.find("ymax").text))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue
                dst = os.path.join(MERGED_DIR, colour)
                os.makedirs(dst, exist_ok=True)
                out_name = f"datacl_{xml_file[:-4]}_{i}.jpg"
                cv2.imwrite(os.path.join(dst, out_name), crop)
                total += 1
        except Exception as e:
            print(f"  Skip {xml_file}: {e}")

    # Summary
    print(f"\nMerged dataset: {total} images")
    for colour in sorted(os.listdir(MERGED_DIR)):
        n = len(os.listdir(os.path.join(MERGED_DIR, colour)))
        print(f"  {colour:12s}: {n}")
    return total


# ── Step 2: Train EfficientNet-B0 ─────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        # CCTV-specific augmentation: simulate different lighting/blur
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds = ImageFolder(MERGED_DIR)
    n_val = max(1, int(0.12 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    train_ds.dataset.transform = train_tfm
    val_ds.dataset.transform   = val_tfm

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(full_ds.classes)
    print(f"Classes ({num_classes}): {full_ds.classes}")

    # EfficientNet-B0 — better than MobileNetV3 for subtle colour differences
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
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
                "arch": "efficientnet_b0",
            }, OUTPUT_MODEL)
            print(f"  Saved best model ({acc:.1f}%)")

    with open(OUTPUT_CLASSES, "w") as f:
        json.dump(full_ds.classes, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_acc:.1f}%")
    print(f"Model: {OUTPUT_MODEL}  |  Classes: {OUTPUT_CLASSES}")


if __name__ == "__main__":
    print("Step 1: Merging datasets...")
    merge_datasets()
    print("\nStep 2: Training EfficientNet-B0...")
    train()
