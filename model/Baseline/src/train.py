import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from .model import CNNModel
except ImportError:
    from model import CNNModel


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "devide_dataset" / "data"
MODELS_DIR = BASELINE_ROOT / "models"
RESULTS_DIR = BASELINE_ROOT / "results"

IMAGE_SIZE = 299
BATCH_SIZE = 128
NUM_EPOCHS = 20
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LeafCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label_raw"])
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(batch_size=BATCH_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = LeafCSVDataset(SPLIT_DIR / "train_split.csv", transform=train_transform)
    val_dataset = LeafCSVDataset(SPLIT_DIR / "val_split.csv", transform=eval_transform)
    test_dataset = LeafCSVDataset(SPLIT_DIR / "test_split.csv", transform=eval_transform)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_num_classes():
    with open(SPLIT_DIR / "class_maps.json", encoding="utf-8") as f:
        class_maps = json.load(f)
    return class_maps["num_classes"]["raw"]


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_accuracy = 100 * correct / total
    return running_loss / len(train_loader.dataset), train_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    return val_loss / len(val_loader.dataset), val_accuracy


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")

    model = CNNModel(num_classes=load_num_classes()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metrics_path = RESULTS_DIR / "cnn_metrics.csv"

    with open(metrics_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])

        best_val_loss = float("inf")
        best_epoch = 0
        model_path = MODELS_DIR / "cnn_leaf.pth"

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_acc = test(model, test_loader, device)
    print(f"Training metrics saved to: {metrics_path}")
    print(f"Best model saved to: {model_path} (epoch {best_epoch})")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
