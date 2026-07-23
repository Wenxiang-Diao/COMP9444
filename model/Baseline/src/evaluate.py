import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import seaborn as sns
except ImportError:
    sns = None

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


def load_class_names():
    with open(SPLIT_DIR / "class_maps.json", encoding="utf-8") as f:
        class_maps = json.load(f)
    return [class_maps["idx_to_raw"][str(i)] for i in range(class_maps["num_classes"]["raw"])]


def build_test_loader():
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = LeafCSVDataset(SPLIT_DIR / "test_split.csv", transform=eval_transform)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    return accuracy, conf_matrix, precision, recall, macro_f1


def plot_confusion_matrix(conf_matrix, class_names, title="Confusion Matrix"):
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(20, 20))
    if sns is not None:
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
    plt.title(title)
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_confusion_matrix.png")
    plt.show()


def main():
    device = get_device()
    class_names = load_class_names()
    test_loader = build_test_loader()

    model = CNNModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_leaf.pth", map_location=device, weights_only=True))
    model.to(device)

    accuracy, conf_matrix, precision, recall, macro_f1 = evaluate_model(model, test_loader, device)
    print(
        f"Accuracy: {accuracy:.4f}%, Macro Precision: {precision:.4f}, "
        f"Macro Recall: {recall:.4f}, Macro F1: {macro_f1:.4f}"
    )
    plot_confusion_matrix(conf_matrix, class_names)


if __name__ == "__main__":
    main()
