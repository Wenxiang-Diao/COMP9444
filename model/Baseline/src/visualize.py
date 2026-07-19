from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASELINE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASELINE_ROOT / "results"


def plot_loss(csv_file=RESULTS_DIR / "cnn_metrics.csv"):
    data = pd.read_csv(csv_file)
    epochs = data["Epoch"]
    train_loss = data["Train Loss"]
    val_loss = data["Val Loss"]

    RESULTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_vs_validation_loss.png")
    plt.show()
    plt.close()


def plot_accuracy(csv_file=RESULTS_DIR / "cnn_metrics.csv"):
    data = pd.read_csv(csv_file)
    epochs = data["Epoch"]
    train_acc = data["Train Accuracy"]
    val_acc = data["Val Accuracy"]

    RESULTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_vs_validation_accuracy.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_loss()
    plot_accuracy()
