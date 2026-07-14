from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from dataset import make_dataloader
from metrics import counts_to_metrics, empty_counts, update_counts
from model import build_model
from train import compute_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GSMo-CNN checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("devide_dataset/data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("gsmo_inception_multioutput/eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_maps = checkpoint["class_maps"]
    train_args = checkpoint["args"]

    model = build_model(
        num_plants=int(class_maps["num_classes"]["plant"]),
        num_diseases=int(class_maps["num_classes"]["disease"]),
        dropout=float(train_args.get("dropout", 0.2)),
        weights="none",
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = make_dataloader(
        args.data_dir / f"{args.split}_split.csv",
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    loss_weights = (
        float(train_args.get("loss_plant", 0.4)),
        float(train_args.get("loss_disease", 0.5)),
        float(train_args.get("loss_plant_t", 0.1)),
        float(train_args.get("loss_disease_t", 0.1)),
    )

    counts = empty_counts()
    total_loss = 0.0
    plant_true, plant_pred = [], []
    disease_true, disease_pred = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            plant_labels = batch["label_plant"].to(device)
            disease_labels = batch["label_disease"].to(device)
            outputs = model(images)
            loss = compute_loss(
                criterion,
                outputs,
                plant_labels,
                disease_labels,
                loss_weights,
            )

            total_loss += float(loss.item()) * int(images.size(0))
            update_counts(counts, outputs, plant_labels, disease_labels)
            plant_true.extend(plant_labels.cpu().tolist())
            disease_true.extend(disease_labels.cpu().tolist())
            plant_pred.extend(outputs["plant_output"].argmax(dim=1).cpu().tolist())
            disease_pred.extend(outputs["disease_output"].argmax(dim=1).cpu().tolist())

    metrics = counts_to_metrics(total_loss, counts)
    with open(args.output_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plant_names = [class_maps["idx_to_plant"][str(i)] for i in range(int(class_maps["num_classes"]["plant"]))]
    disease_names = [
        class_maps["idx_to_disease"][str(i)]
        for i in range(int(class_maps["num_classes"]["disease"]))
    ]

    reports = {
        "plant": classification_report(
            plant_true,
            plant_pred,
            target_names=plant_names,
            zero_division=0,
            output_dict=True,
        ),
        "disease": classification_report(
            disease_true,
            disease_pred,
            target_names=disease_names,
            zero_division=0,
            output_dict=True,
        ),
    }
    with open(args.output_dir / f"{args.split}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    pd.DataFrame(confusion_matrix(plant_true, plant_pred)).to_csv(
        args.output_dir / f"{args.split}_plant_confusion_matrix.csv",
        index=False,
    )
    pd.DataFrame(confusion_matrix(disease_true, disease_pred)).to_csv(
        args.output_dir / f"{args.split}_disease_confusion_matrix.csv",
        index=False,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation files to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

