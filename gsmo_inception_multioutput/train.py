from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from dataset import make_dataloader
from metrics import counts_to_metrics, empty_counts, update_counts
from model import build_model


DEFAULT_DATA_DIR = Path("devide_dataset/data")
DEFAULT_OUTPUT_DIR = Path("gsmo_inception_multioutput/runs/with_aug")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(
    criterion: nn.Module,
    outputs: Dict[str, torch.Tensor],
    plant_labels: torch.Tensor,
    disease_labels: torch.Tensor,
    weights: Tuple[float, float, float, float],
) -> torch.Tensor:
    w_p, w_d, w_p_t, w_d_t = weights
    return (
        w_p * criterion(outputs["plant_output"], plant_labels)
        + w_d * criterion(outputs["disease_output"], disease_labels)
        + w_p_t * criterion(outputs["plant_output_t"], plant_labels)
        + w_d_t * criterion(outputs["disease_output_t"], disease_labels)
    )


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    weights: Tuple[float, float, float, float],
    optimizer=None,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    counts = empty_counts()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        plant_labels = batch["label_plant"].to(device, non_blocking=True)
        disease_labels = batch["label_disease"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = compute_loss(
                criterion,
                outputs,
                plant_labels,
                disease_labels,
                weights,
            )
            if train:
                loss.backward()
                optimizer.step()

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        update_counts(counts, outputs, plant_labels, disease_labels)

    return counts_to_metrics(total_loss, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GSMo-CNN with InceptionV3 on leaf disease splits."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=["imagenet", "none", "random"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss-plant", type=float, default=0.4)
    parser.add_argument("--loss-disease", type=float, default=0.5)
    parser.add_argument("--loss-plant-t", type=float, default=0.1)
    parser.add_argument("--loss-disease-t", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_maps_path = args.data_dir / "class_maps.json"
    with open(class_maps_path, "r", encoding="utf-8") as f:
        class_maps = json.load(f)

    num_plants = int(class_maps["num_classes"]["plant"])
    num_diseases = int(class_maps["num_classes"]["disease"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = make_dataloader(
        args.data_dir / "train_split.csv",
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        args.data_dir / "val_split.csv",
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        num_plants=num_plants,
        num_diseases=num_diseases,
        dropout=args.dropout,
        weights=args.weights,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_weights = (
        args.loss_plant,
        args.loss_disease,
        args.loss_plant_t,
        args.loss_disease_t,
    )

    history = []
    best_val_both = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            loss_weights,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            loss_weights,
            optimizer=None,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["both_acc"] > best_val_both:
            best_val_both = val_metrics["both_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_maps": class_maps,
                    "args": vars(args),
                    "best_val_both_acc": best_val_both,
                },
                args.output_dir / "best_model.pt",
            )

        with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best validation both_acc: {best_val_both:.4f}")
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

