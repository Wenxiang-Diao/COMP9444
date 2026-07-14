from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def update_counts(
    counts: Dict[str, int],
    outputs: Dict[str, torch.Tensor],
    plant_labels: torch.Tensor,
    disease_labels: torch.Tensor,
) -> None:
    plant_pred = outputs["plant_output"].argmax(dim=1)
    disease_pred = outputs["disease_output"].argmax(dim=1)

    counts["n"] += int(plant_labels.numel())
    counts["plant_correct"] += int((plant_pred == plant_labels).sum().item())
    counts["disease_correct"] += int((disease_pred == disease_labels).sum().item())
    counts["both_correct"] += int(
        ((plant_pred == plant_labels) & (disease_pred == disease_labels)).sum().item()
    )


def counts_to_metrics(total_loss: float, counts: Dict[str, int]) -> Dict[str, float]:
    n = max(counts["n"], 1)
    return {
        "loss": total_loss / n,
        "plant_acc": counts["plant_correct"] / n,
        "disease_acc": counts["disease_correct"] / n,
        "both_acc": counts["both_correct"] / n,
    }


def empty_counts() -> Dict[str, int]:
    return {
        "n": 0,
        "plant_correct": 0,
        "disease_correct": 0,
        "both_correct": 0,
    }

