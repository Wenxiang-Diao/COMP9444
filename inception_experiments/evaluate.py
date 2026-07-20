from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from .dataset import make_loader
from .metrics import PredictionAccumulator
from .models import RAW_MODELS, ModelConfig, build_model
from .runtime import runtime_info, select_device, synchronize, write_json
from .train import compute_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Inception experiment checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("devide_dataset/data"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--benchmark-batches", type=int, default=50)
    return parser.parse_args()


def save_confusion(values_true, values_pred, names, prefix: Path) -> None:
    matrix = confusion_matrix(values_true, values_pred, labels=range(len(names)))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(prefix.with_suffix(".csv"))
    figure_size = max(8, min(24, len(names) * 0.55))
    fig, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(prefix.stem.replace("_", " ").title())
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    axis.set_yticks(range(len(names)), names, fontsize=6)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=180)
    plt.close(fig)


def benchmark(model, loader, device, maximum_batches: int, amp: bool, amp_dtype: str) -> dict[str, float]:
    durations = []
    images_seen = 0
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= maximum_batches + 5:
                break
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            synchronize(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16,
                enabled=amp,
            ):
                model(images)
            synchronize(device)
            duration = time.perf_counter() - started
            if index >= 5:
                durations.append(duration)
                images_seen += int(images.shape[0])
    total = sum(durations)
    return {
        "timed_batches": len(durations),
        "images": images_seen,
        "seconds": total,
        "images_per_second": images_seen / max(total, 1e-12),
        "mean_milliseconds_per_image": 1000 * total / max(images_seen, 1),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_maps = checkpoint["class_maps"]
    model_config = dict(checkpoint["model_config"])
    model_config["weights"] = "none"
    config = ModelConfig(**model_config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output_dir = args.output_dir or args.checkpoint.parent / f"{args.split}_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(
        args.data_dir / f"{args.split}_split.csv",
        args.batch_size,
        False,
        args.num_workers,
        device,
        int(checkpoint["training_config"]["seed"]),
    )
    criterion = nn.CrossEntropyLoss()
    loss_args = argparse.Namespace(**checkpoint["training_config"])
    amp = bool(getattr(loss_args, "amp", False))
    amp_dtype = str(getattr(loss_args, "amp_dtype", "bf16"))
    accumulator = PredictionAccumulator(config.name, class_maps)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            for key in ("image", "label_raw", "label_plant", "label_disease"):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16,
                enabled=amp,
            ):
                outputs = model(batch["image"])
                loss = compute_loss(config.name, outputs, batch, criterion, loss_args)
            accumulator.update(float(loss.item()), batch, outputs)
    metrics = accumulator.compute()
    metrics["evaluation_seconds"] = time.perf_counter() - started
    write_json(output_dir / "metrics.json", metrics)

    reports = {
        "plant": classification_report(
            accumulator.plant_true,
            accumulator.plant_pred,
            labels=range(config.num_plants),
            target_names=[class_maps["idx_to_plant"][str(i)] for i in range(config.num_plants)],
            zero_division=0,
            output_dict=True,
        ),
        "disease": classification_report(
            accumulator.disease_true,
            accumulator.disease_pred,
            labels=range(config.num_diseases),
            target_names=[class_maps["idx_to_disease"][str(i)] for i in range(config.num_diseases)],
            zero_division=0,
            output_dict=True,
        ),
    }
    if config.name in RAW_MODELS:
        reports["raw"] = classification_report(
            accumulator.raw_true,
            accumulator.raw_pred,
            labels=range(config.num_raw),
            target_names=[class_maps["idx_to_raw"][str(i)] for i in range(config.num_raw)],
            zero_division=0,
            output_dict=True,
        )
        save_confusion(
            accumulator.raw_true,
            accumulator.raw_pred,
            [class_maps["idx_to_raw"][str(i)] for i in range(config.num_raw)],
            output_dir / "raw_confusion_matrix",
        )
    save_confusion(
        accumulator.plant_true,
        accumulator.plant_pred,
        [class_maps["idx_to_plant"][str(i)] for i in range(config.num_plants)],
        output_dir / "plant_confusion_matrix",
    )
    save_confusion(
        accumulator.disease_true,
        accumulator.disease_pred,
        [class_maps["idx_to_disease"][str(i)] for i in range(config.num_diseases)],
        output_dir / "disease_confusion_matrix",
    )
    write_json(output_dir / "classification_reports.json", reports)
    efficiency = {
        "runtime": runtime_info(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "inference": benchmark(model, loader, device, args.benchmark_batches, amp, amp_dtype),
    }
    write_json(output_dir / "efficiency.json", efficiency)
    print(json.dumps({"metrics": metrics, "efficiency": efficiency}, indent=2))


if __name__ == "__main__":
    main()
