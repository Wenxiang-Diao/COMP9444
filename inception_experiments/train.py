from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .dataset import make_loader
from .metrics import PredictionAccumulator
from .models import MODEL_NAMES, RAW_MODELS, ModelConfig, build_model
from .runtime import runtime_info, select_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shared InceptionV3 experiment.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("devide_dataset/data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weights", choices=("imagenet", "none", "random"), default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--loss-plant", type=float, default=0.4)
    parser.add_argument("--loss-disease", type=float, default=0.5)
    parser.add_argument("--loss-plant-t", type=float, default=0.1)
    parser.add_argument("--loss-disease-t", type=float, default=0.1)
    return parser.parse_args()


def compute_loss(model_name: str, outputs: dict, batch: dict, criterion: nn.Module, args) -> torch.Tensor:
    if model_name in RAW_MODELS:
        return criterion(outputs["raw"], batch["label_raw"])
    plant_loss = criterion(outputs["plant"], batch["label_plant"])
    disease_loss = criterion(outputs["disease"], batch["label_disease"])
    if model_name == "plain_multi":
        return plant_loss + disease_loss
    return (
        args.loss_plant * plant_loss
        + args.loss_disease * disease_loss
        + args.loss_plant_t * criterion(outputs["plant_t"], batch["label_plant"])
        + args.loss_disease_t * criterion(outputs["disease_t"], batch["label_disease"])
    )


def run_epoch(
    model, loader, criterion, device, class_maps, args, optimizer=None,
    scaler=None, max_batches=None, phase="train", epoch=0,
):
    training = optimizer is not None
    model.train(training)
    accumulator = PredictionAccumulator(args.model, class_maps)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        for key in ("image", "label_raw", "label_plant", "label_disease"):
            batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16,
            enabled=args.amp,
        ):
            outputs = model(batch["image"])
            loss = compute_loss(args.model, outputs, batch, criterion, args)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        accumulator.update(float(loss.item()), batch, outputs)
        if args.log_interval and (batch_index + 1) % args.log_interval == 0:
            print(
                f"epoch={epoch} phase={phase} batch={batch_index + 1}/{len(loader)} "
                f"loss={loss.item():.6f}",
                flush=True,
            )
    return accumulator.compute()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp is supported only with CUDA in this experiment runner")
    output_dir = args.output_dir or Path("inception_experiments/runs") / args.model / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_dir / "class_maps.json", encoding="utf-8") as handle:
        class_maps = json.load(handle)
    config = ModelConfig(
        name=args.model,
        num_raw=int(class_maps["num_classes"]["raw"]),
        num_plants=int(class_maps["num_classes"]["plant"]),
        num_diseases=int(class_maps["num_classes"]["disease"]),
        weights=args.weights,
        dropout=args.dropout,
    )
    model = build_model(config).to(device)
    train_loader = make_loader(args.data_dir / "train_split.csv", args.batch_size, True, args.num_workers, device, args.seed)
    val_loader = make_loader(args.data_dir / "val_split.csv", args.batch_size, False, args.num_workers, device, args.seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if args.amp and args.amp_dtype == "fp16"
        else None
    )

    configuration = vars(args).copy()
    configuration.update(
        {
            "output_dir": output_dir,
            "runtime": runtime_info(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    write_json(output_dir / "config.json", configuration)
    print(json.dumps(configuration["runtime"], indent=2), flush=True)

    history = []
    best_primary = -1.0
    best_secondary = -1.0
    first_epoch = 1
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        if scaler is not None and resume.get("scaler_state") is not None:
            scaler.load_state_dict(resume["scaler_state"])
        history = resume["history"]
        best_primary = float(resume["best_primary"])
        best_secondary = float(resume["best_secondary"])
        first_epoch = int(resume["epoch"]) + 1
        print(f"Resuming from epoch {first_epoch}", flush=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, class_maps, args,
            optimizer=optimizer, scaler=scaler, max_batches=args.max_train_batches,
            phase="train", epoch=epoch,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, class_maps, args,
            max_batches=args.max_val_batches, phase="val", epoch=epoch,
        )
        primary = val_metrics["raw_acc"] if args.model in RAW_MODELS else val_metrics["both_acc"]
        secondary = val_metrics["raw_macro_f1"] if args.model in RAW_MODELS else (
            val_metrics["plant_macro_f1"] + val_metrics["disease_macro_f1"]
        ) / 2
        row = {
            "epoch": epoch,
            "duration_seconds": time.perf_counter() - epoch_started,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        write_json(output_dir / "history.json", history)
        print(json.dumps(row, indent=2), flush=True)
        if (primary, secondary) > (best_primary, best_secondary):
            best_primary, best_secondary = primary, secondary
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config.__dict__,
                    "class_maps": class_maps,
                    "training_config": configuration,
                    "best_epoch": epoch,
                    "best_primary": primary,
                    "best_secondary": secondary,
                },
                output_dir / "best_model.pt",
            )
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "history": history,
                "best_primary": best_primary,
                "best_secondary": best_secondary,
            },
            output_dir / "last_state.pt",
        )

    summary = {
        "best_epoch": int(torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=False)["best_epoch"]),
        "best_primary": best_primary,
        "best_secondary": best_secondary,
        "training_seconds": sum(float(row["duration_seconds"]) for row in history),
        "checkpoint_bytes": (output_dir / "best_model.pt").stat().st_size,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
