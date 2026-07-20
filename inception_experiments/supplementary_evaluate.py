from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .dataset import make_loader
from .models import RAW_MODELS, ModelConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add joint 39-class metrics and true batch-1 latency.")
    parser.add_argument("--runs-dir", type=Path, default=Path("inception_experiments/runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("devide_dataset/data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--latency-iterations", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def legal_raw_arrays(class_maps: dict) -> tuple[torch.Tensor, torch.Tensor]:
    raw_to_plant = torch.empty(class_maps["num_classes"]["raw"], dtype=torch.long)
    raw_to_disease = torch.empty_like(raw_to_plant)
    for raw_name, raw_idx in class_maps["raw_to_idx"].items():
        if "___" in raw_name:
            plant, disease = raw_name.split("___", 1)
        else:
            plant, disease = "Background", "without_leaves"
        raw_to_plant[raw_idx] = class_maps["plant_to_idx"][plant]
        raw_to_disease[raw_idx] = class_maps["disease_to_idx"][disease]
    return raw_to_plant, raw_to_disease


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch1_latency(model, sample, device, amp, amp_dtype, warmup, iterations) -> dict:
    durations_ms: list[float] = []
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    model.eval()
    with torch.inference_mode():
        for index in range(warmup + iterations):
            synchronize(device)
            started = time.perf_counter_ns()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                model(sample)
            synchronize(device)
            if index >= warmup:
                durations_ms.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(durations_ms)
    return {
        "definition": "model forward pass only; batch=1; preprocessing and disk I/O excluded",
        "precision": amp_dtype if amp else "fp32",
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "images_per_second_batch1": float(1000.0 / values.mean()),
    }


def evaluate_checkpoint(checkpoint_path: Path, args: argparse.Namespace) -> dict:
    output_dir = checkpoint_path.parent / "supplementary_test_evaluation"
    metrics_path = output_dir / "joint_metrics.json"
    if metrics_path.exists() and not args.force:
        return json.loads(metrics_path.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_maps = checkpoint["class_maps"]
    model_dict = dict(checkpoint["model_config"])
    model_dict["weights"] = "none"
    config = ModelConfig(**model_dict)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    seed = int(checkpoint["training_config"]["seed"])
    amp = bool(checkpoint["training_config"].get("amp", False))
    amp_dtype = str(checkpoint["training_config"].get("amp_dtype", "bf16"))
    loader = make_loader(args.data_dir / "test_split.csv", args.batch_size, False, args.num_workers, device, seed)
    raw_to_plant, raw_to_disease = legal_raw_arrays(class_maps)
    raw_to_plant_device = raw_to_plant.to(device)
    raw_to_disease_device = raw_to_disease.to(device)

    true_raw: list[int] = []
    pred_raw: list[int] = []
    paths: list[str] = []
    unconstrained_legal: list[bool] = []
    sample = None
    started = time.perf_counter()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if sample is None:
                sample = images[:1].clone()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                outputs = model(images)
            if config.name in RAW_MODELS:
                predicted = outputs["raw"].argmax(dim=1)
                unconstrained_legal.extend([True] * len(predicted))
            else:
                plant_logp = outputs["plant"].float().log_softmax(dim=1)
                disease_logp = outputs["disease"].float().log_softmax(dim=1)
                joint_scores = plant_logp[:, raw_to_plant_device] + disease_logp[:, raw_to_disease_device]
                predicted = joint_scores.argmax(dim=1)
                plant_argmax = outputs["plant"].argmax(dim=1)
                disease_argmax = outputs["disease"].argmax(dim=1)
                legal = ((plant_argmax[:, None] == raw_to_plant_device[None, :]) &
                         (disease_argmax[:, None] == raw_to_disease_device[None, :])).any(dim=1)
                unconstrained_legal.extend(legal.cpu().tolist())
            true_raw.extend(batch["label_raw"].tolist())
            pred_raw.extend(predicted.cpu().tolist())
            paths.extend(batch["path"])
    evaluation_seconds = time.perf_counter() - started

    raw_names = [class_maps["idx_to_raw"][str(i)] for i in range(config.num_raw)]
    report = classification_report(
        true_raw, pred_raw, labels=range(config.num_raw), target_names=raw_names,
        zero_division=0, output_dict=True,
    )
    matrix = confusion_matrix(true_raw, pred_raw, labels=range(config.num_raw))
    pd.DataFrame(matrix, index=raw_names, columns=raw_names).to_csv(output_dir / "joint_raw_confusion_matrix.csv")
    Path(output_dir / "joint_classification_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "path": paths,
        "label_raw": true_raw,
        "pred_raw_constrained": pred_raw,
        "correct": np.asarray(true_raw) == np.asarray(pred_raw),
        "unconstrained_pair_legal": unconstrained_legal,
    }).to_csv(output_dir / "joint_predictions.csv", index=False)

    latency = batch1_latency(model, sample, device, amp, amp_dtype, args.warmup, args.latency_iterations)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "model": config.name,
        "seed": seed,
        "num_test_images": len(true_raw),
        "joint_decoding": "direct raw argmax" if config.name in RAW_MODELS else
            "maximum summed plant+disease log-probability over the 39 legal raw combinations",
        "joint_raw_acc": float(accuracy_score(true_raw, pred_raw)),
        "joint_raw_macro_f1": float(f1_score(true_raw, pred_raw, average="macro", zero_division=0)),
        "joint_raw_weighted_f1": float(f1_score(true_raw, pred_raw, average="weighted", zero_division=0)),
        "unconstrained_legal_combination_rate": float(np.mean(unconstrained_legal)),
        "evaluation_seconds": evaluation_seconds,
        "batch1_latency": latency,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    del model, checkpoint
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = parse_args()
    checkpoints = sorted((args.runs_dir / "formal").glob("*/*/best_model.pt"))
    checkpoints += sorted((args.runs_dir / "extended").glob("*/*/best_model.pt"))
    if len(checkpoints) != 18:
        raise RuntimeError(f"Expected 18 formal/extended best checkpoints, found {len(checkpoints)}")
    rows = []
    for index, checkpoint in enumerate(checkpoints, 1):
        print(f"[{index}/18] {checkpoint}", flush=True)
        rows.append(evaluate_checkpoint(checkpoint, args))
        print(json.dumps({k: rows[-1][k] for k in ("model", "seed", "joint_raw_acc", "joint_raw_macro_f1")}), flush=True)
    destination = args.runs_dir / "supplementary_evaluation_summary.csv"
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "batch1_latency"}
        flat.update({f"batch1_{k}": v for k, v in row["batch1_latency"].items()})
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(destination, index=False)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
