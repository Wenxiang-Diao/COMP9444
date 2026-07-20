"""Run the optional E5 and E4 ablations after the nine core runs finish."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


SEEDS = (42, 3407, 2026)


def run(command: list[str], log) -> None:
    rendered = " ".join(command)
    print(f"$ {rendered}", flush=True)
    log.write(f"$ {rendered}\n")
    log.flush()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if process.wait():
        raise RuntimeError(f"Command failed: {rendered}")


def complete(output: Path, epochs: int) -> bool:
    summary = output / "training_summary.json"
    history = output / "history.json"
    return summary.exists() and history.exists() and json.loads(history.read_text())[-1]["epoch"] >= epochs


def experiment(root: Path, data: Path, name: str, model: str, seed: int,
               epochs: int, batch: int, workers: int, weights: str,
               loss_weights: tuple[float, float, float, float] | None, log) -> None:
    output = root / name / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if not complete(output, epochs):
        command = [
            sys.executable, "-m", "inception_experiments.train", "--model", model,
            "--data-dir", str(data), "--output-dir", str(output), "--epochs", str(epochs),
            "--batch-size", str(batch), "--num-workers", str(workers), "--seed", str(seed),
            "--device", "cuda", "--weights", weights, "--amp",
        ]
        if loss_weights:
            for flag, value in zip(("--loss-plant", "--loss-disease", "--loss-plant-t", "--loss-disease-t"), loss_weights):
                command.extend((flag, str(value)))
        state = output / "last_state.pt"
        if state.exists():
            command.extend(("--resume", str(state)))
        run(command, log)
    evaluation = output / "test_evaluation"
    if not (evaluation / "metrics.json").exists():
        run([
            sys.executable, "-m", "inception_experiments.evaluate",
            "--checkpoint", str(output / "best_model.pt"), "--data-dir", str(data),
            "--split", "test", "--output-dir", str(evaluation), "--batch-size", str(batch),
            "--num-workers", str(workers), "--device", "cuda",
        ], log)


def summarize(root: Path, destination: Path) -> None:
    rows = []
    for metrics_path in sorted(root.glob("*/seed_*/test_evaluation/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        seed_dir = metrics_path.parents[1]
        rows.append({"experiment": seed_dir.parent.name, "seed": int(seed_dir.name[5:]), **metrics})
    keys = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", type=Path, default=Path("inception_experiments/runs/formal"))
    parser.add_argument("--runs-dir", type=Path, default=Path("inception_experiments/runs/extended"))
    parser.add_argument("--data-dir", type=Path, default=Path("devide_dataset/data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    # A separate screen can start this immediately; it waits for the core runner.
    while not (args.core_dir / "final_mean_std.csv").exists():
        print("Waiting for the nine core runs...", flush=True)
        time.sleep(60)
    with (args.runs_dir / "extended_run.log").open("a", encoding="utf-8") as log:
        for seed in SEEDS:
            experiment(args.runs_dir, args.data_dir, "plantxvit_reproduction", "plantxvit_raw",
                       seed, args.epochs, args.batch_size, args.num_workers, "none", None, log)
        experiment(args.runs_dir, args.data_dir, "gsmo_no_pretrain", "gsmo", 42,
                   args.epochs, args.batch_size, args.num_workers, "none", None, log)
        experiment(args.runs_dir, args.data_dir, "gsmo_equal_loss", "gsmo", 42,
                   args.epochs, args.batch_size, args.num_workers, "imagenet", (1, 1, 1, 1), log)
        summarize(args.runs_dir, args.runs_dir / "stage_17_group_results.csv")
        (args.runs_dir / "STAGE_17_COMPLETE").write_text("14 local runs + 3 teammate E1 runs = 17\n")
        for seed in SEEDS[1:]:
            experiment(args.runs_dir, args.data_dir, "gsmo_no_pretrain", "gsmo", seed,
                       args.epochs, args.batch_size, args.num_workers, "none", None, log)
            experiment(args.runs_dir, args.data_dir, "gsmo_equal_loss", "gsmo", seed,
                       args.epochs, args.batch_size, args.num_workers, "imagenet", (1, 1, 1, 1), log)
        summarize(args.runs_dir, args.runs_dir / "stage_21_group_results.csv")
        (args.runs_dir / "READY_FOR_DOWNLOAD").write_text("18 local runs + 3 teammate E1 runs = 21\n")


if __name__ == "__main__":
    main()
