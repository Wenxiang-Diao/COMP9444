"""Run the frozen E2 -> E3 -> E4 protocol with restartable checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


EXPERIMENTS = ("single_raw", "plain_multi", "gsmo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--data-dir", type=Path, default=Path("devide_dataset/data"))
    parser.add_argument("--runs-dir", type=Path, default=Path("inception_experiments/runs/formal"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampling", choices=("none", "raw-balanced"), default="none")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_frozen_data(data_dir: Path) -> None:
    metadata_path = data_dir / "split_metadata.json"
    legacy_path = data_dir / "grouped_split_summary.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    elif legacy_path.exists():
        metadata = json.loads(legacy_path.read_text(encoding="utf-8"))
        if metadata.get("groups_crossing_splits") != 0:
            raise RuntimeError("Duplicate groups still cross split boundaries")
    else:
        raise FileNotFoundError(f"Split metadata is missing: {metadata_path}")
    expected_counts = metadata.get("counts", {})
    if set(expected_counts) != {"train", "val", "test"}:
        raise RuntimeError(f"Invalid split counts: {expected_counts}")
    for split, expected in expected_counts.items():
        csv_path = data_dir / f"{split}_split.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        # Header plus one line per image; paths and labels contain no embedded newlines.
        actual = sum(1 for _ in csv_path.open(encoding="utf-8")) - 1
        if actual != expected:
            raise RuntimeError(f"{csv_path}: expected {expected} rows, found {actual}")


def validate_device(device: str, dry_run: bool) -> None:
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }[device]
    if not available and not dry_run:
        raise RuntimeError(
            f"Requested device '{device}' is unavailable. Run this command from a process "
            "with accelerator access; the sequence will not silently fall back to CPU."
        )


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"\n$ {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] $ {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def training_is_complete(output_dir: Path, epochs: int) -> bool:
    history_path = output_dir / "history.json"
    summary_path = output_dir / "training_summary.json"
    if not history_path.exists() or not summary_path.exists():
        return False
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return bool(history) and int(history[-1]["epoch"]) >= epochs


def main() -> None:
    args = parse_args()
    validate_frozen_data(args.data_dir)
    validate_device(args.device, args.dry_run)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.runs_dir / "sequence_state.json"
    state = {
        "protocol": "E2 -> E3 -> E4",
        "models": args.models,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "sampling": args.sampling,
        "updated_at": now(),
        "experiments": {},
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["experiments"] = previous.get("experiments", {})
    write_json(state_path, state)

    for model_name in args.models:
        output_dir = args.runs_dir / model_name / f"seed_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "sequence.log"
        experiment = state["experiments"].setdefault(model_name, {})
        experiment.update({"status": "training", "updated_at": now()})
        write_json(state_path, state)

        if not training_is_complete(output_dir, args.epochs):
            train_command = [
                sys.executable, "-m", "inception_experiments.train",
                "--model", model_name,
                "--data-dir", str(args.data_dir),
                "--output-dir", str(output_dir),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(args.seed),
                "--device", args.device,
                "--num-workers", str(args.num_workers),
                "--weights", "imagenet",
                "--sampling", args.sampling,
            ]
            last_state = output_dir / "last_state.pt"
            if last_state.exists():
                train_command.extend(("--resume", str(last_state)))
            if args.device == "cuda":
                train_command.append("--amp")
            run_logged(train_command, log_path, args.dry_run)

        experiment.update({"status": "trained", "updated_at": now()})
        write_json(state_path, state)
        run_logged(
            [sys.executable, "-m", "inception_experiments.plot_history", str(output_dir / "history.json")],
            log_path,
            args.dry_run,
        )

        evaluation_dir = output_dir / "test_evaluation"
        metrics_path = evaluation_dir / "metrics.json"
        if not metrics_path.exists():
            run_logged(
                [
                    sys.executable, "-m", "inception_experiments.evaluate",
                    "--checkpoint", str(output_dir / "best_model.pt"),
                    "--data-dir", str(args.data_dir),
                    "--split", "test",
                    "--output-dir", str(evaluation_dir),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--device", args.device,
                ],
                log_path,
                args.dry_run,
            )
        experiment.update({"status": "complete", "updated_at": now()})
        state["updated_at"] = now()
        write_json(state_path, state)

    if set(args.models) == set(EXPERIMENTS):
        run_logged(
            [
                sys.executable, "-m", "inception_experiments.compare_results",
                "--runs-dir", str(args.runs_dir),
                "--seed", str(args.seed),
            ],
            args.runs_dir / "sequence.log",
            args.dry_run,
        )
    print(f"Sequence complete. State: {state_path}")


if __name__ == "__main__":
    main()
