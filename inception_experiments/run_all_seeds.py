"""Run E2, E3, and E4 for seeds 42, 3407, and 2026 sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2026])
    parser.add_argument("--runs-dir", type=Path, default=Path("inception_experiments/runs/formal"))
    parser.add_argument("--data-dir", type=Path, default=Path("devide_dataset/data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--sampling", choices=("none", "raw-balanced"), default="none")
    args = parser.parse_args()

    for seed in args.seeds:
        subprocess.run(
            [
                sys.executable, "-m", "inception_experiments.run_sequence",
                "--data-dir", str(args.data_dir),
                "--runs-dir", str(args.runs_dir),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(seed),
                "--device", args.device,
                "--num-workers", str(args.num_workers),
                "--sampling", args.sampling,
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable, "-m", "inception_experiments.aggregate_results",
            "--runs-dir", str(args.runs_dir),
            "--seeds", *(str(seed) for seed in args.seeds),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
