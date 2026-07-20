"""Aggregate the three formal seeds as mean +/- standard deviation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("inception_experiments/runs/formal"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2026])
    args = parser.parse_args()
    frames = []
    for seed in args.seeds:
        path = args.runs_dir / f"final_results_seed_{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(1, "seed", seed)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.runs_dir / "all_seed_results.csv", index=False)

    numeric = [column for column in combined.columns if column not in {"model", "seed"}]
    grouped = combined.groupby("model", sort=False)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(args.runs_dir / "final_mean_std.csv", index=False)
    headers = list(grouped.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in grouped.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    (args.runs_dir / "final_mean_std.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
