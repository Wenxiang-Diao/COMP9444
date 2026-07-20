"""Combine completed E2, E3, and E4 artifacts into one results table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPERIMENTS = ("single_raw", "plain_multi", "gsmo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("inception_experiments/runs/formal"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for name in EXPERIMENTS:
        run_dir = args.runs_dir / name / f"seed_{args.seed}"
        metrics_path = run_dir / "test_evaluation" / "metrics.json"
        efficiency_path = run_dir / "test_evaluation" / "efficiency.json"
        summary_path = run_dir / "training_summary.json"
        if not all(path.exists() for path in (metrics_path, efficiency_path, summary_path)):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        inference = efficiency["inference"]
        rows.append(
            {
                "model": name,
                "comparison_accuracy": metrics["raw_acc"] if name == "single_raw" else metrics["both_acc"],
                "raw_macro_f1": metrics.get("raw_macro_f1"),
                "raw_weighted_f1": metrics.get("raw_weighted_f1"),
                "plant_accuracy": metrics["plant_acc"],
                "plant_macro_f1": metrics["plant_macro_f1"],
                "disease_accuracy": metrics["disease_acc"],
                "disease_macro_f1": metrics["disease_macro_f1"],
                # This metric applies only to independently predicted plant and
                # disease heads; a 39-class raw prediction is legal by design.
                "legal_combination_rate": metrics.get("legal_combination_rate", 1.0),
                "parameters": efficiency["parameter_count"],
                "checkpoint_bytes": efficiency["checkpoint_bytes"],
                "training_seconds": summary["training_seconds"],
                "best_epoch": summary["best_epoch"],
                "inference_ms_per_image": inference["mean_milliseconds_per_image"],
                "inference_images_per_second": inference["images_per_second"],
            }
        )
    if not rows:
        raise RuntimeError(f"No completed formal results found below {args.runs_dir}")
    output = pd.DataFrame(rows)
    stem = f"final_results_seed_{args.seed}"
    output.to_csv(args.runs_dir / f"{stem}.csv", index=False)
    headers = list(output.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in output.itertuples(index=False, name=None):
        markdown_rows.append("| " + " | ".join(str(value) for value in values) + " |")
    (args.runs_dir / f"{stem}.md").write_text(
        "\n".join(markdown_rows) + "\n", encoding="utf-8"
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
