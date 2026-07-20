from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    args = parser.parse_args()
    with open(args.history, encoding="utf-8") as handle:
        history = json.load(handle)
    output_dir = args.history.parent
    epochs = [row["epoch"] for row in history]
    for metric in sorted(set(history[0]["train"]) & set(history[0]["val"])):
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, [row["train"][metric] for row in history], label="train")
        axis.plot(epochs, [row["val"][metric] for row in history], label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric)
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"training_{metric}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
