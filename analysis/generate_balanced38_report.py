"""Generate the compact, reviewable analysis for the balanced 38-class runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SEEDS = (42, 3407, 2026)
MODEL_DIRS = {
    "InceptionV3 38-class": "formal/single_raw",
    "InceptionV3 plain multi-output": "formal/plain_multi",
    "GSMo ImageNet": "formal/gsmo",
    "PlantXViT-style reproduction": "extended/plantxvit_reproduction",
    "GSMo no pretraining": "extended/gsmo_no_pretrain",
    "GSMo equal loss": "extended/gsmo_equal_loss",
}
OLD_NAMES = {
    "InceptionV3 38-class": "single_raw",
    "InceptionV3 plain multi-output": "plain_multi",
    "GSMo ImageNet": "gsmo",
    "PlantXViT-style reproduction": "plantxvit_reproduction",
    "GSMo no pretraining": "gsmo_no_pretrain",
    "GSMo equal loss": "gsmo_equal_loss",
}


def load_runs(runs_root: Path) -> pd.DataFrame:
    records = []
    for model, relative in MODEL_DIRS.items():
        for seed in SEEDS:
            run = runs_root / relative / f"seed_{seed}"
            metrics = json.loads((run / "test_evaluation/metrics.json").read_text())
            joint = json.loads(
                (run / "supplementary_test_evaluation/joint_metrics.json").read_text()
            )
            efficiency = json.loads((run / "test_evaluation/efficiency.json").read_text())
            training = json.loads((run / "training_summary.json").read_text())
            inference = efficiency["inference"]
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "joint_accuracy": joint["joint_raw_acc"],
                    "joint_macro_f1": joint["joint_raw_macro_f1"],
                    "joint_weighted_f1": joint["joint_raw_weighted_f1"],
                    "unconstrained_legal_rate": joint["unconstrained_legal_combination_rate"],
                    "plant_accuracy": metrics["plant_acc"],
                    "plant_macro_f1": metrics["plant_macro_f1"],
                    "disease_accuracy": metrics["disease_acc"],
                    "disease_macro_f1": metrics["disease_macro_f1"],
                    "parameters": efficiency["parameter_count"],
                    "checkpoint_mb": efficiency["checkpoint_bytes"] / 2**20,
                    "training_minutes": training["training_seconds"] / 60,
                    "best_epoch": training["best_epoch"],
                    "batched_images_per_second": inference["images_per_second"],
                    "batch1_mean_ms": joint["batch1_latency"]["mean_ms"],
                    "batch1_p50_ms": joint["batch1_latency"]["p50_ms"],
                    "batch1_p95_ms": joint["batch1_latency"]["p95_ms"],
                }
            )
    return pd.DataFrame(records)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.groupby("model", sort=False).agg(
        runs=("seed", "size"),
        joint_accuracy_mean=("joint_accuracy", "mean"),
        joint_accuracy_sd=("joint_accuracy", "std"),
        joint_macro_f1_mean=("joint_macro_f1", "mean"),
        joint_macro_f1_sd=("joint_macro_f1", "std"),
        unconstrained_legal_rate_mean=("unconstrained_legal_rate", "mean"),
        parameters=("parameters", "first"),
        checkpoint_mb=("checkpoint_mb", "first"),
        training_minutes_mean=("training_minutes", "mean"),
        batch1_p50_ms_mean=("batch1_p50_ms", "mean"),
        batch1_p95_ms_mean=("batch1_p95_ms", "mean"),
    ).reset_index()


def class_analysis(runs_root: Path) -> pd.DataFrame:
    rows = []
    for model, relative in MODEL_DIRS.items():
        for seed in SEEDS:
            path = runs_root / relative / f"seed_{seed}/supplementary_test_evaluation/joint_classification_report.json"
            report = json.loads(path.read_text())
            for label, values in report.items():
                if isinstance(values, dict) and "f1-score" in values and label not in {
                    "macro avg", "weighted avg"
                }:
                    rows.append(
                        {"model": model, "seed": seed, "class": label,
                         "f1": values["f1-score"], "support": values["support"]}
                    )
    return pd.DataFrame(rows).groupby(["model", "class"], as_index=False).agg(
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"), support=("support", "mean")
    )


def save_figures(runs_root: Path, runs: pd.DataFrame, summary: pd.DataFrame,
                 splits: dict[str, pd.DataFrame], output: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    counts = splits["train"].raw_class.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 9))
    counts.plot.barh(ax=ax, color="#4c78a8")
    ax.set_title("Training-set class counts before weighted resampling")
    ax.set_xlabel("Unique training images")
    ax.set_ylabel("")
    fig.savefig(figures / "class_distribution.png")
    plt.close(fig)

    view = summary.sort_values("joint_accuracy_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(view.model, 100 * view.joint_accuracy_mean,
            xerr=100 * view.joint_accuracy_sd, capsize=4,
            color=sns.color_palette("viridis", len(view)))
    ax.set_xlabel("Constrained 38-class test accuracy (%)")
    ax.set_title("Balanced training: accuracy and seed stability (mean ± SD, n=3)")
    ax.set_xlim(max(0, 100 * view.joint_accuracy_mean.min() - 12), 101)
    fig.savefig(figures / "accuracy_stability.png")
    plt.close(fig)

    ablation_names = ["GSMo ImageNet", "GSMo no pretraining", "GSMo equal loss"]
    ablation = summary.set_index("model").loc[ablation_names].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(ablation.model, 100 * ablation.joint_accuracy_mean,
                yerr=100 * ablation.joint_accuracy_sd, capsize=5,
                color=["#4c78a8", "#59a14f", "#e15759"])
    axes[0].set_ylabel("Constrained test accuracy (%)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_title("GSMo ablations")
    sns.pointplot(data=runs[runs.model.isin(ablation_names)], x="seed",
                  y="joint_accuracy", hue="model", ax=axes[1])
    axes[1].set_title("Per-seed behaviour")
    axes[1].set_ylabel("Constrained test accuracy")
    axes[1].legend(title="")
    fig.savefig(figures / "gsmo_ablations.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    latency = summary.sort_values("batch1_p95_ms_mean")
    y = np.arange(len(latency))
    ax.barh(y, latency.batch1_p95_ms_mean, color="#4c78a8", label="P95")
    ax.scatter(latency.batch1_p50_ms_mean, y, color="#e15759", label="P50", zorder=3)
    ax.set_yticks(y, latency.model)
    ax.set_xlabel("Batch-1 forward-pass latency (ms, BF16 CUDA)")
    ax.set_title("RTX PRO 6000 latency")
    ax.legend()
    fig.savefig(figures / "batch1_latency.png")
    plt.close(fig)

    curve_models = list(MODEL_DIRS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for axis, model in zip(axes.flat, curve_models):
        rows = []
        relative = MODEL_DIRS[model]
        for seed in SEEDS:
            history = json.loads((runs_root / relative / f"seed_{seed}/history.json").read_text())
            rows.extend({"epoch": item["epoch"], "train": item["train"]["loss"],
                         "validation": item["val"]["loss"]} for item in history)
        means = pd.DataFrame(rows).groupby("epoch")[["train", "validation"]].mean()
        axis.plot(means.index, means.train, label="train")
        axis.plot(means.index, means.validation, label="validation")
        axis.set_yscale("log")
        axis.set_title(model, fontsize=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss (log scale)")
        axis.legend()
    fig.suptitle("Mean learning curves across three seeds")
    fig.savefig(figures / "learning_curves.png")
    plt.close(fig)

    best_models = ["GSMo no pretraining", "PlantXViT-style reproduction"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, model in zip(axes, best_models):
        seed = int(runs.loc[runs[runs.model == model].joint_accuracy.idxmax(), "seed"])
        path = runs_root / MODEL_DIRS[model] / f"seed_{seed}/supplementary_test_evaluation/joint_raw_confusion_matrix.csv"
        matrix = pd.read_csv(path, index_col=0)
        normalised = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
        sns.heatmap(normalised, cmap="Blues", vmin=0, vmax=1, ax=axis,
                    xticklabels=False, yticklabels=False, cbar=False)
        axis.set_title(f"{model} (best seed {seed})")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
    fig.savefig(figures / "best_model_confusions.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, comparison: pd.DataFrame,
                 class_table: pd.DataFrame, splits: dict[str, pd.DataFrame], output: Path) -> None:
    train_counts = splits["train"].raw_class.value_counts()
    strongest = summary.loc[summary.joint_accuracy_mean.idxmax()]
    fastest = summary.loc[summary.batch1_p95_ms_mean.idxmin()]
    worst = class_table[class_table.model == strongest.model].nsmallest(5, "f1_mean")
    worst_rows = ["| Model | Class | Mean F1 | F1 SD | Support |",
                  "|---|---|---:|---:|---:|"]
    for item in worst.itertuples(index=False):
        worst_rows.append(
            f"| {item.model} | {item._1} | {item.f1_mean:.4f} | "
            f"{item.f1_sd:.4f} | {item.support:.0f} |"
        )
    rows = []
    for item in summary.itertuples(index=False):
        rows.append(
            f"| {item.model} | {100*item.joint_accuracy_mean:.2f} ± {100*item.joint_accuracy_sd:.2f} | "
            f"{100*item.joint_macro_f1_mean:.2f} ± {100*item.joint_macro_f1_sd:.2f} | "
            f"{item.parameters/1e6:.2f}M | {item.batch1_p95_ms_mean:.2f} |"
        )
    deltas = []
    for item in comparison.itertuples(index=False):
        deltas.append(
            f"| {item.model} | {100*item.old_accuracy:.2f} | {100*item.balanced_accuracy:.2f} | "
            f"{100*item.accuracy_delta:+.2f} | {100*item.macro_f1_delta:+.2f} |"
        )
    report = f"""# Balanced 38-class experiment analysis

## Scope and protocol

This report repeats the Project 045 local analysis for the 18 experiments trained on the supplied 38-class split (45,256 train / 9,052 validation / 6,035 test). `Background_without_leaves` is excluded. Training uses inverse-frequency `WeightedRandomSampler` sampling with replacement; validation and test remain untouched. Consequently, the unique training data are still imbalanced ({train_counts.max()/train_counts.min():.2f}:1), while the expected class exposure during each training epoch is approximately uniform.

All six configurations use seeds 42, 3407 and 2026. Multi-output models are compared through constrained decoding over the 38 legal plant-disease combinations. Batch-1 latency is a BF16 model-only forward pass on an NVIDIA RTX PRO 6000 (30 warm-ups, 200 timed iterations), excluding disk I/O and preprocessing.

## Main results

| Configuration | Accuracy %, mean ± SD | Macro-F1 %, mean ± SD | Parameters | Batch-1 P95 ms |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The strongest result is **{strongest.model}** at **{100*strongest.joint_accuracy_mean:.2f}% accuracy** and **{100*strongest.joint_macro_f1_mean:.2f}% Macro-F1**. The fastest is **{fastest.model}** at **{fastest.batch1_p95_ms_mean:.2f} ms P95**. Plain multi-output remains stronger and more stable than default ImageNet GSMo, so the new results still do not support a general claim that stacking alone improves performance.

## Change from the previous grouped 39-class experiments

The two runs are not a controlled sampler-only ablation: the new experiment also removes the background class and uses a different supplied split/test set. Deltas therefore describe the observed pipeline change and must not be attributed solely to rebalancing.

| Configuration | Previous accuracy % | Balanced-38 accuracy % | Accuracy change pp | Macro-F1 change pp |
|---|---:|---:|---:|---:|
{chr(10).join(deltas)}

Weighted sampling did not uniformly improve aggregate accuracy. It shifts optimisation toward minority classes, and Macro-F1 is the more relevant metric for judging that objective. Default GSMo and the Inception controls remain seed-sensitive; GSMo equal loss improves substantially but still varies more than the two strongest configurations.

## Error and class-level analysis

For the strongest configuration, the five lowest mean class F1 values are:

{chr(10).join(worst_rows)}

The per-class table and confusion matrices are included in the compact result package. Near-perfect results from GSMo without pretraining and PlantXViT-style reproduction should be treated cautiously: they warrant an augmentation-family leakage audit or evaluation on a split derived from unaugmented source images before making a real-world generalisation claim.

## Conclusions and limitations

1. Rebalancing was implemented at training time only; validation and test distributions were not modified.
2. Plain multi-output outperforms both the single-output Inception control and default ImageNet GSMo.
3. GSMo is highly sensitive to initialisation and loss weighting. No-pretraining GSMo is best, while equal loss is much stronger than before but remains less stable.
4. PlantXViT-style reproduction retains the best accuracy/latency trade-off, but it is not the authors' official implementation.
5. Three seeds support mean/SD reporting but not strong significance claims.
6. This dataset remains laboratory-style and offline augmented; field robustness, calibration and end-to-end latency are outside the evidence provided here.

## Generated artifacts

- `per_seed_results.csv`: all 18 run-level metrics.
- `model_summary.csv`: mean and sample SD across three seeds.
- `previous_vs_balanced.csv`: descriptive comparison with the earlier 39-class run.
- `per_class_f1.csv`: class-level mean F1 and SD.
- `figures/`: class distribution, stability, ablations, learning curves, latency and confusion matrices.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path,
                        default=Path("inception_experiments/runs/server_balanced_38/extracted"))
    parser.add_argument("--output", type=Path, default=Path("results/project045/balanced38"))
    parser.add_argument("--old-summary", type=Path,
                        default=Path("results/project045/supplementary_results_summary.csv"))
    args = parser.parse_args()
    runs_root = args.export_root / "inception_experiments/runs/balanced_38"
    split_root = args.export_root / "devide_dataset/data_38_balanced"
    required = [runs_root, split_root, args.old_summary]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(required)
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {name: pd.read_csv(split_root / f"{name}_split.csv") for name in ("train", "val", "test")}
    runs = load_runs(runs_root)
    summary = summarise(runs)
    classes = class_analysis(runs_root)
    old = pd.read_csv(args.old_summary).set_index("configuration")
    comparison = summary[["model", "joint_accuracy_mean", "joint_macro_f1_mean"]].copy()
    comparison["old_accuracy"] = comparison.model.map(lambda x: old.loc[OLD_NAMES[x], "joint_accuracy_mean"])
    comparison["old_macro_f1"] = comparison.model.map(lambda x: old.loc[OLD_NAMES[x], "joint_macro_f1_mean"])
    comparison = comparison.rename(columns={"joint_accuracy_mean": "balanced_accuracy",
                                            "joint_macro_f1_mean": "balanced_macro_f1"})
    comparison["accuracy_delta"] = comparison.balanced_accuracy - comparison.old_accuracy
    comparison["macro_f1_delta"] = comparison.balanced_macro_f1 - comparison.old_macro_f1
    runs.to_csv(args.output / "per_seed_results.csv", index=False)
    summary.to_csv(args.output / "model_summary.csv", index=False)
    comparison.to_csv(args.output / "previous_vs_balanced.csv", index=False)
    classes.to_csv(args.output / "per_class_f1.csv", index=False)
    pd.DataFrame({"split": list(splits), "images": [len(x) for x in splits.values()],
                  "raw_classes": [x.raw_class.nunique() for x in splits.values()]}).to_csv(
                      args.output / "split_summary.csv", index=False)
    save_figures(runs_root, runs, summary, splits, args.output)
    write_report(summary, comparison, classes, splits, args.output)
    print(f"Wrote balanced analysis to {args.output}")


if __name__ == "__main__":
    main()
