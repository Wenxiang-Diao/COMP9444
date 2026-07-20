# Shared InceptionV3 experiments

This package implements the three controlled PyTorch experiments used for
COMP9444 Project 045:

- `single_raw`: E2, one 39-class output.
- `plain_multi`: E3, independent 15-class plant and 22-class disease heads.
- `gsmo`: E4, temporary heads plus generalised-stacking final heads.

All commands below run from the repository root in the
`COMP9444_GroupProject` Conda environment.

## Verify implementation

```bash
python -m inception_experiments.smoke_check
```

## Formal training

The default configuration is 30 epochs, seed 42, batch size 16, AdamW,
learning rate `1e-4`, weight decay `1e-4`, and ImageNet weights.

```bash
python -m inception_experiments.train --model single_raw --device mps
python -m inception_experiments.train --model plain_multi --device mps
python -m inception_experiments.train --model gsmo --device mps
```

Recommended: run the restartable sequence from a normal macOS Terminal. It
validates the frozen split, refuses silent CPU fallback, trains E2 then E3 then
E4, generates curves, evaluates each frozen best checkpoint once, and writes
one `final_results_seed_<seed>.csv` plus Markdown table per seed:

```bash
conda activate COMP9444_GroupProject
python -m inception_experiments.run_sequence --device mps
```

Re-running the same command resumes an interrupted `last_state.pt`, skips a
completed 30-epoch training run, and skips an existing test evaluation. Preview
all commands without training with `--dry-run`.

For the recommended three-seed experiment, use the CUDA batch size and worker
count selected by the server benchmark:

```bash
python -m inception_experiments.run_all_seeds \
  --device cuda --batch-size BATCH --num-workers WORKERS
```

This produces `all_seed_results.csv` and `final_mean_std.csv/.md` after all nine
formal runs complete.

Each output directory contains `best_model.pt`, `last_state.pt`, `config.json`,
`history.json`, and `training_summary.json`. Resume an interrupted run with:

```bash
python -m inception_experiments.train \
  --model single_raw \
  --device mps \
  --resume inception_experiments/runs/single_raw/seed_42/last_state.pt
```

## Frozen test evaluation

Run only after the model configuration and best checkpoint have been selected
using validation data:

```bash
python -m inception_experiments.evaluate \
  --checkpoint inception_experiments/runs/single_raw/seed_42/best_model.pt \
  --split test \
  --device mps
```

Generate training curves with:

```bash
python -m inception_experiments.plot_history \
  inception_experiments/runs/single_raw/seed_42/history.json
```

## Common 39-class metrics and batch-1 latency

After the formal and extended checkpoints exist, generate a directly
comparable 39-class endpoint for every model and measure synchronised batch-1
GPU latency:

```bash
python -m inception_experiments.supplementary_evaluate \
  --device cuda --batch-size 256 --num-workers 12
```

Raw-output models use direct class argmax. Multi-output models use constrained
joint decoding over the 39 legal plant-disease combinations. Each run receives
a `supplementary_test_evaluation` directory containing image-level predictions,
a classification report, a confusion matrix and joint metrics. The aggregate
CSV is written under `inception_experiments/runs/` and is intentionally ignored
by Git; the compact three-seed summary is versioned in `results/project045/`.

Do not commit datasets, machine-specific split CSVs, checkpoints, or generated
evaluation artifacts.
