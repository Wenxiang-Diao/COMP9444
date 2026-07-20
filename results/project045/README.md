# Project 045 public experiment summary

This directory contains the compact, version-controlled results used by
`analysis/COMP9444_Project045_Analysis.ipynb`. Large checkpoints, image-level
predictions, the dataset and server archives are intentionally excluded from
Git.

## Evaluation protocol

- Official augmented leaf-disease dataset: 61,486 images and 39 raw classes.
- Frozen group-aware split: 43,040 train, 9,223 validation and 9,223 test.
- Three seeds per configuration: 42, 3407 and 2026.
- All reported values use the final frozen test set after validation-only model
  selection.
- Raw-output models use direct 39-class argmax.
- Multi-output models use constrained decoding: select the legal plant-disease
  combination with the largest sum of plant and disease log-probabilities.
- Batch-1 latency is a BF16 model-only forward pass on an NVIDIA RTX PRO 6000,
  with 30 warm-up and 200 synchronised timed iterations. Image decoding,
  preprocessing and host-to-device transfer are excluded.

`supplementary_results_summary.csv` reports the mean and sample standard
deviation across three seeds. The full executed notebook contains EDA,
learning curves, ablations, confusion matrices, per-class analysis and
limitations.

The `plantxvit_reproduction` configuration is a PlantXViT-style reproduction,
not the authors' official implementation.
