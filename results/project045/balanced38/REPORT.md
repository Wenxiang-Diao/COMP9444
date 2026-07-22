# Balanced 38-class experiment analysis

## Scope and protocol

This report repeats the Project 045 local analysis for the 18 experiments trained on the supplied 38-class split (45,256 train / 9,052 validation / 6,035 test). `Background_without_leaves` is excluded. Training uses inverse-frequency `WeightedRandomSampler` sampling with replacement; validation and test remain untouched. Consequently, the unique training data are still imbalanced (5.51:1), while the expected class exposure during each training epoch is approximately uniform.

All six configurations use seeds 42, 3407 and 2026. Multi-output models are compared through constrained decoding over the 38 legal plant-disease combinations. Batch-1 latency is a BF16 model-only forward pass on an NVIDIA RTX PRO 6000 (30 warm-ups, 200 timed iterations), excluding disk I/O and preprocessing.

## Main results

| Configuration | Accuracy %, mean ± SD | Macro-F1 %, mean ± SD | Parameters | Batch-1 P95 ms |
|---|---:|---:|---:|---:|
| InceptionV3 38-class | 76.50 ± 3.58 | 75.12 ± 4.39 | 21.86M | 5.60 |
| InceptionV3 plain multi-output | 85.26 ± 0.95 | 85.73 ± 2.41 | 21.86M | 5.57 |
| GSMo ImageNet | 83.59 ± 6.21 | 83.29 ± 5.81 | 21.93M | 5.71 |
| PlantXViT-style reproduction | 98.19 ± 0.19 | 98.13 ± 0.10 | 0.61M | 1.01 |
| GSMo no pretraining | 98.82 ± 0.22 | 98.71 ± 0.26 | 21.93M | 5.60 |
| GSMo equal loss | 90.26 ± 3.99 | 90.34 ± 3.28 | 21.93M | 5.62 |

The strongest result is **GSMo no pretraining** at **98.82% accuracy** and **98.71% Macro-F1**. The fastest is **PlantXViT-style reproduction** at **1.01 ms P95**. Plain multi-output remains stronger and more stable than default ImageNet GSMo, so the new results still do not support a general claim that stacking alone improves performance.

## Change from the previous grouped 39-class experiments

The two runs are not a controlled sampler-only ablation: the new experiment also removes the background class and uses a different supplied split/test set. Deltas therefore describe the observed pipeline change and must not be attributed solely to rebalancing.

| Configuration | Previous accuracy % | Balanced-38 accuracy % | Accuracy change pp | Macro-F1 change pp |
|---|---:|---:|---:|---:|
| InceptionV3 38-class | 87.24 | 76.50 | -10.74 | -10.75 |
| InceptionV3 plain multi-output | 90.64 | 85.26 | -5.38 | -3.97 |
| GSMo ImageNet | 86.11 | 83.59 | -2.52 | -2.43 |
| PlantXViT-style reproduction | 98.63 | 98.19 | -0.45 | -0.30 |
| GSMo no pretraining | 98.81 | 98.82 | +0.01 | +0.14 |
| GSMo equal loss | 83.02 | 90.26 | +7.23 | +7.16 |

Weighted sampling did not uniformly improve aggregate accuracy. It shifts optimisation toward minority classes, and Macro-F1 is the more relevant metric for judging that objective. Default GSMo and the Inception controls remain seed-sensitive; GSMo equal loss improves substantially but still varies more than the two strongest configurations.

## Error and class-level analysis

For the strongest configuration, the five lowest mean class F1 values are:

| Model | Class | Mean F1 | F1 SD | Support |
|---|---|---:|---:|---:|
| GSMo no pretraining | Tomato___Early_blight | 0.9602 | 0.0215 | 100 |
| GSMo no pretraining | Tomato___Target_Spot | 0.9621 | 0.0153 | 140 |
| GSMo no pretraining | Corn___Northern_Leaf_Blight | 0.9665 | 0.0181 | 100 |
| GSMo no pretraining | Tomato___Late_blight | 0.9717 | 0.0044 | 191 |
| GSMo no pretraining | Tomato___Spider_mites Two-spotted_spider_mite | 0.9724 | 0.0063 | 168 |

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
