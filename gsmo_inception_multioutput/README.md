# GSMo-CNN InceptionV3 Multi-output

This folder contains a clean PyTorch implementation of the GSMo-CNN multi-output model adapted to the local COMP9444 leaf disease split files in `devide_dataset/data`.

## What This Uses

- Dataset: `Dataset/Plant_leave_diseases_dataset_with_augmentation`
- Split files: `devide_dataset/data/train_split.csv`, `val_split.csv`, `test_split.csv`
- Labels:
  - `label_plant`: plant/crop head, 15 classes
  - `label_disease`: disease/status head, 22 classes
  - `label_raw`: original 39-class folder label, kept for baseline and reference
- Input size: `299 x 299`
- Backbone: InceptionV3
- Model structure: GSMo-CNN `new_model` style from the downloaded paper code

## Model Outputs

The model returns four logits:

```text
plant_output
disease_output
plant_output_t
disease_output_t
```

The temporary outputs are used for generalized stacking:

```text
plant_output_t  = plant prediction from shared Inception features
disease_output_t = disease prediction from shared Inception features
plant_output    = plant prediction from [features, disease_output_t]
disease_output  = disease prediction from [features, plant_output_t]
```

## Install

```bash
pip install -r gsmo_inception_multioutput/requirements.txt
```

## Verify Split Files

```bash
python gsmo_inception_multioutput/verify_splits.py
```

Expected current result:

```text
train: 43040
val: 9223
test: 9223
raw classes: 39
plant classes: 15
disease classes: 22
missing paths: 0
split overlaps: 0
```

## Smoke Test

Use random InceptionV3 weights so no ImageNet download is needed:

```bash
python gsmo_inception_multioutput/smoke_check.py
```

Expected output shapes:

```text
image shape: (2, 3, 299, 299)
plant_output: (2, 15)
disease_output: (2, 22)
plant_output_t: (2, 15)
disease_output_t: (2, 22)
```

## Train

Recommended first run:

```bash
python gsmo_inception_multioutput/train.py --epochs 1 --batch-size 8 --weights none
```

Then train with ImageNet weights:

```bash
python gsmo_inception_multioutput/train.py --epochs 30 --batch-size 16 --weights imagenet
```

The first ImageNet run may download torchvision weights.

## Evaluate

```bash
python gsmo_inception_multioutput/evaluate.py --checkpoint gsmo_inception_multioutput/runs/with_aug/best_model.pt --split test
```

## Notes

- The CSV files are currently encoded with a Windows Chinese encoding. `dataset.py` tries UTF-8 first and then falls back to `gbk`.
- This implementation intentionally does not modify the downloaded `plant_pathology_dl` source folder.
- Because the current split uses `with_augmentation`, report it clearly as a with-augmentation experiment.

