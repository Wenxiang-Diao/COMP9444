# CNN Baseline for Plant Leaf Disease Classification

This directory contains a PyTorch CNN baseline for classifying plant-leaf images into 38 combined plant/disease classes. The `Background_without_leaves` directory is excluded. The final, self-contained analysis is in `notebooks/BaselineCnn.ipynb`; the files under `src/` provide supporting command-line training, basic evaluation, and plotting entry points.

## Project layout

Run all commands from the repository root (`COMP9444/`).

```text
COMP9444/
├── devide_dataset/
│   └── data/
│       ├── train_split.csv
│       ├── val_split.csv
│       ├── test_split.csv
│       └── class_maps.json
└── model/Baseline/
    ├── src/
    ├── models/
    ├── results/
    └── notebooks/
```

The CSV files must contain `path` and `label_raw` columns. Generate them with `devide_dataset/prepare_data.py` before training. If the dataset is moved after generating the CSV files, regenerate the splits because their image paths will no longer be valid.

The prepared data uses a stratified 75% / 15% / 10% train, validation, and test split with random seed `42`. The preparation script separates the sorted first 80% and last 20% of each class, applies the same stratified 75% / 15% / 10% split to both portions, and then merges the corresponding splits.

## Dependencies

Install the baseline dependencies:

```bash
python -m pip install -r model/Baseline/requirements.txt
```

Seaborn is optional at runtime; if it is unavailable, the confusion matrix uses Matplotlib directly.

To run the notebook in this environment, select the project virtual environment as the Jupyter kernel.

## Data preprocessing

All images are resized to `299 x 299`, converted to tensors, and normalized with ImageNet mean and standard deviation. Training images additionally use random horizontal flipping and random rotation up to 15 degrees. Validation and test images do not use random augmentation.

The final training configuration uses:

- Batch size: `128`
- Epochs: `20`
- Loss: cross-entropy
- Optimizer: Adam with learning rate `0.001`
- DataLoader workers: `0` for stable execution in a macOS notebook
- Model selection: lowest validation loss

## Training

```bash
python model/Baseline/src/train.py
```

Training evaluates only the validation split after each epoch. The checkpoint with the lowest validation loss is saved to `model/Baseline/models/cnn_leaf.pth`. Once training and model selection are complete, the test split is evaluated exactly once.

Per-epoch training and validation metrics are written to `model/Baseline/results/cnn_metrics.csv`.

The training loop uses `optimizer.zero_grad(set_to_none=True)` to avoid unnecessary gradient-buffer writes. The Batch size of 128 was selected after a short local throughput comparison against Batch sizes 64 and 96; the benchmark code is not part of the final notebook.

## Evaluation

```bash
python model/Baseline/src/evaluate.py
```

This loads the saved checkpoint, reports test accuracy, macro precision, macro recall and macro F1, and writes the confusion matrix to `model/Baseline/results/cnn_confusion_matrix.png`. Macro F1 gives every class equal weight, making it more informative than accuracy alone when class frequencies differ.

## Complete notebook analysis

Open and run:

```text
model/Baseline/notebooks/BaselineCnn.ipynb
```

The notebook contains the complete workflow and submission output:

- dataset exploration and class distribution;
- preprocessing definitions;
- unchanged three-convolution-layer CNN definition;
- optional full training controlled by `RUN_FULL_TRAINING`;
- best-checkpoint loading;
- test accuracy, macro precision, macro recall, and macro F1;
- confusion matrix;
- training and validation loss/accuracy curves;
- per-image predictions with Top-3 probabilities;
- five high-confidence correct predictions;
- five low-confidence correct predictions;
- five high-confidence incorrect predictions; and
- five failures drawn from the most frequent confusion pairs.

After a completed training run, set `RUN_FULL_TRAINING = False` before submitting or running the notebook again. This makes `Run All` load the existing best checkpoint instead of retraining for 20 epochs.

## Visualisation

```bash
python model/Baseline/src/visualize.py
```

This creates training-versus-validation loss and accuracy plots in `model/Baseline/results/`.

## Generated outputs

The complete notebook generates the following files under `model/Baseline/results/`:

```text
cnn_metrics.csv
cnn_confusion_matrix.png
training_vs_validation_loss.png
training_vs_validation_accuracy.png
test_predictions.csv
high_confidence_successes.png
low_confidence_successes.png
high_confidence_failures.png
top_confusion_failures.png
```

`test_predictions.csv` records each test image path, true and predicted class, predicted confidence, true-class probability, correctness, and Top-3 predictions. The case figures are selected by fixed confidence and confusion-count rules rather than manual image selection.

## Reproducibility

The command-line training script uses random seed `42` for Python, NumPy, PyTorch, and training-data shuffling. The notebook fixes the training-data shuffle generator to seed `42`. Exact results can still vary across sessions, PyTorch versions, and hardware backends. The reported confidence values are softmax scores and should not be interpreted as guaranteed probabilities of correctness.
