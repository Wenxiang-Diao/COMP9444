# Change Log

## 2026-07-14

- Created a new independent implementation folder: `gsmo_inception_multioutput/`.
- Added `dataset.py`:
  - Reads existing `devide_dataset/data/*_split.csv` files.
  - Supports the current non-UTF-8 CSV encoding by falling back to `gbk`.
  - Returns `image`, `label_raw`, `label_plant`, and `label_disease`.
  - Uses InceptionV3-compatible `299 x 299` image transforms.
- Added `model.py`:
  - Implements GSMo-CNN with an InceptionV3 backbone.
  - Mirrors the downloaded paper code's `new_model` structure with `plant_output_t`, `disease_output_t`, `plant_output`, and `disease_output`.
  - Uses logits for PyTorch `CrossEntropyLoss`.
- Added `train.py`:
  - Trains the GSMo-CNN model on the current with-augmentation split.
  - Uses loss weights matching the downloaded paper default: plant `0.4`, disease `0.5`, plant temporary `0.1`, disease temporary `0.1`.
  - Saves `best_model.pt` by validation `both_acc`.
  - Writes `history.json`.
- Added `evaluate.py`:
  - Loads a saved checkpoint.
  - Reports plant accuracy, disease accuracy, and both-head exact-match accuracy.
  - Saves classification reports and confusion matrices.
- Added `verify_splits.py`:
  - Rechecks split size, path existence, class coverage, and split overlap.
- Added `smoke_check.py`:
  - Runs a minimal dataloader and model forward-pass check with random weights.
- Added `README.md` and `requirements.txt` for setup and run commands.
- Updated InceptionV3 construction to pass `init_weights` explicitly, avoiding torchvision's default-initialization warning during smoke checks.

Validation performed in this environment:

- Rechecked existing split files before implementation:
  - `train=43040`, `val=9223`, `test=9223`
  - `raw=39`, `plant=15`, `disease=22`
  - duplicate image paths across all splits: `0`
  - missing image paths: `0`
  - train/val/test overlap: `0`
- Confirmed `D:\Python\python.exe` can run this implementation:
  - Python `3.12.4`
  - PyTorch `2.5.1+cpu`
  - torchvision `0.20.1+cpu`
  - CUDA unavailable in this interpreter.
- Ran `verify_splits.py` successfully with `D:\Python\python.exe`.
- Ran `smoke_check.py` successfully with `D:\Python\python.exe`:
  - input image batch: `(2, 3, 299, 299)`
  - `plant_output`: `(2, 15)`
  - `disease_output`: `(2, 22)`
  - `plant_output_t`: `(2, 15)`
  - `disease_output_t`: `(2, 22)`
