import json
import random
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Basic Configuration
DATA_ROOT = Path(
    "Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/Plant_leave_diseases_dataset_with_augmentation")

IMAGE_SIZE = 299
BATCH_SIZE = 32
SEED = 42
THRESHOLD = 0.80

random.seed(SEED)
torch.manual_seed(SEED)

# Collect image paths (first 80% and last 20% per class, in original sorted order)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class_folders = sorted([
    d for d in DATA_ROOT.iterdir() 
    if d.is_dir() and not d.name.startswith(".") and d.name != "Background_without_leaves"
])

first_portion_samples = []
last_portion_samples = []

print("Extracting the first 80% and last 20% of images per class...")
for folder in class_folders:
    img_paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    img_paths.sort()

    split_idx = int(len(img_paths) * THRESHOLD)
    for img_path in img_paths[:split_idx]:
        first_portion_samples.append({
            "path": str(img_path.resolve()),
            "raw_class": folder.name,
        })
    for img_path in img_paths[split_idx:]:
        last_portion_samples.append({
            "path": str(img_path.resolve()),
            "raw_class": folder.name,
        })

df_first = pd.DataFrame(first_portion_samples)
df_last = pd.DataFrame(last_portion_samples)
df = pd.concat([df_first, df_last], ignore_index=True)
print(
    f"Collected {len(df)} images across {df['raw_class'].nunique()} raw classes "
    f"(first 80%: {len(df_first)}, last 20%: {len(df_last)})."
)

# Parse plant / disease names and build integer ID mappings
plants = set()
diseases = set()
for c in df["raw_class"].unique():
    plant, disease = c.split("___") if "___" in c else ("Background", "without_leaves")
    plants.add(plant)
    diseases.add(disease)

plant_to_idx = {p: i for i, p in enumerate(sorted(plants))}
disease_to_idx = {d: i for i, d in enumerate(sorted(diseases))}
raw_to_idx = {r: i for i, r in enumerate(sorted(df["raw_class"].unique()))}

def stratified_split_75_15_10(dataframe: pd.DataFrame):
    """Split into 75% train, 15% val, 10% test (stratified by label_raw)."""
    train_part, temp_part = train_test_split(
        dataframe, test_size=0.25, stratify=dataframe["label_raw"], random_state=SEED
    )
    val_part, test_part = train_test_split(
        temp_part, test_size=0.40, stratify=temp_part["label_raw"], random_state=SEED
    )
    return train_part, val_part, test_part


def add_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    labeled = dataframe.copy()
    labeled["label_raw"] = labeled["raw_class"].map(raw_to_idx)
    labeled["label_plant"] = labeled["raw_class"].apply(
        lambda x: plant_to_idx[x.split("___")[0] if "___" in x else "Background"]
    )
    labeled["label_disease"] = labeled["raw_class"].apply(
        lambda x: disease_to_idx[x.split("___")[1] if "___" in x else "without_leaves"]
    )
    return labeled


df_first = add_labels(df_first)
df_last = add_labels(df_last)

# Stratified split each portion (75/15/10), then merge
print("Splitting first 80% and last 20% into train/val/test (75/15/10), then merging...")
train_first, val_first, test_first = stratified_split_75_15_10(df_first)
train_last, val_last, test_last = stratified_split_75_15_10(df_last)

train_df = pd.concat([train_first, train_last], ignore_index=True)
val_df = pd.concat([val_first, val_last], ignore_index=True)
test_df = pd.concat([test_first, test_last], ignore_index=True)

# Save CSVs and label maps next to this script: devid_dataset/data/
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(OUT_DIR / "train_split.csv", index=False)
val_df.to_csv(OUT_DIR / "val_split.csv", index=False)
test_df.to_csv(OUT_DIR / "test_split.csv", index=False)

# name -> id maps used for training; idx_* are id -> name for eval / plots
class_maps = {
    "raw_to_idx": raw_to_idx,
    "plant_to_idx": plant_to_idx,
    "disease_to_idx": disease_to_idx,
    "idx_to_raw": {str(i): name for name, i in raw_to_idx.items()},
    "idx_to_plant": {str(i): name for name, i in plant_to_idx.items()},
    "idx_to_disease": {str(i): name for name, i in disease_to_idx.items()},
    "num_classes": {
        "raw": len(raw_to_idx),
        "plant": len(plant_to_idx),
        "disease": len(disease_to_idx),
    },
}
with open(OUT_DIR / "class_maps.json", "w", encoding="utf-8") as f:
    json.dump(class_maps, f, indent=2, ensure_ascii=False)

print(f"Saved splits and class_maps.json to {OUT_DIR.resolve()}/")
print(f"Split complete: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}.")

# Train: random flips, rotation, and color jitter for lighting / orientation variation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Val / test: resize and normalize only (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# PyTorch Dataset
class LeafDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label_raw": int(row["label_raw"]),
            "label_plant": int(row["label_plant"]),
            "label_disease": int(row["label_disease"])
        }


# Instantiate datasets
train_dataset = LeafDataset(train_df, transform=train_transforms)
val_dataset = LeafDataset(val_df, transform=val_transforms)
test_dataset = LeafDataset(test_df, transform=val_transforms)

# DataLoaders
# Enable pin_memory when CUDA is available for faster host-to-device transfer
use_cuda = torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=use_cuda)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_cuda)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_cuda)

print("DataLoaders ready for training.")