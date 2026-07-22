from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_SIZE = 299
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_split(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def build_transform(train: bool) -> transforms.Compose:
    operations: list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class LeafDataset(Dataset):
    required_columns = {
        "path",
        "raw_class",
        "label_raw",
        "label_plant",
        "label_disease",
    }

    def __init__(self, frame: pd.DataFrame, train: bool) -> None:
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"Missing split columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transform(train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = self.transform(source.convert("RGB"))
        return {
            "image": image,
            "label_raw": int(row["label_raw"]),
            "label_plant": int(row["label_plant"]),
            "label_disease": int(row["label_disease"]),
            "path": row["path"],
        }


def make_loader(
    csv_path: Path,
    batch_size: int,
    train: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
    sampling: str = "none",
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    frame = read_split(csv_path)
    sampler = None
    if train and sampling == "raw-balanced":
        counts = frame["label_raw"].value_counts()
        sample_weights = frame["label_raw"].map(lambda label: 1.0 / counts[label]).to_numpy()
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(frame),
            replacement=True,
            generator=generator,
        )
    elif sampling != "none":
        raise ValueError(f"Unknown sampling strategy: {sampling}")
    return DataLoader(
        LeafDataset(frame, train=train),
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )
