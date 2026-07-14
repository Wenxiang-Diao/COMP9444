from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_SIZE = 299
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def read_split_csv(path: str | Path) -> pd.DataFrame:
    """Read split CSVs generated under devide_dataset/data.

    The current CSVs are encoded with the local Windows Chinese encoding, while
    class_maps.json is UTF-8. Try UTF-8 first for portability, then fall back.
    """
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def build_transforms(train: bool, image_size: int = IMAGE_SIZE):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )


class LeafMultiOutputDataset(Dataset):
    """Dataset returning labels required by GSMo-CNN.

    Returns:
        image: Tensor [3, 299, 299]
        label_raw: joint crop-disease class id, 39 classes
        label_plant: plant/crop id, 15 classes
        label_disease: disease/status id, 22 classes
        path: source image path
        raw_class: source folder name
    """

    required_columns = {
        "path",
        "raw_class",
        "label_raw",
        "label_plant",
        "label_disease",
    }

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        missing = self.required_columns - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing split CSV columns: {sorted(missing)}")
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label_raw": int(row["label_raw"]),
            "label_plant": int(row["label_plant"]),
            "label_disease": int(row["label_disease"]),
            "path": row["path"],
            "raw_class": row["raw_class"],
        }


def make_dataloader(
    csv_path: str | Path,
    batch_size: int,
    train: bool,
    num_workers: int = 0,
    image_size: int = IMAGE_SIZE,
    pin_memory: bool = False,
) -> DataLoader:
    df = read_split_csv(csv_path)
    dataset = LeafMultiOutputDataset(
        df,
        transform=build_transforms(train=train, image_size=image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_splits(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        "train": read_split_csv(data_dir / "train_split.csv"),
        "val": read_split_csv(data_dir / "val_split.csv"),
        "test": read_split_csv(data_dir / "test_split.csv"),
    }

