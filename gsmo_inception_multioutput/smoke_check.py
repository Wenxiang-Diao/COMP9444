from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset import make_dataloader
from model import build_model


def main() -> None:
    data_dir = Path("devide_dataset/data")
    with open(data_dir / "class_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)

    loader = make_dataloader(
        data_dir / "train_split.csv",
        batch_size=2,
        train=True,
        num_workers=0,
    )
    batch = next(iter(loader))
    print("image shape:", tuple(batch["image"].shape))
    print("plant labels:", batch["label_plant"].tolist())
    print("disease labels:", batch["label_disease"].tolist())

    model = build_model(
        num_plants=int(maps["num_classes"]["plant"]),
        num_diseases=int(maps["num_classes"]["disease"]),
        weights="none",
    )
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image"])

    for name, tensor in outputs.items():
        print(name, tuple(tensor.shape))


if __name__ == "__main__":
    main()

