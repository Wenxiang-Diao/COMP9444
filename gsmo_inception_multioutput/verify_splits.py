from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from dataset import read_split_csv


DATA_DIR = Path("devide_dataset/data")


def main() -> None:
    splits = {
        name: read_split_csv(DATA_DIR / f"{name}_split.csv")
        for name in ["train", "val", "test"]
    }
    with open(DATA_DIR / "class_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)

    all_df = pd.concat(
        [df.assign(split=name) for name, df in splits.items()],
        ignore_index=True,
    )

    print("Split sizes:", {name: len(df) for name, df in splits.items()})
    print("Total rows:", len(all_df))
    print("Unique paths:", all_df["path"].nunique())
    print("Duplicate paths:", len(all_df) - all_df["path"].nunique())
    print("Missing image paths:", int((~all_df["path"].map(os.path.exists)).sum()))
    print("Num classes from class_maps:", maps["num_classes"])
    print(
        "Observed classes:",
        {
            "raw": int(all_df["label_raw"].nunique()),
            "plant": int(all_df["label_plant"].nunique()),
            "disease": int(all_df["label_disease"].nunique()),
        },
    )

    for name, df in splits.items():
        per_class = df.groupby("label_raw").size()
        print(
            name,
            {
                "raw_classes": int(df["label_raw"].nunique()),
                "plant_classes": int(df["label_plant"].nunique()),
                "disease_classes": int(df["label_disease"].nunique()),
                "min_per_raw": int(per_class.min()),
                "max_per_raw": int(per_class.max()),
            },
        )

    print(
        "Overlaps:",
        {
            "train_val": len(set(splits["train"]["path"]) & set(splits["val"]["path"])),
            "train_test": len(set(splits["train"]["path"]) & set(splits["test"]["path"])),
            "val_test": len(set(splits["val"]["path"]) & set(splits["test"]["path"])),
        },
    )


if __name__ == "__main__":
    main()

