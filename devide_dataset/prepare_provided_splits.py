"""Validate supplied split CSVs and build a self-contained experiment data directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABEL_COLUMNS = ("label_raw", "label_plant", "label_disease")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root", type=Path,
        help="Replace the machine-specific prefix with IMAGE_ROOT/raw_class/image-name",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "raw_class", *LABEL_COLUMNS}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"{path} is empty or missing columns: {sorted(required)}")
    return rows


def name_maps(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    raw: dict[str, int] = {}
    plants: dict[str, int] = {}
    diseases: dict[str, int] = {}
    for row in rows:
        raw_name = row["raw_class"]
        if "___" not in raw_name:
            raise RuntimeError(f"Expected a plant___disease class, found {raw_name!r}")
        plant_name, disease_name = raw_name.split("___", 1)
        mappings = (
            (raw, raw_name, int(row["label_raw"])),
            (plants, plant_name, int(row["label_plant"])),
            (diseases, disease_name, int(row["label_disease"])),
        )
        for mapping, name, index in mappings:
            previous = mapping.setdefault(name, index)
            if previous != index:
                raise RuntimeError(f"Inconsistent label for {name!r}: {previous} versus {index}")
    return raw, plants, diseases


def ensure_contiguous(mapping: dict[str, int], label: str) -> None:
    observed = sorted(mapping.values())
    if observed != list(range(len(observed))):
        raise RuntimeError(f"{label} labels are not contiguous: {observed}")


def main() -> None:
    args = parse_args()
    split_paths = {"train": args.train, "val": args.val, "test": args.test}
    split_rows = {name: read_rows(path) for name, path in split_paths.items()}
    all_rows = [row for rows in split_rows.values() for row in rows]
    raw, plants, diseases = name_maps(all_rows)
    for mapping, label in ((raw, "raw"), (plants, "plant"), (diseases, "disease")):
        ensure_contiguous(mapping, label)

    path_sets = {name: {row["path"] for row in rows} for name, rows in split_rows.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = path_sets[left] & path_sets[right]
        if overlap:
            raise RuntimeError(f"{len(overlap)} paths overlap between {left} and {right}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "raw_class", *LABEL_COLUMNS]
    for split, rows in split_rows.items():
        if args.image_root:
            for row in rows:
                row["path"] = str(args.image_root / row["raw_class"] / Path(row["path"]).name)
        with (args.output_dir / f"{split}_split.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    class_maps = {
        "raw_to_idx": raw,
        "plant_to_idx": plants,
        "disease_to_idx": diseases,
        "idx_to_raw": {str(index): name for name, index in raw.items()},
        "idx_to_plant": {str(index): name for name, index in plants.items()},
        "idx_to_disease": {str(index): name for name, index in diseases.items()},
        "num_classes": {"raw": len(raw), "plant": len(plants), "disease": len(diseases)},
    }
    metadata = {
        "counts": {name: len(rows) for name, rows in split_rows.items()},
        "num_classes": class_maps["num_classes"],
        "class_balance_policy": "stratified natural splits; training-only raw-balanced sampling",
        "sampling_num_samples": len(split_rows["train"]),
    }
    for filename, value in (("class_maps.json", class_maps), ("split_metadata.json", metadata)):
        (args.output_dir / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
