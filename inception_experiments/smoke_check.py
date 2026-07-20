from __future__ import annotations

import json
from pathlib import Path

import torch

from .models import MODEL_NAMES, ModelConfig, build_model


def main() -> None:
    with open(Path("devide_dataset/data/class_maps.json"), encoding="utf-8") as handle:
        maps = json.load(handle)
    expected = {
        "single_raw": {"raw": (2, 39)},
        "plantxvit_raw": {"raw": (2, 39)},
        "plain_multi": {"plant": (2, 15), "disease": (2, 22)},
        "gsmo": {
            "plant": (2, 15),
            "disease": (2, 22),
            "plant_t": (2, 15),
            "disease_t": (2, 22),
        },
    }
    output_dir = Path("inception_experiments/runs/smoke_shapes")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = torch.randn(2, 3, 299, 299)
    for name in MODEL_NAMES:
        config = ModelConfig(
            name=name,
            num_raw=maps["num_classes"]["raw"],
            num_plants=maps["num_classes"]["plant"],
            num_diseases=maps["num_classes"]["disease"],
            weights="none",
        )
        model = build_model(config).eval()
        with torch.inference_mode():
            outputs = model(images)
        shapes = {key: tuple(value.shape) for key, value in outputs.items()}
        if shapes != expected[name]:
            raise AssertionError(f"{name}: expected {expected[name]}, got {shapes}")
        checkpoint = output_dir / f"{name}.pt"
        torch.save(model.state_dict(), checkpoint)
        reloaded = build_model(config)
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        print(name, shapes, "checkpoint reload OK")


if __name__ == "__main__":
    main()
