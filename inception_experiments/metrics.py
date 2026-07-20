from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .models import RAW_MODELS


def raw_label_lookups(class_maps: dict) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    raw_to_plant = np.empty(class_maps["num_classes"]["raw"], dtype=np.int64)
    raw_to_disease = np.empty(class_maps["num_classes"]["raw"], dtype=np.int64)
    legal: set[tuple[int, int]] = set()
    for raw_name, raw_index in class_maps["raw_to_idx"].items():
        if "___" in raw_name:
            plant_name, disease_name = raw_name.split("___", maxsplit=1)
        else:
            plant_name, disease_name = "Background", "without_leaves"
        plant_index = class_maps["plant_to_idx"][plant_name]
        disease_index = class_maps["disease_to_idx"][disease_name]
        raw_to_plant[raw_index] = plant_index
        raw_to_disease[raw_index] = disease_index
        legal.add((plant_index, disease_index))
    return raw_to_plant, raw_to_disease, legal


@dataclass
class PredictionAccumulator:
    model_name: str
    class_maps: dict
    total_loss: float = 0.0
    count: int = 0
    raw_true: list[int] = field(default_factory=list)
    raw_pred: list[int] = field(default_factory=list)
    plant_true: list[int] = field(default_factory=list)
    plant_pred: list[int] = field(default_factory=list)
    disease_true: list[int] = field(default_factory=list)
    disease_pred: list[int] = field(default_factory=list)

    def update(self, loss: float, batch: dict, outputs: dict) -> None:
        batch_size = int(batch["image"].shape[0])
        self.total_loss += float(loss) * batch_size
        self.count += batch_size
        raw_true = batch["label_raw"].detach().cpu().numpy()
        plant_true = batch["label_plant"].detach().cpu().numpy()
        disease_true = batch["label_disease"].detach().cpu().numpy()
        self.raw_true.extend(raw_true.tolist())
        self.plant_true.extend(plant_true.tolist())
        self.disease_true.extend(disease_true.tolist())

        raw_to_plant, raw_to_disease, _ = raw_label_lookups(self.class_maps)
        if self.model_name in RAW_MODELS:
            raw_pred = outputs["raw"].argmax(dim=1).detach().cpu().numpy()
            self.raw_pred.extend(raw_pred.tolist())
            self.plant_pred.extend(raw_to_plant[raw_pred].tolist())
            self.disease_pred.extend(raw_to_disease[raw_pred].tolist())
        else:
            self.plant_pred.extend(outputs["plant"].argmax(dim=1).detach().cpu().tolist())
            self.disease_pred.extend(outputs["disease"].argmax(dim=1).detach().cpu().tolist())

    def compute(self) -> dict[str, float]:
        metrics = {
            "loss": self.total_loss / max(self.count, 1),
            "plant_acc": accuracy_score(self.plant_true, self.plant_pred),
            "plant_macro_f1": f1_score(self.plant_true, self.plant_pred, average="macro", zero_division=0),
            "plant_weighted_f1": f1_score(self.plant_true, self.plant_pred, average="weighted", zero_division=0),
            "disease_acc": accuracy_score(self.disease_true, self.disease_pred),
            "disease_macro_f1": f1_score(self.disease_true, self.disease_pred, average="macro", zero_division=0),
            "disease_weighted_f1": f1_score(self.disease_true, self.disease_pred, average="weighted", zero_division=0),
        }
        if self.model_name in RAW_MODELS:
            metrics.update(
                {
                    "raw_acc": accuracy_score(self.raw_true, self.raw_pred),
                    "raw_macro_f1": f1_score(self.raw_true, self.raw_pred, average="macro", zero_division=0),
                    "raw_weighted_f1": f1_score(self.raw_true, self.raw_pred, average="weighted", zero_division=0),
                }
            )
        else:
            plant = np.asarray(self.plant_pred)
            disease = np.asarray(self.disease_pred)
            plant_true = np.asarray(self.plant_true)
            disease_true = np.asarray(self.disease_true)
            _, _, legal = raw_label_lookups(self.class_maps)
            metrics["both_acc"] = float(np.mean((plant == plant_true) & (disease == disease_true)))
            metrics["legal_combination_rate"] = float(
                np.mean([pair in legal for pair in zip(plant.tolist(), disease.tolist())])
            )
        return {key: float(value) for key, value in metrics.items()}
