from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class GSMoConfig:
    num_plants: int
    num_diseases: int
    dropout: float = 0.2
    weights: str = "imagenet"


def _build_inception_backbone(weights: str) -> nn.Module:
    """Build InceptionV3 without its classifier.

    The original GSMo-CNN paper found InceptionV3 to be a strong backbone. This
    implementation uses torchvision and disables the auxiliary classifier so
    forward() always returns a single feature tensor.
    """
    weights_obj = None
    if weights.lower() == "imagenet":
        try:
            weights_obj = models.Inception_V3_Weights.IMAGENET1K_V1
        except AttributeError:
            weights_obj = "IMAGENET1K_V1"
    elif weights.lower() not in {"none", "random"}:
        raise ValueError("weights must be one of: imagenet, none, random")

    try:
        backbone = models.inception_v3(
            weights=weights_obj,
            aux_logits=False,
            init_weights=weights_obj is None,
        )
    except TypeError:
        pretrained = weights.lower() == "imagenet"
        backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)

    backbone.fc = nn.Identity()
    return backbone


class GSMoInceptionV3(nn.Module):
    """Generalised Stacking Multi-output CNN with an InceptionV3 backbone.

    This mirrors the paper's "new_model" logic:
      1. shared feature vector x from the CNN backbone
      2. temporary plant and disease predictions from x
      3. final plant head receives [x, temporary disease logits]
      4. final disease head receives [x, temporary plant logits]

    The model returns logits, not softmax probabilities, so it can be trained
    with torch.nn.CrossEntropyLoss.
    """

    feature_dim = 2048

    def __init__(self, config: GSMoConfig):
        super().__init__()
        self.config = config
        self.backbone = _build_inception_backbone(config.weights)
        self.dropout = nn.Dropout(p=config.dropout)

        self.plant_output_t = nn.Linear(self.feature_dim, config.num_plants)
        self.disease_output_t = nn.Linear(self.feature_dim, config.num_diseases)

        self.plant_output = nn.Linear(
            self.feature_dim + config.num_diseases,
            config.num_plants,
        )
        self.disease_output = nn.Linear(
            self.feature_dim + config.num_plants,
            config.num_diseases,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = self.dropout(features)

        plant_t = self.plant_output_t(features)
        disease_t = self.disease_output_t(features)

        plant_features = torch.cat([features, disease_t], dim=1)
        disease_features = torch.cat([features, plant_t], dim=1)

        plant = self.plant_output(plant_features)
        disease = self.disease_output(disease_features)

        return {
            "plant_output": plant,
            "disease_output": disease,
            "plant_output_t": plant_t,
            "disease_output_t": disease_t,
        }


def build_model(
    num_plants: int,
    num_diseases: int,
    dropout: float = 0.2,
    weights: str = "imagenet",
) -> GSMoInceptionV3:
    return GSMoInceptionV3(
        GSMoConfig(
            num_plants=num_plants,
            num_diseases=num_diseases,
            dropout=dropout,
            weights=weights,
        )
    )
