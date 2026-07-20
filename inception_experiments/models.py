from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


MODEL_NAMES = ("single_raw", "plain_multi", "gsmo", "plantxvit_raw")
RAW_MODELS = ("single_raw", "plantxvit_raw")


def build_backbone(weights: str) -> nn.Module:
    if weights == "imagenet":
        weights_object = models.Inception_V3_Weights.IMAGENET1K_V1
    elif weights in {"none", "random"}:
        weights_object = None
    else:
        raise ValueError("weights must be imagenet, none, or random")
    if weights_object is None:
        backbone = models.inception_v3(
            weights=None,
            aux_logits=False,
            init_weights=True,
        )
    else:
        # Torchvision's pretrained InceptionV3 weights require construction
        # with AuxLogits enabled. Remove that training-only head after loading.
        backbone = models.inception_v3(weights=weights_object, aux_logits=True)
        backbone.AuxLogits = None
        backbone.aux_logits = False
    backbone.fc = nn.Identity()
    return backbone


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_raw: int
    num_plants: int
    num_diseases: int
    weights: str = "imagenet"
    dropout: float = 0.2


class InceptionExperimentModel(nn.Module):
    feature_dim = 2048

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {config.name!r}")
        self.config = config
        self.backbone = build_backbone(config.weights)
        self.dropout = nn.Dropout(config.dropout)

        if config.name == "single_raw":
            self.raw_head = nn.Linear(self.feature_dim, config.num_raw)
        else:
            self.plant_t = nn.Linear(self.feature_dim, config.num_plants)
            self.disease_t = nn.Linear(self.feature_dim, config.num_diseases)
            if config.name == "gsmo":
                self.plant_head = nn.Linear(
                    self.feature_dim + config.num_diseases,
                    config.num_plants,
                )
                self.disease_head = nn.Linear(
                    self.feature_dim + config.num_plants,
                    config.num_diseases,
                )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(images))
        if self.config.name == "single_raw":
            return {"raw": self.raw_head(features)}

        plant_t = self.plant_t(features)
        disease_t = self.disease_t(features)
        if self.config.name == "plain_multi":
            return {"plant": plant_t, "disease": disease_t}
        return {
            "plant": self.plant_head(torch.cat([features, disease_t], dim=1)),
            "disease": self.disease_head(torch.cat([features, plant_t], dim=1)),
            "plant_t": plant_t,
            "disease_t": disease_t,
        }


class PlantXViTReproduction(nn.Module):
    """Lightweight CNN/Transformer reproduction for the optional E5 control.

    The PlantXViT paper does not provide official source code.  This deliberately
    named reproduction follows the published high-level design rather than
    claiming to be the authors' exact implementation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.GELU(),
            nn.AdaptiveAvgPool2d((10, 10)),
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, 101, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=384,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(width)
        self.raw_head = nn.Linear(width, config.num_raw)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.features(images).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(images.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1) + self.position)
        return {"raw": self.raw_head(self.norm(encoded[:, 0]))}


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "plantxvit_raw":
        return PlantXViTReproduction(config)
    return InceptionExperimentModel(config)
