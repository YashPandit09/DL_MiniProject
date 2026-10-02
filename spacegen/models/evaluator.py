"""CNN evaluator (T20, Tech Spec 4.2): scores a layout from its 4-channel raster.

  4 x [Conv 3x3 (padding 1) -> BatchNorm -> ReLU -> MaxPool 2]      128 px -> 8 px
  Flatten (64 x 8 x 8 = 4096) -> Dropout(0.3) -> Linear(4096, 128) -> ReLU
  head_valid: Linear(128, 1), a logit (BCE with logits in the loss, Sigmoid at inference)
  head_score: Linear(128, 1), the rule quality score as a regression

Loss: L = BCE(valid) + lambda_score * MSE (or Huber)(score), on every sample, because the
score is defined for invalid layouts too. The checker stays the authority on validity; the
evaluator is a CNN study, a possible surrogate and a learned ranker (Tech Spec 4.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from spacegen.config import DEFAULT_CONFIG, load_config


@dataclass(frozen=True)
class EvaluatorConfig:
    channels: tuple[int, ...] = (16, 32, 64, 64)
    hidden: int = 128
    dropout: float = 0.3
    lambda_score: float = 1.0
    score_loss: str = "mse"  # or "huber"
    huber_delta: float = 0.1
    pos_weight: float | None = None  # BCE weight of the valid class
    batch: int = 256
    lr: float = 1e-3

    def __post_init__(self):
        if self.score_loss not in ("mse", "huber"):
            raise ValueError(f"score_loss must be mse or huber, got {self.score_loss!r}")


def load_evaluator_config(path: Path = DEFAULT_CONFIG, **overrides) -> EvaluatorConfig:
    raw = {**load_config(path)["evaluator"], **overrides}
    return EvaluatorConfig(**{**raw, "channels": tuple(raw["channels"])})


class Evaluator(nn.Module):
    def __init__(self, config: EvaluatorConfig = EvaluatorConfig(), pixels: int = 128, in_channels: int = 4):
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        for c_in, c_out in zip((in_channels, *config.channels), config.channels):
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
        side = pixels // 2 ** len(config.channels)
        if side < 1 or pixels % 2 ** len(config.channels):
            raise ValueError(f"{pixels} px cannot be halved {len(config.channels)} times")
        self.features = nn.Sequential(*blocks, nn.Flatten())
        self.dense = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.channels[-1] * side * side, config.hidden),
                                   nn.ReLU())
        self.head_valid = nn.Linear(config.hidden, 1)
        self.head_score = nn.Linear(config.hidden, 1)

    def forward(self, raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Validity logits (B,) and predicted quality scores (B,) for rasters (B, 4, P, P)."""
        h = self.dense(self.features(raster))
        return self.head_valid(h).squeeze(1), self.head_score(h).squeeze(1)


@dataclass
class EvaluatorLoss:
    total: torch.Tensor
    bce: torch.Tensor
    score: torch.Tensor


def evaluator_loss(valid_logit: torch.Tensor, score: torch.Tensor, valid: torch.Tensor, quality: torch.Tensor,
                   config: EvaluatorConfig) -> EvaluatorLoss:
    """BCE on the validity logit plus lambda_score times the score regression, batch means."""
    pos_weight = None if config.pos_weight is None else torch.tensor(config.pos_weight, device=valid.device)
    bce = F.binary_cross_entropy_with_logits(valid_logit, valid, pos_weight=pos_weight)
    if config.score_loss == "mse":
        score_loss = F.mse_loss(score, quality)
    else:
        score_loss = F.huber_loss(score, quality, delta=config.huber_delta)
    return EvaluatorLoss(bce + config.lambda_score * score_loss, bce, score_loss)
