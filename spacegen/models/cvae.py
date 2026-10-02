"""Conditional VAE generator (T19, Tech Spec 4.1): learns p(x | c), layouts given the room.

  encoder q(z | x, c): [x ; c] -> depth x (Linear, BatchNorm, activation, Dropout) -> mu, logvar
  decoder p(x | z, c): [z ; c] -> the same hidden stack -> positions (K, 2), rotation logits (K, 4)
  z = mu + exp(logvar / 2) * eps, eps ~ N(0, I)   (reparameterization)

The position head is a Sigmoid by default; "linear" (E7) is clamped to [0, 1] only when
layouts are generated, so training still sees its errors outside the room. The rotation head
returns logits: softmax happens inside F.cross_entropy, and at inference only.

Loss per sample, masked by presence (Hadamard product, Appendix N):
  recon   = sum_k m_k * position_loss_k + lambda_rot * sum_k r_k * CE(logits_k, rotation_k)
            (r_k = m_k except for items whose rotation is not learned: rot_symmetry 4)
  KL      = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))
  overlap = sum_{i<j} pen_ij^2 on the decoded positions in meters and the true rotations
  L       = mean(recon) + beta * mean(KL) + lambda_overlap * mean(overlap)

Every switch the Week 2 experiments sweep is a CVAEConfig field (configs/default.yaml, cvae).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from spacegen import geometry
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import TrainingTensors, condition_dim, split_targets, target_dim, unpack_conditions

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.01), "elu": nn.ELU, "gelu": nn.GELU,
               "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
POSITION_LOSSES = ("mse", "mae", "huber")


@dataclass(frozen=True)
class CVAEConfig:
    num_slots: int = 6  # K
    latent: int = 16
    hidden: int = 256
    depth: int = 2  # hidden layers in the encoder and in the decoder
    activation: str = "relu"
    batch_norm: bool = True
    dropout: float = 0.1
    position_head: str = "sigmoid"  # or "linear"
    position_loss: str = "mse"  # or "mae", "huber"
    huber_delta: float = 0.05
    lambda_rot: float = 1.0
    lambda_overlap: float = 0.0
    overlap_margin: float = 0.0

    def __post_init__(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {self.activation!r}")
        if self.position_head not in ("sigmoid", "linear"):
            raise ValueError(f"position_head must be sigmoid or linear, got {self.position_head!r}")
        if self.position_loss not in POSITION_LOSSES:
            raise ValueError(f"position_loss must be one of {POSITION_LOSSES}, got {self.position_loss!r}")
        if self.depth < 1 or not 0 <= self.dropout < 1:
            raise ValueError("depth must be at least 1 and dropout in [0, 1)")


def load_cvae_config(path: Path = DEFAULT_CONFIG, **overrides) -> CVAEConfig:
    return CVAEConfig(**{**load_config(path)["cvae"], **overrides})


@dataclass
class CVAEOutput:
    positions: torch.Tensor  # (B, K, 2) normalized centres (u, v)
    rot_logits: torch.Tensor  # (B, K, 4)
    mu: torch.Tensor  # (B, latent)
    logvar: torch.Tensor  # (B, latent)
    z: torch.Tensor  # (B, latent)


class CVAE(nn.Module):
    def __init__(self, config: CVAEConfig = CVAEConfig()):
        super().__init__()
        self.config = config
        k = config.num_slots
        self.encoder = _hidden_stack(target_dim(k) + condition_dim(k), config)
        self.to_mu = nn.Linear(config.hidden, config.latent)
        self.to_logvar = nn.Linear(config.hidden, config.latent)
        self.decoder = _hidden_stack(config.latent + condition_dim(k), config)
        self.head_pos = nn.Linear(config.hidden, 2 * k)
        self.head_rot = nn.Linear(config.hidden, 4 * k)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z | x, c): mean and log-variance, each (B, latent)."""
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(x | z, c): positions (B, K, 2), unclamped for a linear head, and rotation logits (B, K, 4)."""
        h = self.decoder(torch.cat([z, c], dim=1))
        positions = self.head_pos(h)
        if self.config.position_head == "sigmoid":
            positions = torch.sigmoid(positions)
        k = self.config.num_slots
        return positions.view(-1, k, 2), self.head_rot(h).view(-1, k, 4)

    def forward(self, x: torch.Tensor, c: torch.Tensor, sample: bool = True,
                generator: torch.Generator | None = None) -> CVAEOutput:
        """Encode, draw z (or take z = mu when sample is False), decode."""
        mu, logvar = self.encode(x, c)
        z = reparameterize(mu, logvar, generator) if sample else mu
        positions, logits = self.decode(z, c)
        return CVAEOutput(positions, logits, mu, logvar, z)

    @torch.no_grad()
    def generate(self, c: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Layouts for conditions: z ~ N(0, I), decoded; positions clamped to [0, 1]."""
        z = torch.randn(c.shape[0], self.config.latent, generator=generator, device=c.device, dtype=c.dtype)
        positions, logits = self.decode(z, c)
        return positions.clamp(0.0, 1.0), logits


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """z = mu + sigma * eps with sigma = exp(logvar / 2): the sample stays differentiable in mu and logvar."""
    eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu + torch.exp(0.5 * logvar) * eps


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)) per sample, closed form (Appendix I): (B,)."""
    return -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)


def active_units(mu: torch.Tensor, logvar: torch.Tensor, threshold: float = 0.01) -> int:
    """Latent dimensions whose KL, averaged over the samples, exceeds `threshold` nats.

    0 means posterior collapse: the decoder ignores z (Tech Spec 4.1).
    """
    per_dimension = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
    return int((per_dimension > threshold).sum())


def position_error(predicted: torch.Tensor, target: torch.Tensor, config: CVAEConfig) -> torch.Tensor:
    """Per item, summed over u and v: squared, absolute or Huber error (Appendix D). (B, K)."""
    if config.position_loss == "mse":
        error = (predicted - target) ** 2
    elif config.position_loss == "mae":
        error = (predicted - target).abs()
    else:
        error = F.huber_loss(predicted, target, reduction="none", delta=config.huber_delta)
    return error.sum(dim=-1)


def decoded_overlap(positions: torch.Tensor, c: torch.Tensor, rotations: torch.Tensor, config: CVAEConfig) -> torch.Tensor:
    """sum_{i<j} pen_ij^2 per sample, for decoded positions placed in meters with the given rotations."""
    room, mask, size = unpack_conditions(c, config.num_slots)
    center = positions * room[:, None, :]
    return geometry.overlap_penalty(center, geometry.effective_size(size, rotations), margin=config.overlap_margin,
                                    mask=mask > 0.5)


@dataclass
class LossTerms:
    total: torch.Tensor  # what backward() is called on
    recon: torch.Tensor  # batch means from here on
    position: torch.Tensor
    rotation: torch.Tensor
    kl: torch.Tensor
    overlap: torch.Tensor


def cvae_loss(output: CVAEOutput, batch: TrainingTensors, beta: float, config: CVAEConfig) -> LossTerms:
    """The masked CVAE loss of Tech Spec 4.1 for a mini-batch."""
    target_positions, target_rotations = split_targets(batch.x)
    classes = target_rotations.argmax(dim=-1)  # canonical rotation class; masked where absent
    position = (batch.mask * position_error(output.positions, target_positions, config)).sum(dim=1)
    cross_entropy = F.cross_entropy(output.rot_logits.reshape(-1, 4), classes.reshape(-1),
                                    reduction="none").view_as(classes)
    rotation = (batch.rot_mask * cross_entropy).sum(dim=1)
    recon = position + config.lambda_rot * rotation
    kl = kl_divergence(output.mu, output.logvar)
    total = recon.mean() + beta * kl.mean()
    overlap = torch.zeros_like(recon)
    if config.lambda_overlap > 0:
        overlap = decoded_overlap(output.positions, batch.c, classes, config)
        total = total + config.lambda_overlap * overlap.mean()
    return LossTerms(total, recon.mean(), position.mean(), rotation.mean(), kl.mean(), overlap.mean())


def _hidden_stack(in_features: int, config: CVAEConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(config.depth):
        layers.append(nn.Linear(in_features if i == 0 else config.hidden, config.hidden))
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden))
        layers.append(ACTIVATIONS[config.activation]())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
    return nn.Sequential(*layers)
