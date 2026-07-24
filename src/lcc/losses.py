import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .config import TrainConfig
from .data import Sample


def build_target_distribution(
    targets: torch.Tensor,
    num_classes: int,
    label_smoothing: float = 0.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if targets.ndim == 2:
        return targets.to(dtype=dtype or targets.dtype)

    if targets.ndim != 1:
        raise ValueError(f"Targets must be rank-1 indices or rank-2 distributions, got shape {tuple(targets.shape)}.")

    target_dist = F.one_hot(targets, num_classes=num_classes).to(dtype=dtype or torch.float32)
    if label_smoothing > 0.0:
        target_dist = target_dist * (1.0 - label_smoothing) + label_smoothing / num_classes
    return target_dist


class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if weight is None:
            weight = torch.tensor([], dtype=torch.float32)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.register_buffer("weight", weight.float())
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_dist = build_target_distribution(
            targets,
            num_classes=logits.size(1),
            label_smoothing=self.label_smoothing if targets.ndim == 1 else 0.0,
            dtype=logits.dtype,
        )
        log_probs = F.log_softmax(logits, dim=1)
        if self.weight.numel() > 0:
            loss = -(target_dist * log_probs * self.weight.unsqueeze(0)).sum(dim=1)
        else:
            loss = -(target_dist * log_probs).sum(dim=1)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if alpha is None:
            alpha = torch.tensor([], dtype=torch.float32)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_dist = build_target_distribution(
            targets,
            num_classes=logits.size(1),
            label_smoothing=self.label_smoothing if targets.ndim == 1 else 0.0,
            dtype=logits.dtype,
        )

        target_probs = (probs * target_dist).sum(dim=1)
        ce_loss = -(target_dist * log_probs).sum(dim=1)
        loss = ((1.0 - target_probs).clamp_min(1e-8) ** self.gamma) * ce_loss

        if self.alpha.numel() > 0:
            loss = loss * (target_dist * self.alpha).sum(dim=1)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def compute_train_class_weights(train_samples: Sequence[Sample], class_names: Sequence[str]) -> Dict[str, float]:
    counts = {class_name: 0 for class_name in class_names}
    for sample in train_samples:
        if sample.class_name in counts:
            counts[sample.class_name] += 1

    total_samples = sum(counts.values())
    if total_samples == 0:
        raise ValueError("Training split is empty; cannot compute class weights.")

    missing = [class_name for class_name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Cannot compute class weights because these classes have zero training samples: {missing}")

    return {
        class_name: total_samples / (len(class_names) * counts[class_name])
        for class_name in class_names
    }


def parse_manual_class_weights(weights_arg: str, class_names: Sequence[str]) -> Dict[str, float]:
    values = [item.strip() for item in weights_arg.split(",") if item.strip()]
    if len(values) != len(class_names):
        raise ValueError(
            f"--class-weights must provide exactly {len(class_names)} comma-separated values "
            f"for {list(class_names)}."
        )

    return {
        class_name: float(value)
        for class_name, value in zip(class_names, values)
    }


def resolve_class_weights(
    config: TrainConfig,
    train_samples: Sequence[Sample],
    device: torch.device,
    class_names: Sequence[str],
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
    if config.class_weighting == "none":
        return None, None

    if config.class_weighting == "balanced":
        class_weights = compute_train_class_weights(train_samples, class_names)
    elif config.class_weighting == "manual":
        if not config.class_weights_arg:
            raise ValueError("--class-weights is required when --class-weighting manual is used.")
        class_weights = parse_manual_class_weights(config.class_weights_arg, class_names)
    else:
        raise ValueError(f"Unsupported class weighting: {config.class_weighting}")

    weight_tensor = torch.tensor(
        [class_weights[class_name] for class_name in class_names],
        dtype=torch.float32,
        device=device,
    )
    return weight_tensor, class_weights


def create_scheduler(optimizer: AdamW, config: TrainConfig):
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    if config.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def step_scheduler(
    scheduler: Optional[object],
    scheduler_name: str,
    valid_loss: float,
) -> None:
    if scheduler is None:
        return
    if scheduler_name == "plateau":
        scheduler.step(valid_loss)
        return
    scheduler.step()


def get_current_lr(optimizer: AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def create_criterion(config: TrainConfig, weight_tensor: Optional[torch.Tensor]) -> nn.Module:
    if config.loss == "cross_entropy":
        return SoftTargetCrossEntropyLoss(weight=weight_tensor, label_smoothing=config.label_smoothing)
    if config.loss == "focal":
        return FocalLoss(alpha=weight_tensor, gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
    raise ValueError(f"Unsupported loss: {config.loss}")


def has_mix_augmentation(config: TrainConfig) -> bool:
    return config.mixup_alpha > 0.0 or config.cutmix_alpha > 0.0


def rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    cut_ratio = np.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    center_x = np.random.randint(width)
    center_y = np.random.randint(height)

    x1 = np.clip(center_x - cut_w // 2, 0, width)
    x2 = np.clip(center_x + cut_w // 2, 0, width)
    y1 = np.clip(center_y - cut_h // 2, 0, height)
    y2 = np.clip(center_y + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mix_augmentation(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    config: TrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not has_mix_augmentation(config) or labels.size(0) < 2 or random.random() > config.mix_prob:
        return images, labels

    permutation = torch.randperm(labels.size(0), device=labels.device)
    use_cutmix = False
    if config.cutmix_alpha > 0.0 and config.mixup_alpha > 0.0:
        use_cutmix = random.random() < config.mix_switch_prob
    elif config.cutmix_alpha > 0.0:
        use_cutmix = True

    if use_cutmix:
        lam = float(np.random.beta(config.cutmix_alpha, config.cutmix_alpha))
        x1, y1, x2, y2 = rand_bbox(images.size(-1), images.size(-2), lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        patch_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - patch_area / float(images.size(-1) * images.size(-2))
    else:
        lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
        mixed_images = images * lam + images[permutation] * (1.0 - lam)

    labels_a = build_target_distribution(
        labels,
        num_classes=num_classes,
        label_smoothing=config.label_smoothing,
        dtype=images.dtype,
    )
    labels_b = build_target_distribution(
        labels[permutation],
        num_classes=num_classes,
        label_smoothing=config.label_smoothing,
        dtype=images.dtype,
    )
    mixed_labels = labels_a * lam + labels_b * (1.0 - lam)
    return mixed_images, mixed_labels
