import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def canonical_class_name(folder_name: str) -> str:
    if folder_name.startswith("adenocarcinoma"):
        return "adenocarcinoma"
    if folder_name.startswith("large.cell.carcinoma"):
        return "large.cell.carcinoma"
    if folder_name == "normal":
        return "normal"
    if folder_name.startswith("squamous.cell.carcinoma"):
        return "squamous.cell.carcinoma"
    raise ValueError(f"Unknown class folder: {folder_name}")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


@dataclass(frozen=True)
class Sample:
    image_path: Path
    class_name: str
    label: int
    split: str


def collect_samples(split_dir: Path, split_name: str) -> List[Sample]:
    samples: List[Sample] = []
    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        class_name = canonical_class_name(class_dir.name)
        label = CLASS_TO_INDEX[class_name]
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for image_path in image_paths:
            samples.append(
                Sample(
                    image_path=image_path,
                    class_name=class_name,
                    label=label,
                    split=split_name,
                )
            )
    return samples


class LungCancerDataset(Dataset):
    def __init__(self, samples: List[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, sample.label, str(sample.image_path)


class FeatureSqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.reduce(scale)
        scale = self.act(scale)
        scale = self.expand(scale)
        return x * self.gate(scale)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.gate(attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * self.gate(attention)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class AttentionAugmentedModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_attention: str):
        super().__init__()
        if not hasattr(backbone, "forward_features") or not hasattr(backbone, "forward_head"):
            raise ValueError(
                f"Model {backbone.__class__.__name__} does not expose forward_features/forward_head, "
                f"so feature attention '{feature_attention}' cannot be attached safely."
            )

        self.backbone = backbone
        self.feature_attention_name = feature_attention
        num_features = getattr(backbone, "num_features", None)
        if num_features is None:
            raise ValueError("Backbone does not expose num_features, so feature attention cannot be initialized.")

        if feature_attention == "se":
            self.feature_attention = FeatureSqueezeExcite(num_features)
        elif feature_attention == "cbam":
            self.feature_attention = CBAM(num_features)
        else:
            raise ValueError(f"Unsupported feature attention: {feature_attention}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        x = self.feature_attention(x)
        return self.backbone.forward_head(x, pre_logits=False)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access a GPU. "
                "Install a CUDA-enabled PyTorch build and verify that torch.cuda.is_available() returns True."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_num_workers(requested_workers: Optional[int], device: torch.device) -> int:
    if requested_workers is not None:
        return max(requested_workers, 0)

    cpu_count = os.cpu_count() or 0
    if device.type == "cuda" and cpu_count > 1:
        return min(4, cpu_count)
    return 0


def resolve_amp_enabled(requested_amp: Optional[bool], device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if requested_amp is None:
        return True
    return requested_amp


def autocast_context(device: torch.device, amp_enabled: bool):
    if device.type == "cuda" and amp_enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[Sample]]]:
    train_transform, eval_transform = build_transforms(image_size)

    samples_by_split = {
        "train": collect_samples(data_dir / "train", "train"),
        "valid": collect_samples(data_dir / "valid", "valid"),
        "test": collect_samples(data_dir / "test", "test"),
    }

    datasets = {
        "train": LungCancerDataset(samples_by_split["train"], train_transform),
        "valid": LungCancerDataset(samples_by_split["valid"], eval_transform),
        "test": LungCancerDataset(samples_by_split["test"], eval_transform),
    }

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        datasets["valid"],
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, valid_loader, test_loader, samples_by_split


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    feature_attention: str,
) -> nn.Module:
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if feature_attention == "none":
        return backbone
    return AttentionAugmentedModel(backbone, feature_attention=feature_attention)


def compute_train_class_weights(train_samples: List[Sample]) -> Dict[str, float]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for sample in train_samples:
        counts[sample.class_name] += 1

    total_samples = sum(counts.values())
    if total_samples == 0:
        raise ValueError("Training split is empty; cannot compute class weights.")

    return {
        class_name: total_samples / (len(CLASS_NAMES) * counts[class_name])
        for class_name in CLASS_NAMES
    }


def parse_manual_class_weights(weights_arg: str) -> Dict[str, float]:
    values = [item.strip() for item in weights_arg.split(",") if item.strip()]
    if len(values) != len(CLASS_NAMES):
        raise ValueError(
            f"--class-weights must provide exactly {len(CLASS_NAMES)} comma-separated values "
            f"for {CLASS_NAMES}."
        )

    return {
        class_name: float(value)
        for class_name, value in zip(CLASS_NAMES, values)
    }


def resolve_class_weights(
    args: argparse.Namespace,
    train_samples: List[Sample],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
    if args.class_weighting == "none":
        return None, None

    if args.class_weighting == "balanced":
        class_weights = compute_train_class_weights(train_samples)
    elif args.class_weighting == "manual":
        if not args.class_weights:
            raise ValueError("--class-weights is required when --class-weighting manual is used.")
        class_weights = parse_manual_class_weights(args.class_weights)
    else:
        raise ValueError(f"Unsupported class weighting: {args.class_weighting}")

    weight_tensor = torch.tensor(
        [class_weights[class_name] for class_name in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    return weight_tensor, class_weights


def create_scheduler(optimizer: AdamW, args: argparse.Namespace):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


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


def create_criterion(args: argparse.Namespace, weight_tensor: Optional[torch.Tensor]) -> nn.Module:
    if args.loss == "cross_entropy":
        return SoftTargetCrossEntropyLoss(weight=weight_tensor, label_smoothing=args.label_smoothing)
    if args.loss == "focal":
        return FocalLoss(alpha=weight_tensor, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    raise ValueError(f"Unsupported loss: {args.loss}")


def has_mix_augmentation(args: argparse.Namespace) -> bool:
    return args.mixup_alpha > 0.0 or args.cutmix_alpha > 0.0


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
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not has_mix_augmentation(args) or labels.size(0) < 2 or random.random() > args.mix_prob:
        return images, labels

    permutation = torch.randperm(labels.size(0), device=labels.device)
    use_cutmix = False
    if args.cutmix_alpha > 0.0 and args.mixup_alpha > 0.0:
        use_cutmix = random.random() < args.mix_switch_prob
    elif args.cutmix_alpha > 0.0:
        use_cutmix = True

    if use_cutmix:
        lam = float(np.random.beta(args.cutmix_alpha, args.cutmix_alpha))
        x1, y1, x2, y2 = rand_bbox(images.size(-1), images.size(-2), lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        patch_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - patch_area / float(images.size(-1) * images.size(-2))
    else:
        lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
        mixed_images = images * lam + images[permutation] * (1.0 - lam)

    labels_a = build_target_distribution(
        labels,
        num_classes=len(CLASS_NAMES),
        label_smoothing=args.label_smoothing,
        dtype=images.dtype,
    )
    labels_b = build_target_distribution(
        labels[permutation],
        num_classes=len(CLASS_NAMES),
        label_smoothing=args.label_smoothing,
        dtype=images.dtype,
    )
    mixed_labels = labels_a * lam + labels_b * (1.0 - lam)
    return mixed_images, mixed_labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    scaler: Optional[object],
    amp_enabled: bool,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    non_blocking = device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        metric_labels = labels

        optimizer.zero_grad(set_to_none=True)
        images, targets = apply_mix_augmentation(images, labels, args)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(metric_labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_paths: List[str] = []
    non_blocking = device.type == "cuda"

    for images, labels, paths in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_paths.extend(paths)

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "labels": all_labels,
        "preds": all_preds,
        "paths": all_paths,
        "probs": all_probs,
        "report": report,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES)))),
    }


def save_predictions(output_path: Path, eval_result: Dict[str, object]) -> None:
    header = ["image_path", "true_label", "pred_label"] + [f"prob_{name}" for name in CLASS_NAMES]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for path, true_label, pred_label, probs in zip(
            eval_result["paths"],
            eval_result["labels"],
            eval_result["preds"],
            eval_result["probs"],
        ):
            writer.writerow(
                [
                    path,
                    CLASS_NAMES[true_label],
                    CLASS_NAMES[pred_label],
                    *[f"{prob:.6f}" for prob in probs],
                ]
            )


def save_confusion_matrix(output_path: Path, matrix: np.ndarray) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])


def save_json(output_path: Path, payload: Dict[str, object]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_run_summary(
    args: argparse.Namespace,
    train_history: List[Dict[str, float]],
    best_epoch: int,
    best_val: Dict[str, object],
    test_result: Dict[str, object],
    samples_by_split: Dict[str, List[Sample]],
    elapsed_seconds: float,
    device: torch.device,
    amp_enabled: bool,
    pin_memory: bool,
    gpu_name: Optional[str],
    class_weights: Optional[Dict[str, float]],
) -> Dict[str, object]:
    class_distribution: Dict[str, Dict[str, int]] = {}
    for split_name, samples in samples_by_split.items():
        distribution = {class_name: 0 for class_name in CLASS_NAMES}
        for sample in samples:
            distribution[sample.class_name] += 1
        class_distribution[split_name] = distribution

    return {
        "config": {
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "loss": args.loss,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma,
            "class_weighting": args.class_weighting,
            "class_weights": class_weights,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "mix_prob": args.mix_prob,
            "mix_switch_prob": args.mix_switch_prob,
            "feature_attention": args.feature_attention,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device_request": args.device,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "deterministic": args.deterministic,
            "pin_memory": pin_memory,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
        },
        "dataset": {
            "data_dir": str(args.data_dir),
            "class_names": CLASS_NAMES,
            "split_sizes": {split_name: len(samples) for split_name, samples in samples_by_split.items()},
            "class_distribution": class_distribution,
        },
        "training_history": train_history,
        "best_epoch": best_epoch,
        "best_validation": {
            "loss": best_val["loss"],
            "accuracy": best_val["accuracy"],
            "report": best_val["report"],
        },
        "test": {
            "loss": test_result["loss"],
            "accuracy": test_result["accuracy"],
            "report": test_result["report"],
        },
        "elapsed_seconds": elapsed_seconds,
    }


def resolve_default_data_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "附件" / "Data",
        repo_root / "附件" / "Data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_dir = resolve_default_data_dir(repo_root)

    parser = argparse.ArgumentParser(
        description="Problem 2 baseline: timm EfficientNet-B0 for four-class lung cancer image classification."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="efficientnet_b0")
    parser.add_argument("--pretrained", action="store_true", help="Use timm pretrained weights if available.")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy", help="Training loss; label smoothing applies to both options.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor used by cross-entropy and focal loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focusing parameter when --loss focal is selected.")
    parser.add_argument("--class-weighting", choices=["none", "balanced", "manual"], default="none")
    parser.add_argument("--class-weights", type=str, default=None, help="Comma-separated weights in CLASS_NAMES order when class-weighting=manual.")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Enable MixUp when > 0. Typical values: 0.2 to 0.4.")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0, help="Enable CutMix when > 0. Typical values: 0.5 to 1.0.")
    parser.add_argument("--mix-prob", type=float, default=1.0, help="Probability of applying MixUp/CutMix to a training batch.")
    parser.add_argument("--mix-switch-prob", type=float, default=0.5, help="When MixUp and CutMix are both enabled, probability of selecting CutMix.")
    parser.add_argument("--feature-attention", choices=["none", "se", "cbam"], default="none", help="Extra attention attached on the final feature map. EfficientNet-B0 already includes internal SE blocks.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to 4 on CUDA and 0 on CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic backend settings. This is slower on GPU.")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision on CUDA.")
    parser.set_defaults(amp=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.data_dir = args.data_dir.resolve()
    device = resolve_device(args.device)
    if args.output_dir is None:
        if device.type == "cuda":
            default_output_name = "ablation_pretrained_ce" if args.pretrained else "ablation_scratch_ce_cuda"
        else:
            default_output_name = "ablation_scratch_ce"
        args.output_dir = repo_root / "outputs" / default_output_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mixup_alpha < 0.0 or args.cutmix_alpha < 0.0:
        raise ValueError("--mixup-alpha and --cutmix-alpha must be >= 0.")
    if not 0.0 <= args.mix_prob <= 1.0:
        raise ValueError("--mix-prob must be in [0, 1].")
    if not 0.0 <= args.mix_switch_prob <= 1.0:
        raise ValueError("--mix-switch-prob must be in [0, 1].")
    seed_everything(args.seed, deterministic=args.deterministic)
    args.num_workers = resolve_num_workers(args.num_workers, device)
    amp_enabled = resolve_amp_enabled(args.amp, device)
    pin_memory = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(
        f"Using device: {device}"
        + (f" ({gpu_name})" if gpu_name else "")
        + f", amp={'on' if amp_enabled else 'off'}, num_workers={args.num_workers}"
    )

    train_loader, valid_loader, test_loader, samples_by_split = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    weight_tensor, class_weights = resolve_class_weights(args, samples_by_split["train"], device)
    model = create_model(
        args.model_name,
        num_classes=len(CLASS_NAMES),
        pretrained=args.pretrained,
        feature_attention=args.feature_attention,
    ).to(device)
    criterion = create_criterion(args, weight_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    if class_weights is not None:
        formatted_weights = ", ".join(f"{name}={class_weights[name]:.4f}" for name in CLASS_NAMES)
        print(f"Using class weights: {formatted_weights}")
    if args.loss == "focal":
        print(f"Using focal loss: gamma={args.focal_gamma}")
    if has_mix_augmentation(args):
        print(
            "Using mixed augmentation: "
            f"mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}, "
            f"mix_prob={args.mix_prob}, cutmix_switch_prob={args.mix_switch_prob}"
        )
    if args.feature_attention != "none":
        print(
            f"Using extra feature attention: {args.feature_attention} "
            "(note: EfficientNet-B0 already contains internal SE blocks)."
        )

    history: List[Dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_path = args.output_dir / "best_model.pt"

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            args=args,
        )
        valid_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled=amp_enabled)
        step_scheduler(scheduler, args.scheduler, valid_metrics["loss"])
        current_lr = get_current_lr(optimizer)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "learning_rate": current_lr,
        }
        history.append(epoch_result)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"lr={current_lr:.7f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f}"
        )

        if valid_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = valid_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "valid_accuracy": valid_metrics["accuracy"],
                    "class_names": CLASS_NAMES,
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                    "feature_attention": args.feature_attention,
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_val_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled=amp_enabled)
    test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled=amp_enabled)
    elapsed_seconds = time.time() - start_time

    save_predictions(args.output_dir / "test_predictions.csv", test_metrics)
    save_confusion_matrix(args.output_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"])
    save_confusion_matrix(args.output_dir / "valid_confusion_matrix.csv", best_val_metrics["confusion_matrix"])

    summary = build_run_summary(
        args=args,
        train_history=history,
        best_epoch=best_epoch,
        best_val=best_val_metrics,
        test_result=test_metrics,
        samples_by_split=samples_by_split,
        elapsed_seconds=elapsed_seconds,
        device=device,
        amp_enabled=amp_enabled,
        pin_memory=pin_memory,
        gpu_name=gpu_name,
        class_weights=class_weights,
    )
    save_json(args.output_dir / "metrics_summary.json", summary)

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
