import argparse
import csv
import json
import os
import random
import shutil
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE_REL = Path(".cache") / "huggingface"
HF_CACHE_ROOT = PROJECT_ROOT / DEFAULT_HF_CACHE_REL
HF_HUB_CACHE_DIR = HF_CACHE_ROOT / "hub"
HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
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
DEFAULT_EXPERT_CLASSES = [
    "adenocarcinoma",
    "squamous.cell.carcinoma",
]
DEFAULT_MODEL_NAME = "efficientnet_b4"
MODEL_DEFAULT_IMAGE_SIZE = {
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 256,
    "efficientnet_b3": 288,
    "efficientnet_b4": 320,
}
LOCAL_PRETRAINED_FILES = {
    "efficientnet_b1": {
        "format": "safetensors",
        "url": "https://huggingface.co/timm/efficientnet_b1.ra4_e3600_r240_in1k/resolve/main/model.safetensors",
        "relative_path": Path(".cache") / "weights" / "efficientnet_b1.ra4_e3600_r240_in1k" / "model.safetensors",
    },
    "efficientnet_b2": {
        "format": "torch",
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth",
        "relative_path": Path(".cache") / "weights" / "efficientnet_b2.ra_in1k" / "model.pth",
    },
    "efficientnet_b3": {
        "format": "safetensors",
        "url": "https://huggingface.co/timm/efficientnet_b3.ra2_in1k/resolve/main/model.safetensors",
        "relative_path": Path(".cache") / "weights" / "efficientnet_b3.ra2_in1k" / "model.safetensors",
    },
    "efficientnet_b4": {
        "format": "safetensors",
        "url": "https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/model.safetensors",
        "relative_path": Path(".cache") / "weights" / "efficientnet_b4.ra2_in1k" / "model.safetensors",
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_class_names_arg(value: str) -> List[str]:
    class_names = [item.strip() for item in value.split(",") if item.strip()]
    if not class_names:
        raise ValueError("At least one class name must be provided.")

    unknown = [name for name in class_names if name not in CLASS_TO_INDEX]
    if unknown:
        raise ValueError(f"Unknown class names: {unknown}. Valid options: {CLASS_NAMES}")

    if len(set(class_names)) != len(class_names):
        raise ValueError(f"Duplicate class names are not allowed: {class_names}")
    return class_names


def sanitize_name(text: str) -> str:
    return text.replace(".", "_").replace(",", "_").replace("/", "_").replace("\\", "_")


def build_output_slug(class_names: Sequence[str]) -> str:
    return "_".join(sanitize_name(name) for name in class_names)


def build_backbone_tag(model_name: str) -> str:
    if model_name == "efficientnet_b0":
        return "b0"
    if model_name == "efficientnet_b1":
        return "b1"
    if model_name == "efficientnet_b2":
        return "b2"
    if model_name == "efficientnet_b3":
        return "b3"
    if model_name == "efficientnet_b4":
        return "b4"
    return sanitize_name(model_name)


def build_backbone_suffix(model_name: str) -> str:
    if model_name == "efficientnet_b0":
        return ""
    return f"_{build_backbone_tag(model_name)}"


def resolve_image_size(model_name: str, requested_image_size: Optional[int]) -> int:
    if requested_image_size is not None:
        return requested_image_size
    return MODEL_DEFAULT_IMAGE_SIZE.get(model_name, 224)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, timeout_seconds: int = 120, max_retries: int = 3) -> Path:
    ensure_parent_dir(destination)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            bytes_written = 0
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response, temp_path.open("wb") as output_file:
                expected_length_header = response.headers.get("Content-Length")
                expected_length = int(expected_length_header) if expected_length_header and expected_length_header.isdigit() else None
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    bytes_written += len(chunk)
            if expected_length is not None and bytes_written != expected_length:
                raise RuntimeError(
                    f"Downloaded file size mismatch for {url}: expected {expected_length} bytes, got {bytes_written} bytes."
                )
            temp_path.replace(destination)
            return destination
        except Exception as error:
            last_error = error
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt == max_retries:
                break
            time.sleep(min(attempt * 2, 5))

    raise RuntimeError(f"Failed to download pretrained weights from {url}: {last_error}") from last_error


def ensure_local_pretrained_file(model_name: str, force_redownload: bool = False) -> Optional[Tuple[Path, str]]:
    spec = LOCAL_PRETRAINED_FILES.get(model_name)
    if spec is None:
        return None

    local_path = PROJECT_ROOT / spec["relative_path"]
    if local_path.exists() and not force_redownload:
        return local_path, str(spec["format"])

    if force_redownload and local_path.exists():
        local_path.unlink(missing_ok=True)

    print(f"Downloading local pretrained weights for {model_name} -> {local_path}")
    return download_file(spec["url"], local_path), str(spec["format"])


def load_pretrained_state_dict(local_weights: Path, weight_format: str) -> Dict[str, Any]:
    state_dict: Any
    if weight_format == "safetensors":
        state_dict = load_safetensors_file(str(local_weights))
    elif weight_format == "torch":
        state_dict = torch.load(local_weights, map_location="cpu")
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[len("module.") :] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
    else:
        raise ValueError(f"Unsupported local pretrained weight format: {weight_format}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported pretrained state dict type: {type(state_dict)!r}")
    return state_dict


def build_pretrained_backbone(model_name: str, num_classes: int) -> nn.Module:
    local_pretrained = ensure_local_pretrained_file(model_name)
    if local_pretrained is None:
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    last_error: Optional[Exception] = None
    for attempt in range(2):
        local_weights, weight_format = local_pretrained
        backbone = timm.create_model(model_name, pretrained=False, num_classes=1000)
        try:
            state_dict = load_pretrained_state_dict(local_weights, weight_format)
            backbone.load_state_dict(state_dict, strict=True)
            if hasattr(backbone, "reset_classifier"):
                backbone.reset_classifier(num_classes)
            else:
                raise ValueError(
                    f"Backbone {model_name} does not support reset_classifier after local weight loading."
                )
            return backbone
        except Exception as error:
            last_error = error
            if attempt == 0:
                print(f"Detected invalid local pretrained file for {model_name}, deleting and re-downloading: {local_weights}")
                local_pretrained = ensure_local_pretrained_file(model_name, force_redownload=True)
                continue
            break

    raise RuntimeError(
        f"Failed to load local pretrained weights for {model_name} after re-download: {last_error}"
    ) from last_error


def build_output_subdirs(output_dir: Path) -> Tuple[Path, Path]:
    output_dir = output_dir.resolve()
    if output_dir.name in {"outputs", "weights", "results"}:
        raise ValueError(
            "--output-dir must include an experiment name, for example outputs/v2.0_pretrained_ce_b4."
        )

    if output_dir.parent.name in {"weights", "results"}:
        outputs_root = output_dir.parent.parent
        experiment_name = output_dir.name
    else:
        outputs_root = output_dir.parent
        experiment_name = output_dir.name

    weights_dir = outputs_root / "weights" / experiment_name
    results_dir = outputs_root / "results" / experiment_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir, results_dir


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
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

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


def collect_samples_by_split(data_dir: Path) -> Dict[str, List[Sample]]:
    return {
        "train": collect_samples(data_dir / "train", "train"),
        "valid": collect_samples(data_dir / "valid", "valid"),
        "test": collect_samples(data_dir / "test", "test"),
    }


def filter_samples_by_classes(samples: Sequence[Sample], class_names: Sequence[str]) -> List[Sample]:
    class_name_set = set(class_names)
    return [sample for sample in samples if sample.class_name in class_name_set]


class LungCancerDataset(Dataset):
    def __init__(self, samples: List[Sample], transform: transforms.Compose, class_names: Sequence[str]):
        self.samples = samples
        self.transform = transform
        self.class_names = list(class_names)
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}

        unknown = sorted({sample.class_name for sample in samples if sample.class_name not in self.class_to_index})
        if unknown:
            raise ValueError(f"Dataset received samples outside class space {self.class_names}: {unknown}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, self.class_to_index[sample.class_name], str(sample.image_path)


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
    class_names: Sequence[str],
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[Sample]]]:
    train_transform, eval_transform = build_transforms(image_size)
    all_samples_by_split = collect_samples_by_split(data_dir)
    samples_by_split = {
        split_name: filter_samples_by_classes(split_samples, class_names)
        for split_name, split_samples in all_samples_by_split.items()
    }

    datasets = {
        "train": LungCancerDataset(samples_by_split["train"], train_transform, class_names),
        "valid": LungCancerDataset(samples_by_split["valid"], eval_transform, class_names),
        "test": LungCancerDataset(samples_by_split["test"], eval_transform, class_names),
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
    if pretrained:
        backbone = build_pretrained_backbone(model_name, num_classes=num_classes)
    else:
        backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if feature_attention == "none":
        return backbone
    return AttentionAugmentedModel(backbone, feature_attention=feature_attention)


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
    args: argparse.Namespace,
    train_samples: Sequence[Sample],
    device: torch.device,
    class_names: Sequence[str],
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
    if args.class_weighting == "none":
        return None, None

    if args.class_weighting == "balanced":
        class_weights = compute_train_class_weights(train_samples, class_names)
    elif args.class_weighting == "manual":
        if not args.class_weights:
            raise ValueError("--class-weights is required when --class-weighting manual is used.")
        class_weights = parse_manual_class_weights(args.class_weights, class_names)
    else:
        raise ValueError(f"Unsupported class weighting: {args.class_weighting}")

    weight_tensor = torch.tensor(
        [class_weights[class_name] for class_name in class_names],
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
    num_classes: int,
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
        num_classes=num_classes,
        label_smoothing=args.label_smoothing,
        dtype=images.dtype,
    )
    labels_b = build_target_distribution(
        labels[permutation],
        num_classes=num_classes,
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
    num_classes: int,
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
        images, targets = apply_mix_augmentation(images, labels, num_classes, args)
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


def build_prediction_report(
    labels: Sequence[int],
    preds: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    return {
        "accuracy": accuracy_score(labels, preds),
        "report": report,
        "confusion_matrix": matrix,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    class_names: Sequence[str],
) -> Dict[str, Any]:
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

    metrics = build_prediction_report(all_labels, all_preds, class_names)
    metrics.update(
        {
            "loss": total_loss / max(len(loader.dataset), 1),
            "labels": all_labels,
            "preds": all_preds,
            "paths": all_paths,
            "probs": all_probs,
        }
    )
    return metrics


def save_predictions(output_path: Path, eval_result: Dict[str, Any], class_names: Sequence[str]) -> None:
    ensure_parent_dir(output_path)
    header = ["image_path", "true_label", "pred_label"] + [f"prob_{name}" for name in class_names]
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
                    class_names[true_label],
                    class_names[pred_label],
                    *[f"{prob:.6f}" for prob in probs],
                ]
            )


def save_confusion_matrix(output_path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def save_json(output_path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_class_distribution(
    samples_by_split: Dict[str, Sequence[Sample]],
    class_names: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    class_distribution: Dict[str, Dict[str, int]] = {}
    class_name_set = set(class_names)
    for split_name, samples in samples_by_split.items():
        distribution = {class_name: 0 for class_name in class_names}
        for sample in samples:
            if sample.class_name in class_name_set:
                distribution[sample.class_name] += 1
        class_distribution[split_name] = distribution
    return class_distribution


def build_training_run_summary(
    args: argparse.Namespace,
    train_history: List[Dict[str, float]],
    best_epoch: int,
    best_val: Dict[str, Any],
    test_result: Dict[str, Any],
    samples_by_split: Dict[str, List[Sample]],
    elapsed_seconds: float,
    device: torch.device,
    amp_enabled: bool,
    pin_memory: bool,
    gpu_name: Optional[str],
    class_weights: Optional[Dict[str, float]],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    return {
        "mode": args.run_mode,
        "config": {
            "run_mode": args.run_mode,
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
            "active_class_names": list(class_names),
            "expert_classes_arg": args.expert_classes,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
        },
        "dataset": {
            "data_dir": str(args.data_dir),
            "class_names": list(class_names),
            "split_sizes": {split_name: len(samples) for split_name, samples in samples_by_split.items()},
            "class_distribution": build_class_distribution(samples_by_split, class_names),
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


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    epoch: int,
    valid_accuracy: float,
    args: argparse.Namespace,
    class_names: Sequence[str],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "valid_accuracy": valid_accuracy,
            "class_names": list(class_names),
            "model_name": args.model_name,
            "image_size": args.image_size,
            "feature_attention": args.feature_attention,
            "run_mode": args.run_mode,
            "pretrained": args.pretrained,
        },
        output_path,
    )


def resolve_default_data_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "附件" / "Data",
        repo_root / "附件" / "Data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def configure_huggingface_cache(repo_root: Path) -> Path:
    cache_root = repo_root / DEFAULT_HF_CACHE_REL
    hub_cache = cache_root / "hub"
    cache_root.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache_root


def resolve_active_class_names(args: argparse.Namespace) -> List[str]:
    if args.run_mode == "expert":
        return parse_class_names_arg(args.expert_classes)
    return list(CLASS_NAMES)


def resolve_training_output_dir(
    args: argparse.Namespace,
    repo_root: Path,
    device: torch.device,
    class_names: Sequence[str],
) -> Path:
    if args.output_dir is not None:
        return args.output_dir

    backbone_suffix = build_backbone_suffix(args.model_name)

    if args.run_mode == "expert":
        return repo_root / "outputs" / f"expert_{build_output_slug(class_names)}{backbone_suffix}"

    if device.type == "cuda":
        default_output_name = "v2.0_pretrained_ce" if args.pretrained else "v1.1_scratch_ce_cuda"
    else:
        default_output_name = "v1.0_scratch_ce_cpu"
    return repo_root / "outputs" / f"{default_output_name}{backbone_suffix}"


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(f"Checkpoint is missing class_names metadata: {checkpoint_path}")

    model_name = checkpoint.get("model_name", "efficientnet_b0")
    feature_attention = checkpoint.get("feature_attention", "none")
    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
        feature_attention=feature_attention,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_probabilities(
    model: nn.Module,
    image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
    amp_enabled: bool,
) -> List[float]:
    tensor = transform(image).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    with autocast_context(device, amp_enabled):
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).detach().cpu().tolist()


def should_invoke_expert(
    main_probs: Sequence[float],
    main_class_names: Sequence[str],
    expert_class_names: Sequence[str],
    top_k: int,
    margin_threshold: float,
) -> Tuple[bool, str, float]:
    if top_k < 2:
        raise ValueError("--expert-trigger-topk must be at least 2.")
    if top_k > len(main_class_names):
        raise ValueError("--expert-trigger-topk cannot exceed the number of main classes.")

    ranked_indices = sorted(range(len(main_probs)), key=lambda idx: main_probs[idx], reverse=True)
    top_indices = ranked_indices[:top_k]
    top_names = [main_class_names[idx] for idx in top_indices]
    if not set(top_names).issubset(set(expert_class_names)):
        return False, top_names[1], float(main_probs[top_indices[0]] - main_probs[top_indices[1]])

    top_margin = float(main_probs[top_indices[0]] - main_probs[top_indices[1]])
    if margin_threshold >= 0.0 and top_margin > margin_threshold:
        return False, top_names[1], top_margin
    return True, top_names[1], top_margin


def merge_main_and_expert_probabilities(
    main_probs: Sequence[float],
    main_class_names: Sequence[str],
    expert_probs: Sequence[float],
    expert_class_names: Sequence[str],
) -> List[float]:
    final_probs = list(main_probs)
    expert_indices = [main_class_names.index(class_name) for class_name in expert_class_names]
    expert_mass = sum(main_probs[index] for index in expert_indices)
    if expert_mass <= 0.0:
        expert_mass = 1.0

    for local_index, class_name in enumerate(expert_class_names):
        global_index = main_class_names.index(class_name)
        final_probs[global_index] = expert_probs[local_index] * expert_mass

    total_prob = sum(final_probs)
    if total_prob > 0.0:
        final_probs = [prob / total_prob for prob in final_probs]
    return final_probs


def evaluate_cascade(
    main_model: nn.Module,
    expert_model: nn.Module,
    samples: Sequence[Sample],
    main_transform: transforms.Compose,
    expert_transform: transforms.Compose,
    main_class_names: Sequence[str],
    expert_class_names: Sequence[str],
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    labels: List[int] = []
    preds: List[int] = []
    paths: List[str] = []
    probs: List[List[float]] = []
    records: List[Dict[str, Any]] = []
    expert_invocations = 0
    expert_changed_predictions = 0
    expert_corrected_predictions = 0
    expert_hurt_predictions = 0

    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        main_probs = predict_probabilities(main_model, image, main_transform, device, amp_enabled)
        main_pred = int(np.argmax(main_probs))
        invoke_expert, main_runner_up_name, margin = should_invoke_expert(
            main_probs=main_probs,
            main_class_names=main_class_names,
            expert_class_names=expert_class_names,
            top_k=args.expert_trigger_topk,
            margin_threshold=args.expert_margin_threshold,
        )

        expert_pred_name = ""
        final_probs = list(main_probs)
        if invoke_expert:
            expert_invocations += 1
            expert_probs = predict_probabilities(expert_model, image, expert_transform, device, amp_enabled)
            final_probs = merge_main_and_expert_probabilities(
                main_probs=main_probs,
                main_class_names=main_class_names,
                expert_probs=expert_probs,
                expert_class_names=expert_class_names,
            )
            expert_pred_name = expert_class_names[int(np.argmax(expert_probs))]

        final_pred = int(np.argmax(final_probs))
        true_label = sample.label
        if invoke_expert and final_pred != main_pred:
            expert_changed_predictions += 1
            if main_pred != true_label and final_pred == true_label:
                expert_corrected_predictions += 1
            elif main_pred == true_label and final_pred != true_label:
                expert_hurt_predictions += 1

        labels.append(true_label)
        preds.append(final_pred)
        paths.append(str(sample.image_path))
        probs.append(final_probs)
        records.append(
            {
                "image_path": str(sample.image_path),
                "true_label": main_class_names[true_label],
                "main_pred_label": main_class_names[main_pred],
                "final_pred_label": main_class_names[final_pred],
                "expert_invoked": invoke_expert,
                "expert_pred_label": expert_pred_name,
                "main_runner_up_label": main_runner_up_name,
                "main_margin": margin,
                "final_probs": final_probs,
            }
        )

    metrics = build_prediction_report(labels, preds, main_class_names)
    metrics.update(
        {
            "loss": None,
            "labels": labels,
            "preds": preds,
            "paths": paths,
            "probs": probs,
            "records": records,
            "cascade_stats": {
                "expert_invocations": expert_invocations,
                "expert_changed_predictions": expert_changed_predictions,
                "expert_corrected_predictions": expert_corrected_predictions,
                "expert_hurt_predictions": expert_hurt_predictions,
            },
        }
    )
    return metrics


def save_cascade_predictions(output_path: Path, cascade_result: Dict[str, Any], class_names: Sequence[str]) -> None:
    ensure_parent_dir(output_path)
    header = [
        "image_path",
        "true_label",
        "main_pred_label",
        "final_pred_label",
        "expert_invoked",
        "expert_pred_label",
        "main_runner_up_label",
        "main_margin",
    ] + [f"final_prob_{name}" for name in class_names]

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for record in cascade_result["records"]:
            writer.writerow(
                [
                    record["image_path"],
                    record["true_label"],
                    record["main_pred_label"],
                    record["final_pred_label"],
                    record["expert_invoked"],
                    record["expert_pred_label"],
                    record["main_runner_up_label"],
                    f"{record['main_margin']:.6f}",
                    *[f"{prob:.6f}" for prob in record["final_probs"]],
                ]
            )


def build_cascade_run_summary(
    args: argparse.Namespace,
    valid_result: Dict[str, Any],
    test_result: Dict[str, Any],
    samples_by_split: Dict[str, List[Sample]],
    elapsed_seconds: float,
    device: torch.device,
    amp_enabled: bool,
    gpu_name: Optional[str],
    main_checkpoint_path: Path,
    expert_checkpoint_path: Path,
    main_checkpoint: Dict[str, Any],
    expert_checkpoint: Dict[str, Any],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    return {
        "mode": "cascade",
        "config": {
            "run_mode": "cascade",
            "data_dir": str(args.data_dir),
            "device_request": args.device,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "expert_trigger_topk": args.expert_trigger_topk,
            "expert_margin_threshold": args.expert_margin_threshold,
            "main_checkpoint": str(main_checkpoint_path),
            "expert_checkpoint": str(expert_checkpoint_path),
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
        },
        "dataset": {
            "class_names": list(class_names),
            "split_sizes": {split_name: len(samples) for split_name, samples in samples_by_split.items()},
            "class_distribution": build_class_distribution(samples_by_split, class_names),
        },
        "main_model": {
            "model_name": main_checkpoint.get("model_name"),
            "image_size": main_checkpoint.get("image_size"),
            "feature_attention": main_checkpoint.get("feature_attention"),
            "class_names": main_checkpoint.get("class_names"),
            "best_epoch": main_checkpoint.get("epoch"),
            "best_valid_accuracy": main_checkpoint.get("valid_accuracy"),
        },
        "expert_model": {
            "model_name": expert_checkpoint.get("model_name"),
            "image_size": expert_checkpoint.get("image_size"),
            "feature_attention": expert_checkpoint.get("feature_attention"),
            "class_names": expert_checkpoint.get("class_names"),
            "best_epoch": expert_checkpoint.get("epoch"),
            "best_valid_accuracy": expert_checkpoint.get("valid_accuracy"),
        },
        "validation": {
            "accuracy": valid_result["accuracy"],
            "report": valid_result["report"],
            "cascade_stats": valid_result["cascade_stats"],
        },
        "test": {
            "accuracy": test_result["accuracy"],
            "report": test_result["report"],
            "cascade_stats": test_result["cascade_stats"],
        },
        "elapsed_seconds": elapsed_seconds,
    }


def validate_training_args(args: argparse.Namespace) -> None:
    if args.mixup_alpha < 0.0 or args.cutmix_alpha < 0.0:
        raise ValueError("--mixup-alpha and --cutmix-alpha must be >= 0.")
    if not 0.0 <= args.mix_prob <= 1.0:
        raise ValueError("--mix-prob must be in [0, 1].")
    if not 0.0 <= args.mix_switch_prob <= 1.0:
        raise ValueError("--mix-switch-prob must be in [0, 1].")


def train_and_evaluate(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    configure_huggingface_cache(repo_root)
    args.data_dir = args.data_dir.resolve()
    device = resolve_device(args.device)
    args.image_size = resolve_image_size(args.model_name, args.image_size)
    class_names = resolve_active_class_names(args)
    args.output_dir = resolve_training_output_dir(args, repo_root, device, class_names)
    weights_dir, results_dir = build_output_subdirs(args.output_dir)
    validate_training_args(args)

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
    if args.run_mode == "expert":
        print(f"Training expert branch for classes: {class_names}")

    train_loader, valid_loader, test_loader, samples_by_split = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        class_names=class_names,
    )

    weight_tensor, class_weights = resolve_class_weights(args, samples_by_split["train"], device, class_names)
    model = create_model(
        args.model_name,
        num_classes=len(class_names),
        pretrained=args.pretrained,
        feature_attention=args.feature_attention,
    ).to(device)
    criterion = create_criterion(args, weight_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    if class_weights is not None:
        formatted_weights = ", ".join(f"{name}={class_weights[name]:.4f}" for name in class_names)
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
            "(note: EfficientNet backbones already contain internal SE blocks)."
        )

    history: List[Dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_path = weights_dir / "best_model.pt"

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
            num_classes=len(class_names),
        )
        valid_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled=amp_enabled, class_names=class_names)
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
            save_checkpoint(best_model_path, model, epoch, valid_metrics["accuracy"], args, class_names)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_val_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled=amp_enabled, class_names=class_names)
    test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled=amp_enabled, class_names=class_names)
    elapsed_seconds = time.time() - start_time

    save_predictions(results_dir / "test_predictions.csv", test_metrics, class_names)
    save_confusion_matrix(results_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"], class_names)
    save_confusion_matrix(results_dir / "valid_confusion_matrix.csv", best_val_metrics["confusion_matrix"], class_names)

    summary = build_training_run_summary(
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
        class_names=class_names,
    )
    save_json(results_dir / "metrics_summary.json", summary)

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Weights saved to: {weights_dir}")
    print(f"Results saved to: {results_dir}")


def run_cascade_evaluation(args: argparse.Namespace) -> None:
    if args.main_checkpoint is None or args.expert_checkpoint is None:
        raise ValueError("--main-checkpoint and --expert-checkpoint are required when --run-mode cascade is used.")

    repo_root = Path(__file__).resolve().parents[1]
    configure_huggingface_cache(repo_root)
    args.data_dir = args.data_dir.resolve()
    device = resolve_device(args.device)
    amp_enabled = resolve_amp_enabled(args.amp, device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    main_model, main_checkpoint = load_model_from_checkpoint(args.main_checkpoint, device)
    expert_model, expert_checkpoint = load_model_from_checkpoint(args.expert_checkpoint, device)
    main_class_names = list(main_checkpoint["class_names"])
    expert_class_names = list(expert_checkpoint["class_names"])

    if main_class_names != CLASS_NAMES:
        raise ValueError(
            "Cascade mode expects the main checkpoint to be trained on the full four-class space. "
            f"Got {main_class_names}."
        )
    if len(expert_class_names) < 2:
        raise ValueError(f"Expert checkpoint must contain at least two classes, got {expert_class_names}.")
    if not set(expert_class_names).issubset(set(main_class_names)):
        raise ValueError(
            "Expert checkpoint classes must be a subset of the main checkpoint classes. "
            f"Main: {main_class_names}, expert: {expert_class_names}"
        )

    if args.output_dir is None:
        expert_checkpoint_dir = args.expert_checkpoint.resolve().parent
        expert_parent_name = expert_checkpoint_dir.name
        if expert_parent_name == "weights":
            expert_parent_name = expert_checkpoint_dir.parent.name
        if expert_parent_name.startswith("expert_"):
            output_name = f"cascade_{expert_parent_name[len('expert_'):]}"
        else:
            output_name = f"cascade_{build_output_slug(expert_class_names)}"
        args.output_dir = repo_root / "outputs" / output_name
    _, results_dir = build_output_subdirs(args.output_dir)

    print(
        f"Using device: {device}"
        + (f" ({gpu_name})" if gpu_name else "")
        + f", amp={'on' if amp_enabled else 'off'}"
    )
    print(f"Main checkpoint: {args.main_checkpoint}")
    print(f"Expert checkpoint: {args.expert_checkpoint}")
    print(
        "Cascade trigger: "
        f"top-{args.expert_trigger_topk} classes must stay inside {expert_class_names}"
        + (
            f", margin <= {args.expert_margin_threshold:.4f}"
            if args.expert_margin_threshold >= 0.0
            else ", margin filter disabled"
        )
    )

    all_samples_by_split = collect_samples_by_split(args.data_dir)
    main_eval_transform = build_transforms(int(main_checkpoint.get("image_size", args.image_size)))[1]
    expert_eval_transform = build_transforms(int(expert_checkpoint.get("image_size", args.image_size)))[1]

    start_time = time.time()
    valid_result = evaluate_cascade(
        main_model=main_model,
        expert_model=expert_model,
        samples=all_samples_by_split["valid"],
        main_transform=main_eval_transform,
        expert_transform=expert_eval_transform,
        main_class_names=main_class_names,
        expert_class_names=expert_class_names,
        device=device,
        amp_enabled=amp_enabled,
        args=args,
    )
    test_result = evaluate_cascade(
        main_model=main_model,
        expert_model=expert_model,
        samples=all_samples_by_split["test"],
        main_transform=main_eval_transform,
        expert_transform=expert_eval_transform,
        main_class_names=main_class_names,
        expert_class_names=expert_class_names,
        device=device,
        amp_enabled=amp_enabled,
        args=args,
    )
    elapsed_seconds = time.time() - start_time

    save_cascade_predictions(results_dir / "valid_cascade_predictions.csv", valid_result, main_class_names)
    save_cascade_predictions(results_dir / "test_cascade_predictions.csv", test_result, main_class_names)
    save_confusion_matrix(results_dir / "valid_confusion_matrix.csv", valid_result["confusion_matrix"], main_class_names)
    save_confusion_matrix(results_dir / "test_confusion_matrix.csv", test_result["confusion_matrix"], main_class_names)

    summary = build_cascade_run_summary(
        args=args,
        valid_result=valid_result,
        test_result=test_result,
        samples_by_split=all_samples_by_split,
        elapsed_seconds=elapsed_seconds,
        device=device,
        amp_enabled=amp_enabled,
        gpu_name=gpu_name,
        main_checkpoint_path=args.main_checkpoint,
        expert_checkpoint_path=args.expert_checkpoint,
        main_checkpoint=main_checkpoint,
        expert_checkpoint=expert_checkpoint,
        class_names=main_class_names,
    )
    save_json(results_dir / "metrics_summary.json", summary)

    print()
    print(f"Cascade validation accuracy: {valid_result['accuracy']:.4f}")
    print(f"Cascade test accuracy: {test_result['accuracy']:.4f}")
    print(f"Validation cascade stats: {valid_result['cascade_stats']}")
    print(f"Test cascade stats: {test_result['cascade_stats']}")
    print(f"Results saved to: {results_dir}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_dir = resolve_default_data_dir(repo_root)

    parser = argparse.ArgumentParser(
        description=(
            "Problem 2 pipeline: single four-class baseline, expert-branch subset training, "
            "or cascade evaluation that combines a main model with an expert branch."
        )
    )
    parser.add_argument("--run-mode", choices=["single", "expert", "cascade"], default="single")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Experiment slug path. For example outputs/v2.0_pretrained_ce_b4. "
            "Checkpoints are written to outputs/weights/<experiment>/ and reports to outputs/results/<experiment>/."
        ),
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", action="store_true", help="Use timm pretrained weights if available.")
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
        help="Training loss; label smoothing applies to both options.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size. Defaults to 320 for EfficientNet-B4, 288 for EfficientNet-B3, 256 for EfficientNet-B2, 240 for EfficientNet-B1, and 224 for EfficientNet-B0.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor used by cross-entropy and focal loss.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter when --loss focal is selected.",
    )
    parser.add_argument("--class-weighting", choices=["none", "balanced", "manual"], default="none")
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Comma-separated weights in active class order when class-weighting=manual.",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Enable MixUp when > 0. Typical values: 0.2 to 0.4.")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0, help="Enable CutMix when > 0. Typical values: 0.5 to 1.0.")
    parser.add_argument("--mix-prob", type=float, default=1.0, help="Probability of applying MixUp/CutMix to a training batch.")
    parser.add_argument(
        "--mix-switch-prob",
        type=float,
        default=0.5,
        help="When MixUp and CutMix are both enabled, probability of selecting CutMix.",
    )
    parser.add_argument(
        "--feature-attention",
        choices=["none", "se", "cbam"],
        default="none",
        help="Extra attention attached on the final feature map. EfficientNet backbones already include internal SE blocks.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to 4 on CUDA and 0 on CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic backend settings. This is slower on GPU.")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision on CUDA.")
    parser.add_argument(
        "--expert-classes",
        type=str,
        default=",".join(DEFAULT_EXPERT_CLASSES),
        help=(
            "Comma-separated class list for expert-branch training. "
            "Default: adenocarcinoma,squamous.cell.carcinoma"
        ),
    )
    parser.add_argument("--main-checkpoint", type=Path, default=None, help="Main four-class checkpoint used in cascade mode.")
    parser.add_argument("--expert-checkpoint", type=Path, default=None, help="Expert-branch checkpoint used in cascade mode.")
    parser.add_argument(
        "--expert-trigger-topk",
        type=int,
        default=2,
        help="Invoke the expert branch only when the main model's top-k classes all stay inside the expert class subset.",
    )
    parser.add_argument(
        "--expert-margin-threshold",
        type=float,
        default=0.12,
        help="Maximum main-model top1-top2 probability margin for invoking the expert branch. Set a negative value to disable the margin filter.",
    )
    parser.set_defaults(amp=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_mode == "cascade":
        run_cascade_evaluation(args)
        return
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
