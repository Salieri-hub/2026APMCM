import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .runtime import PROJECT_ROOT, autocast_context, ensure_parent_dir

import timm
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from torch import nn
from torchvision import transforms

from .constants import LOCAL_PRETRAINED_FILES


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
