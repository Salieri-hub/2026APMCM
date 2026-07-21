from pathlib import Path
from typing import List, Optional, Sequence


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
