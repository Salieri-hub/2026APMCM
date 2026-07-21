from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import CLASS_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD


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
