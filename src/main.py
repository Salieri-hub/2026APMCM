import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        datasets["valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader, test_loader, samples_by_split


def create_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
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
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_paths: List[str] = []

    for images, labels, paths in loader:
        images = images.to(device)
        labels = labels.to(device)

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
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device": str(device),
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
    default_output_dir = repo_root / "outputs" / "problem2_baseline"

    parser = argparse.ArgumentParser(
        description="Problem 2 baseline: timm EfficientNet-B0 for four-class lung cancer image classification."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--model-name", type=str, default="efficientnet_b0")
    parser.add_argument("--pretrained", action="store_true", help="Use timm pretrained weights if available.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader, samples_by_split = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model(args.model_name, num_classes=len(CLASS_NAMES), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_path = args.output_dir / "best_model.pt"

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_metrics = evaluate(model, valid_loader, criterion, device)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(epoch_result)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
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
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_val_metrics = evaluate(model, valid_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
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
    )
    save_json(args.output_dir / "metrics_summary.json", summary)

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
