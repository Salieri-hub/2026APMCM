import time
from typing import Any, Dict, List, Sequence

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import create_dataloaders
from .losses import (
    apply_mix_augmentation,
    create_criterion,
    create_scheduler,
    get_current_lr,
    has_mix_augmentation,
    resolve_class_weights,
    step_scheduler,
)
from .models import create_model
from .reporting import (
    build_prediction_report,
    build_training_run_summary,
    save_checkpoint,
    save_confusion_matrix,
    save_json,
    save_predictions,
)
from .runtime import (
    PROJECT_ROOT,
    autocast_context,
    build_output_subdirs,
    configure_huggingface_cache,
    seed_everything,
)


def validate_training_config(config: TrainConfig) -> None:
    if config.mixup_alpha < 0.0 or config.cutmix_alpha < 0.0:
        raise ValueError("--mixup-alpha and --cutmix-alpha must be >= 0.")
    if not 0.0 <= config.mix_prob <= 1.0:
        raise ValueError("--mix-prob must be in [0, 1].")
    if not 0.0 <= config.mix_switch_prob <= 1.0:
        raise ValueError("--mix-switch-prob must be in [0, 1].")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
    config: TrainConfig,
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
        images, targets = apply_mix_augmentation(images, labels, num_classes, config)
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


def train_and_evaluate(config: TrainConfig) -> None:
    configure_huggingface_cache(PROJECT_ROOT)
    device = config.device
    class_names = config.class_names
    weights_dir, results_dir = build_output_subdirs(config.output_dir)
    validate_training_config(config)

    seed_everything(config.seed, deterministic=config.deterministic)
    amp_enabled = config.amp_enabled
    pin_memory = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(
        f"Using device: {device}"
        + (f" ({gpu_name})" if gpu_name else "")
        + f", amp={'on' if amp_enabled else 'off'}, num_workers={config.num_workers}"
    )
    if config.run_mode == "expert":
        print(f"Training expert branch for classes: {class_names}")

    train_loader, valid_loader, test_loader, samples_by_split = create_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        class_names=class_names,
    )

    weight_tensor, class_weights = resolve_class_weights(config, samples_by_split["train"], device, class_names)
    model = create_model(
        config.model_name,
        num_classes=len(class_names),
        pretrained=config.pretrained,
        feature_attention=config.feature_attention,
    ).to(device)
    criterion = create_criterion(config, weight_tensor)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    if class_weights is not None:
        formatted_weights = ", ".join(f"{name}={class_weights[name]:.4f}" for name in class_names)
        print(f"Using class weights: {formatted_weights}")
    if config.loss == "focal":
        print(f"Using focal loss: gamma={config.focal_gamma}")
    if has_mix_augmentation(config):
        print(
            "Using mixed augmentation: "
            f"mixup_alpha={config.mixup_alpha}, cutmix_alpha={config.cutmix_alpha}, "
            f"mix_prob={config.mix_prob}, cutmix_switch_prob={config.mix_switch_prob}"
        )
    if config.feature_attention != "none":
        print(
            f"Using extra feature attention: {config.feature_attention} "
            "(note: EfficientNet backbones already contain internal SE blocks)."
        )

    history: List[Dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_path = weights_dir / "best_model.pt"

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            config=config,
            num_classes=len(class_names),
        )
        valid_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled=amp_enabled, class_names=class_names)
        step_scheduler(scheduler, config.scheduler, valid_metrics["loss"])
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
            f"Epoch [{epoch:02d}/{config.epochs}] "
            f"lr={current_lr:.7f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f}"
        )

        if valid_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = valid_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(best_model_path, model, epoch, valid_metrics["accuracy"], config, class_names)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_val_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled=amp_enabled, class_names=class_names)
    test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled=amp_enabled, class_names=class_names)
    elapsed_seconds = time.time() - start_time

    save_predictions(results_dir / "test_predictions.csv", test_metrics, class_names)
    save_confusion_matrix(results_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"], class_names)
    save_confusion_matrix(results_dir / "valid_confusion_matrix.csv", best_val_metrics["confusion_matrix"], class_names)

    summary = build_training_run_summary(
        config=config,
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
