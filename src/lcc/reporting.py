import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn

from .config import CascadeConfig, TrainConfig
from .data import Sample
from .runtime import ensure_parent_dir


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
    config: TrainConfig,
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
        "mode": config.run_mode,
        "config": {
            "run_mode": config.run_mode,
            "model_name": config.model_name,
            "pretrained": config.pretrained,
            "loss": config.loss,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "focal_gamma": config.focal_gamma,
            "class_weighting": config.class_weighting,
            "class_weights": class_weights,
            "scheduler": config.scheduler,
            "min_lr": config.min_lr,
            "mixup_alpha": config.mixup_alpha,
            "cutmix_alpha": config.cutmix_alpha,
            "mix_prob": config.mix_prob,
            "mix_switch_prob": config.mix_switch_prob,
            "feature_attention": config.feature_attention,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "device_request": config.device_request,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "deterministic": config.deterministic,
            "pin_memory": pin_memory,
            "active_class_names": list(class_names),
            "expert_classes_arg": config.expert_classes_arg,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
        },
        "dataset": {
            "data_dir": str(config.data_dir),
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
    config: TrainConfig,
    class_names: Sequence[str],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "valid_accuracy": valid_accuracy,
            "class_names": list(class_names),
            "model_name": config.model_name,
            "image_size": config.image_size,
            "feature_attention": config.feature_attention,
            "run_mode": config.run_mode,
            "pretrained": config.pretrained,
        },
        output_path,
    )


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
    config: CascadeConfig,
    valid_result: Dict[str, Any],
    test_result: Dict[str, Any],
    samples_by_split: Dict[str, List[Sample]],
    elapsed_seconds: float,
    device: torch.device,
    amp_enabled: bool,
    gpu_name: Optional[str],
    main_checkpoint: Dict[str, Any],
    expert_checkpoint: Dict[str, Any],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    return {
        "mode": config.run_mode,
        "config": {
            "run_mode": config.run_mode,
            "data_dir": str(config.data_dir),
            "device_request": config.device_request,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "expert_trigger_topk": config.expert_trigger_topk,
            "expert_margin_threshold": config.expert_margin_threshold,
            "main_checkpoint": str(config.main_checkpoint),
            "expert_checkpoint": str(config.expert_checkpoint),
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
