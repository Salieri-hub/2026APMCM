from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from .constants import CLASS_NAMES, parse_class_names_arg, resolve_image_size
from .runtime import resolve_amp_enabled, resolve_device, resolve_num_workers, resolve_training_output_dir


@dataclass(frozen=True, slots=True)
class TrainConfig:
    run_mode: str
    data_dir: Path
    output_dir: Path
    model_name: str
    pretrained: bool
    loss: str
    epochs: int
    batch_size: int
    image_size: int
    lr: float
    weight_decay: float
    label_smoothing: float
    focal_gamma: float
    class_weighting: str
    class_weights_arg: Optional[str]
    scheduler: str
    min_lr: float
    plateau_patience: int
    plateau_factor: float
    mixup_alpha: float
    cutmix_alpha: float
    mix_prob: float
    mix_switch_prob: float
    feature_attention: str
    num_workers: int
    seed: int
    device_request: str
    device: torch.device
    deterministic: bool
    amp_enabled: bool
    expert_classes_arg: str
    class_names: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    run_mode: str
    data_dir: Path
    output_dir: Optional[Path]
    device_request: str
    device: torch.device
    amp_enabled: bool
    main_checkpoint: Path
    expert_checkpoint: Path
    expert_trigger_topk: int
    expert_margin_threshold: float
    image_size: int


def resolve_train_class_names(run_mode: str, expert_classes_arg: str) -> Tuple[str, ...]:
    if run_mode == "expert":
        return tuple(parse_class_names_arg(expert_classes_arg))
    return tuple(CLASS_NAMES)


def build_train_config(args: argparse.Namespace, repo_root: Path) -> TrainConfig:
    if args.run_mode == "cascade":
        raise ValueError("build_train_config does not support --run-mode cascade.")

    device = resolve_device(args.device)
    class_names = resolve_train_class_names(args.run_mode, args.expert_classes)
    image_size = resolve_image_size(args.model_name, args.image_size)
    output_dir = resolve_training_output_dir(
        output_dir=args.output_dir,
        repo_root=repo_root,
        device=device,
        run_mode=args.run_mode,
        model_name=args.model_name,
        pretrained=args.pretrained,
        class_names=class_names,
    )

    return TrainConfig(
        run_mode=args.run_mode,
        data_dir=args.data_dir.resolve(),
        output_dir=output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        class_weighting=args.class_weighting,
        class_weights_arg=args.class_weights,
        scheduler=args.scheduler,
        min_lr=args.min_lr,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        mix_switch_prob=args.mix_switch_prob,
        feature_attention=args.feature_attention,
        num_workers=resolve_num_workers(args.num_workers, device),
        seed=args.seed,
        device_request=args.device,
        device=device,
        deterministic=args.deterministic,
        amp_enabled=resolve_amp_enabled(args.amp, device),
        expert_classes_arg=args.expert_classes,
        class_names=class_names,
    )


def build_cascade_config(args: argparse.Namespace) -> CascadeConfig:
    if args.main_checkpoint is None or args.expert_checkpoint is None:
        raise ValueError("--main-checkpoint and --expert-checkpoint are required when --run-mode cascade is used.")

    device = resolve_device(args.device)
    return CascadeConfig(
        run_mode=args.run_mode,
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir,
        device_request=args.device,
        device=device,
        amp_enabled=resolve_amp_enabled(args.amp, device),
        main_checkpoint=args.main_checkpoint,
        expert_checkpoint=args.expert_checkpoint,
        expert_trigger_topk=args.expert_trigger_topk,
        expert_margin_threshold=args.expert_margin_threshold,
        image_size=resolve_image_size(args.model_name, args.image_size),
    )
