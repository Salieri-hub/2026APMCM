import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config import CascadeConfig
from .constants import CLASS_NAMES, build_output_slug
from .data import Sample, build_transforms, collect_samples_by_split
from .models import load_model_from_checkpoint, predict_probabilities
from .reporting import (
    build_cascade_run_summary,
    build_prediction_report,
    save_cascade_predictions,
    save_confusion_matrix,
    save_json,
)
from .runtime import (
    PROJECT_ROOT,
    build_output_subdirs,
    configure_huggingface_cache,
)


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
    main_model: torch.nn.Module,
    expert_model: torch.nn.Module,
    samples: Sequence[Sample],
    main_transform,
    expert_transform,
    main_class_names: Sequence[str],
    expert_class_names: Sequence[str],
    device: torch.device,
    amp_enabled: bool,
    config: CascadeConfig,
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
            top_k=config.expert_trigger_topk,
            margin_threshold=config.expert_margin_threshold,
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


def run_cascade_evaluation(config: CascadeConfig) -> None:
    configure_huggingface_cache(PROJECT_ROOT)
    device = config.device
    amp_enabled = config.amp_enabled
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    main_model, main_checkpoint = load_model_from_checkpoint(config.main_checkpoint, device)
    expert_model, expert_checkpoint = load_model_from_checkpoint(config.expert_checkpoint, device)
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

    output_dir = config.output_dir
    if output_dir is None:
        expert_checkpoint_dir = config.expert_checkpoint.resolve().parent
        expert_parent_name = expert_checkpoint_dir.name
        if expert_parent_name == "weights":
            expert_parent_name = expert_checkpoint_dir.parent.name
        if expert_parent_name.startswith("expert_"):
            output_name = f"cascade_{expert_parent_name[len('expert_'):]}"
        else:
            output_name = f"cascade_{build_output_slug(expert_class_names)}"
        output_dir = PROJECT_ROOT / "outputs" / output_name
    _, results_dir = build_output_subdirs(output_dir)

    print(
        f"Using device: {device}"
        + (f" ({gpu_name})" if gpu_name else "")
        + f", amp={'on' if amp_enabled else 'off'}"
    )
    print(f"Main checkpoint: {config.main_checkpoint}")
    print(f"Expert checkpoint: {config.expert_checkpoint}")
    print(
        "Cascade trigger: "
        f"top-{config.expert_trigger_topk} classes must stay inside {expert_class_names}"
        + (
            f", margin <= {config.expert_margin_threshold:.4f}"
            if config.expert_margin_threshold >= 0.0
            else ", margin filter disabled"
        )
    )

    all_samples_by_split = collect_samples_by_split(config.data_dir)
    main_eval_transform = build_transforms(int(main_checkpoint.get("image_size", config.image_size)))[1]
    expert_eval_transform = build_transforms(int(expert_checkpoint.get("image_size", config.image_size)))[1]

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
        config=config,
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
        config=config,
    )
    elapsed_seconds = time.time() - start_time

    save_cascade_predictions(results_dir / "valid_cascade_predictions.csv", valid_result, main_class_names)
    save_cascade_predictions(results_dir / "test_cascade_predictions.csv", test_result, main_class_names)
    save_confusion_matrix(results_dir / "valid_confusion_matrix.csv", valid_result["confusion_matrix"], main_class_names)
    save_confusion_matrix(results_dir / "test_confusion_matrix.csv", test_result["confusion_matrix"], main_class_names)

    summary = build_cascade_run_summary(
        config=config,
        valid_result=valid_result,
        test_result=test_result,
        samples_by_split=all_samples_by_split,
        elapsed_seconds=elapsed_seconds,
        device=device,
        amp_enabled=amp_enabled,
        gpu_name=gpu_name,
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
