from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_EXPERIMENT = "v3.2_pretrained_focal_ls_cosine_cutmix_b4"
DEFAULT_CASCADE_MAIN_EXPERIMENT = "v3.2_pretrained_focal_ls_cosine_cutmix_b4"
DEFAULT_CASCADE_EXPERT_EXPERIMENT = "expert_pair_ad_sq_v3.2_pretrained_focal_ls_cosine_cutmix_b4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cla_pre.csv from /testdata with single-model or cascade inference.")
    parser.add_argument(
        "--run-mode",
        choices=["single", "cascade"],
        default="single",
        help="Inference mode. Defaults to single.",
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        type=Path,
        default=None,
        help="Input test image directory. Defaults to /testdata.",
    )
    parser.add_argument("--testdata-dir", dest="input_dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Single-model checkpoint path. Defaults to "
            f"{DEFAULT_SINGLE_EXPERIMENT}/best_model.pt."
        ),
    )
    parser.add_argument(
        "--main-checkpoint",
        type=Path,
        default=None,
        help=(
            "Cascade main-model checkpoint path. Defaults to "
            f"{DEFAULT_CASCADE_MAIN_EXPERIMENT}/best_model.pt."
        ),
    )
    parser.add_argument(
        "--expert-checkpoint",
        type=Path,
        default=None,
        help=(
            "Cascade expert checkpoint path. Defaults to "
            f"{DEFAULT_CASCADE_EXPERT_EXPERIMENT}/best_model.pt."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        type=Path,
        default=Path("cla_pre.csv"),
        help="Output CSV path. Defaults to ./cla_pre.csv.",
    )
    parser.add_argument("--output-file", dest="output_file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--expert-trigger-topk",
        type=int,
        default=2,
        help="Cascade trigger top-k. Defaults to 2.",
    )
    parser.add_argument(
        "--expert-margin-threshold",
        type=float,
        default=0.12,
        help="Cascade trigger margin threshold. Defaults to 0.12.",
    )
    parser.set_defaults(amp=None)
    return parser.parse_args()


def resolve_testdata_dir(requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested.resolve()

    system_drive = os.environ.get("SystemDrive", "C:")
    candidates = [Path("/testdata"), Path(f"{system_drive}/testdata"), Path("C:/testdata")]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find testdata directory. Tried: {', '.join(str(p) for p in candidates)}")


def iter_image_paths(testdata_dir: Path) -> List[Path]:
    image_paths = [
        path
        for path in testdata_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=lambda path: path.relative_to(testdata_dir).as_posix())
    return image_paths


def checkpoint_candidates(experiment_name: str) -> List[Path]:
    return [
        PROJECT_ROOT / "outputs" / "results" / experiment_name / "best_model.pt",
        PROJECT_ROOT / "outputs" / "weights" / experiment_name / "best_model.pt",
    ]


def resolve_checkpoint(requested: Optional[Path], candidates: Sequence[Path], label: str) -> Path:
    if requested is not None:
        checkpoint_path = requested.resolve()
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Default {label} checkpoint not found. Tried: {', '.join(str(path) for path in candidates)}")


def load_image(image_path: Path):
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB")


def predict_single_submission_rows(
    checkpoint_path: Path,
    testdata_dir: Path,
    image_paths: Sequence[Path],
    device_name: str,
    amp: Optional[bool],
) -> List[Tuple[str, str]]:
    try:
        from .data import build_transforms
        from .models import load_model_from_checkpoint, predict_probabilities
        from .runtime import resolve_amp_enabled, resolve_device
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing inference dependency. Install requirements with "
            "`python -m pip install -r requirements.txt` before running main.py."
        ) from error

    checkpoint_path = checkpoint_path.resolve()
    device = resolve_device(device_name)
    amp_enabled = resolve_amp_enabled(amp, device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint.get("image_size") or 224)
    _, eval_transform = build_transforms(image_size)

    rows: List[Tuple[str, str]] = []
    for image_path in image_paths:
        image = load_image(image_path)
        probs = predict_probabilities(model, image, eval_transform, device, amp_enabled)
        pred_index = max(range(len(probs)), key=probs.__getitem__)
        try:
            image_name = image_path.relative_to(testdata_dir).as_posix()
        except ValueError:
            image_name = image_path.name
        rows.append((image_name, class_names[pred_index]))
    return rows


def predict_cascade_submission_rows(
    main_checkpoint_path: Path,
    expert_checkpoint_path: Path,
    testdata_dir: Path,
    image_paths: Sequence[Path],
    device_name: str,
    amp: Optional[bool],
    expert_trigger_topk: int,
    expert_margin_threshold: float,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    try:
        from .cascade import merge_main_and_expert_probabilities, should_invoke_expert
        from .data import build_transforms
        from .models import load_model_from_checkpoint, predict_probabilities
        from .runtime import resolve_amp_enabled, resolve_device
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing inference dependency. Install requirements with "
            "`python -m pip install -r requirements.txt` before running main.py."
        ) from error

    main_checkpoint_path = main_checkpoint_path.resolve()
    expert_checkpoint_path = expert_checkpoint_path.resolve()
    device = resolve_device(device_name)
    amp_enabled = resolve_amp_enabled(amp, device)
    main_model, main_checkpoint = load_model_from_checkpoint(main_checkpoint_path, device)
    expert_model, expert_checkpoint = load_model_from_checkpoint(expert_checkpoint_path, device)

    main_class_names = list(main_checkpoint["class_names"])
    expert_class_names = list(expert_checkpoint["class_names"])
    if len(expert_class_names) < 2:
        raise ValueError(f"Expert checkpoint must contain at least two classes, got {expert_class_names}.")
    if not set(expert_class_names).issubset(set(main_class_names)):
        raise ValueError(
            "Expert checkpoint classes must be a subset of the main checkpoint classes. "
            f"Main: {main_class_names}, expert: {expert_class_names}"
        )

    main_image_size = int(main_checkpoint.get("image_size") or 224)
    expert_image_size = int(expert_checkpoint.get("image_size") or 224)
    _, main_eval_transform = build_transforms(main_image_size)
    _, expert_eval_transform = build_transforms(expert_image_size)

    rows: List[Tuple[str, str]] = []
    expert_invocations = 0
    expert_changed_predictions = 0
    for image_path in image_paths:
        image = load_image(image_path)
        main_probs = predict_probabilities(main_model, image, main_eval_transform, device, amp_enabled)
        main_pred_index = max(range(len(main_probs)), key=main_probs.__getitem__)
        invoke_expert, _, _ = should_invoke_expert(
            main_probs=main_probs,
            main_class_names=main_class_names,
            expert_class_names=expert_class_names,
            top_k=expert_trigger_topk,
            margin_threshold=expert_margin_threshold,
        )

        final_probs = list(main_probs)
        if invoke_expert:
            expert_invocations += 1
            expert_probs = predict_probabilities(expert_model, image, expert_eval_transform, device, amp_enabled)
            final_probs = merge_main_and_expert_probabilities(
                main_probs=main_probs,
                main_class_names=main_class_names,
                expert_probs=expert_probs,
                expert_class_names=expert_class_names,
            )

        final_pred_index = max(range(len(final_probs)), key=final_probs.__getitem__)
        if final_pred_index != main_pred_index:
            expert_changed_predictions += 1

        try:
            image_name = image_path.relative_to(testdata_dir).as_posix()
        except ValueError:
            image_name = image_path.name
        rows.append((image_name, main_class_names[final_pred_index]))

    return rows, {
        "expert_invocations": expert_invocations,
        "expert_changed_predictions": expert_changed_predictions,
    }


def save_submission(output_path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    testdata_dir = resolve_testdata_dir(args.input_dir)
    image_paths = iter_image_paths(testdata_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {testdata_dir}")
    output_path = args.output_file.resolve()

    if args.run_mode == "cascade":
        main_checkpoint = resolve_checkpoint(
            args.main_checkpoint if args.main_checkpoint is not None else args.checkpoint,
            checkpoint_candidates(DEFAULT_CASCADE_MAIN_EXPERIMENT),
            "cascade main-model",
        )
        expert_checkpoint = resolve_checkpoint(
            args.expert_checkpoint,
            checkpoint_candidates(DEFAULT_CASCADE_EXPERT_EXPERIMENT),
            "cascade expert-model",
        )
        rows, cascade_stats = predict_cascade_submission_rows(
            main_checkpoint_path=main_checkpoint,
            expert_checkpoint_path=expert_checkpoint,
            testdata_dir=testdata_dir,
            image_paths=image_paths,
            device_name=args.device,
            amp=args.amp,
            expert_trigger_topk=args.expert_trigger_topk,
            expert_margin_threshold=args.expert_margin_threshold,
        )
        save_submission(output_path, rows)
        print(f"Saved {len(rows)} cascade predictions to {output_path}")
        print(f"Cascade stats: {cascade_stats}")
        return

    checkpoint_path = resolve_checkpoint(
        args.checkpoint,
        checkpoint_candidates(DEFAULT_SINGLE_EXPERIMENT),
        "single-model",
    )
    rows = predict_single_submission_rows(checkpoint_path, testdata_dir, image_paths, args.device, args.amp)
    save_submission(output_path, rows)
    print(f"Saved {len(rows)} predictions to {output_path}")
