from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .constants import CLASS_NAMES
except ImportError:
    from lcc.constants import CLASS_NAMES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATASET_SPLITS = ("train", "valid", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate accuracy by first invoking this project's main.py to generate predictions, "
            "then comparing those predictions against folder labels under the test split."
        )
    )
    parser.add_argument(
        "--run-mode",
        choices=["single", "cascade"],
        default="single",
        help="Inference mode passed through to main.py. Defaults to single.",
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        type=Path,
        default=None,
        help=(
            "Dataset root or explicit test directory. If omitted, defaults to /testdata/test "
            "when /testdata contains train/valid/test."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Single-model checkpoint path, or fallback main checkpoint for cascade mode.",
    )
    parser.add_argument("--main-checkpoint", type=Path, default=None, help="Cascade main-model checkpoint path.")
    parser.add_argument("--expert-checkpoint", type=Path, default=None, help="Cascade expert checkpoint path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("main_accuracy_eval"),
        help="Directory used to save main.py predictions, confusion matrix, and metrics summary.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--expert-trigger-topk",
        type=int,
        default=2,
        help="Cascade trigger top-k passed through to main.py. Defaults to 2.",
    )
    parser.add_argument(
        "--expert-margin-threshold",
        type=float,
        default=0.12,
        help="Cascade trigger margin threshold passed through to main.py. Defaults to 0.12.",
    )
    parser.set_defaults(amp=None)
    return parser.parse_args()


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


def resolve_default_testdata_root() -> Path:
    system_drive = os.environ.get("SystemDrive", "C:")
    candidates = [Path("/testdata"), Path(f"{system_drive}/testdata"), Path("C:/testdata")]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find testdata directory. Tried: {', '.join(str(path) for path in candidates)}")


def is_dataset_root(path: Path) -> bool:
    return all((path / split_name).is_dir() for split_name in DATASET_SPLITS)


def resolve_test_input_dir(requested: Optional[Path]) -> Path:
    if requested is None:
        dataset_root = resolve_default_testdata_root()
        if is_dataset_root(dataset_root):
            return (dataset_root / "test").resolve()
        return dataset_root

    requested_path = requested.resolve()
    if not requested_path.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {requested_path}")
    if is_dataset_root(requested_path):
        return (requested_path / "test").resolve()
    return requested_path


def iter_image_paths(root_dir: Path) -> List[Path]:
    image_paths = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=lambda path: path.relative_to(root_dir).as_posix())
    return image_paths


def collect_ground_truth_records(test_dir: Path) -> List[Tuple[str, str, Path]]:
    image_paths = iter_image_paths(test_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {test_dir}")

    records: List[Tuple[str, str, Path]] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(test_dir)
        if len(relative_path.parts) < 2:
            raise ValueError(f"Expected class subdirectories under {test_dir}, got {relative_path}")
        true_label = canonical_class_name(relative_path.parts[0])
        records.append((relative_path.as_posix(), true_label, image_path))
    return records


def run_main_prediction(args: argparse.Namespace, test_dir: Path, output_csv: Path) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--run-mode",
        args.run_mode,
        "--input",
        str(test_dir),
        "--output",
        str(output_csv),
        "--device",
        args.device,
    ]

    if args.amp is True:
        command.append("--amp")
    elif args.amp is False:
        command.append("--no-amp")

    if args.run_mode == "cascade":
        if args.checkpoint is not None and args.main_checkpoint is None:
            command.extend(["--checkpoint", str(args.checkpoint.resolve())])
        if args.main_checkpoint is not None:
            command.extend(["--main-checkpoint", str(args.main_checkpoint.resolve())])
        if args.expert_checkpoint is not None:
            command.extend(["--expert-checkpoint", str(args.expert_checkpoint.resolve())])
        command.extend(["--expert-trigger-topk", str(args.expert_trigger_topk)])
        command.extend(["--expert-margin-threshold", str(args.expert_margin_threshold)])
    elif args.checkpoint is not None:
        command.extend(["--checkpoint", str(args.checkpoint.resolve())])

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return command


def load_prediction_rows(prediction_csv: Path) -> List[Tuple[str, str]]:
    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [(row["image_name"], row["label"]) for row in reader]


def resolve_report_class_names(true_labels: Sequence[str], pred_labels: Sequence[str]) -> List[str]:
    seen = set(true_labels) | set(pred_labels)
    ordered = [class_name for class_name in CLASS_NAMES if class_name in seen]
    ordered.extend(sorted(seen - set(CLASS_NAMES)))
    return ordered


def build_confusion_matrix(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    class_names: Sequence[str],
) -> List[List[int]]:
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    matrix = [[0 for _ in class_names] for _ in class_names]
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[class_to_index[true_label]][class_to_index[pred_label]] += 1
    return matrix


def build_per_class_accuracy(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    per_class: Dict[str, Dict[str, Any]] = {}
    for class_name in class_names:
        total = sum(1 for label in true_labels if label == class_name)
        correct = sum(
            1
            for true_label, pred_label in zip(true_labels, pred_labels)
            if true_label == class_name and pred_label == class_name
        )
        per_class[class_name] = {
            "total": total,
            "correct": correct,
            "accuracy": float(correct / total) if total else 0.0,
        }
    return per_class


def build_accuracy_result(
    ground_truth_records: Sequence[Tuple[str, str, Path]],
    prediction_rows: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    prediction_by_name = {image_name: pred_label for image_name, pred_label in prediction_rows}
    ground_truth_names = {image_name for image_name, _, _ in ground_truth_records}

    missing_predictions = sorted(image_name for image_name, _, _ in ground_truth_records if image_name not in prediction_by_name)
    extra_predictions = sorted(set(prediction_by_name) - ground_truth_names)
    if missing_predictions or extra_predictions:
        raise ValueError(
            "Prediction/ground-truth file sets do not match. "
            f"Missing predictions: {missing_predictions[:5]}, extra predictions: {extra_predictions[:5]}"
        )

    records: List[Dict[str, Any]] = []
    true_labels: List[str] = []
    pred_labels: List[str] = []
    correct_images = 0

    for image_name, true_label, image_path in ground_truth_records:
        pred_label = prediction_by_name[image_name]
        is_correct = pred_label == true_label
        if is_correct:
            correct_images += 1
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        records.append(
            {
                "image_name": image_name,
                "image_path": str(image_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": is_correct,
            }
        )

    class_names = resolve_report_class_names(true_labels, pred_labels)
    confusion_matrix = build_confusion_matrix(true_labels, pred_labels, class_names)
    total_images = len(records)
    summary: Dict[str, Any] = {
        "total_images": total_images,
        "correct_images": correct_images,
        "accuracy": float(correct_images / total_images) if total_images else 0.0,
        "class_names": class_names,
        "per_class": build_per_class_accuracy(true_labels, pred_labels, class_names),
        "confusion_matrix": confusion_matrix,
    }
    return {
        "summary": summary,
        "records": records,
        "class_names": class_names,
        "confusion_matrix": confusion_matrix,
    }


def save_prediction_records(output_path: Path, records: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "true_label", "pred_label", "correct", "image_path"])
        for record in records:
            writer.writerow(
                [
                    record["image_name"],
                    record["true_label"],
                    record["pred_label"],
                    record["correct"],
                    record["image_path"],
                ]
            )


def save_confusion_matrix(output_path: Path, matrix: Sequence[Sequence[int]], class_names: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def save_summary(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    test_dir = resolve_test_input_dir(args.input_dir)
    ground_truth_records = collect_ground_truth_records(test_dir)
    output_dir = args.output_dir.resolve()
    prediction_csv = output_dir / "main_predictions.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    executed_command = run_main_prediction(args, test_dir, prediction_csv)
    prediction_rows = load_prediction_rows(prediction_csv)
    result = build_accuracy_result(ground_truth_records, prediction_rows)

    summary = {
        "mode": args.run_mode,
        "input_dir": str(test_dir),
        "main_prediction_csv": str(prediction_csv),
        "executed_command": executed_command,
        **result["summary"],
    }

    save_prediction_records(output_dir / "predictions_with_truth.csv", result["records"])
    save_confusion_matrix(output_dir / "confusion_matrix.csv", result["confusion_matrix"], result["class_names"])
    save_summary(output_dir / "metrics_summary.json", summary)

    print(f"Evaluated {summary['total_images']} test images from {test_dir}")
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct_images']}/{summary['total_images']})")
    print(f"Saved main.py predictions to {prediction_csv}")
    print(f"Saved truth-aligned predictions to {output_dir / 'predictions_with_truth.csv'}")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.csv'}")
    print(f"Saved metrics summary to {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
