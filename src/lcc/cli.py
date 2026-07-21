import argparse
from pathlib import Path

from .cascade import run_cascade_evaluation
from .config import build_cascade_config, build_train_config
from .constants import DEFAULT_EXPERT_CLASSES, DEFAULT_MODEL_NAME
from .runtime import PROJECT_ROOT, resolve_default_data_dir
from .train import train_and_evaluate


def parse_args() -> argparse.Namespace:
    default_data_dir = resolve_default_data_dir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description=(
            "Problem 2 pipeline: single four-class baseline, expert-branch subset training, "
            "or cascade evaluation that combines a main model with an expert branch."
        )
    )
    parser.add_argument("--run-mode", choices=["single", "expert", "cascade"], default="single")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Experiment slug path. For example outputs/v2.0_pretrained_ce_b4. "
            "Checkpoints are written to outputs/weights/<experiment>/ and reports to outputs/results/<experiment>/."
        ),
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", action="store_true", help="Use timm pretrained weights if available.")
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
        help="Training loss; label smoothing applies to both options.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size. Defaults to 320 for EfficientNet-B4, 288 for EfficientNet-B3, 256 for EfficientNet-B2, 240 for EfficientNet-B1, and 224 for EfficientNet-B0.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor used by cross-entropy and focal loss.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter when --loss focal is selected.",
    )
    parser.add_argument("--class-weighting", choices=["none", "balanced", "manual"], default="none")
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Comma-separated weights in active class order when class-weighting=manual.",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Enable MixUp when > 0. Typical values: 0.2 to 0.4.")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0, help="Enable CutMix when > 0. Typical values: 0.5 to 1.0.")
    parser.add_argument("--mix-prob", type=float, default=1.0, help="Probability of applying MixUp/CutMix to a training batch.")
    parser.add_argument(
        "--mix-switch-prob",
        type=float,
        default=0.5,
        help="When MixUp and CutMix are both enabled, probability of selecting CutMix.",
    )
    parser.add_argument(
        "--feature-attention",
        choices=["none", "se", "cbam"],
        default="none",
        help="Extra attention attached on the final feature map. EfficientNet backbones already include internal SE blocks.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to 4 on CUDA and 0 on CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic backend settings. This is slower on GPU.")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision on CUDA.")
    parser.add_argument(
        "--expert-classes",
        type=str,
        default=",".join(DEFAULT_EXPERT_CLASSES),
        help=(
            "Comma-separated class list for expert-branch training. "
            "Default: adenocarcinoma,squamous.cell.carcinoma"
        ),
    )
    parser.add_argument("--main-checkpoint", type=Path, default=None, help="Main four-class checkpoint used in cascade mode.")
    parser.add_argument("--expert-checkpoint", type=Path, default=None, help="Expert-branch checkpoint used in cascade mode.")
    parser.add_argument(
        "--expert-trigger-topk",
        type=int,
        default=2,
        help="Invoke the expert branch only when the main model's top-k classes all stay inside the expert class subset.",
    )
    parser.add_argument(
        "--expert-margin-threshold",
        type=float,
        default=0.12,
        help="Maximum main-model top1-top2 probability margin for invoking the expert branch. Set a negative value to disable the margin filter.",
    )
    parser.set_defaults(amp=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_mode == "cascade":
        run_cascade_evaluation(build_cascade_config(args))
        return
    train_and_evaluate(build_train_config(args, PROJECT_ROOT))
