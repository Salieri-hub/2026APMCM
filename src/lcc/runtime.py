import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import build_backbone_suffix, build_output_slug


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE_REL = Path(".cache") / "huggingface"


def configure_huggingface_cache(repo_root: Path = PROJECT_ROOT) -> Path:
    cache_root = repo_root / DEFAULT_HF_CACHE_REL
    hub_cache = cache_root / "hub"
    cache_root.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache_root


HF_CACHE_ROOT = configure_huggingface_cache(PROJECT_ROOT)
HF_HUB_CACHE_DIR = HF_CACHE_ROOT / "hub"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_output_subdirs(output_dir: Path) -> Tuple[Path, Path]:
    output_dir = output_dir.resolve()
    if output_dir.name in {"outputs", "weights", "results"}:
        raise ValueError(
            "--output-dir must include an experiment name, for example outputs/v2.0_pretrained_ce_b4."
        )

    if output_dir.parent.name in {"weights", "results"}:
        outputs_root = output_dir.parent.parent
        experiment_name = output_dir.name
    else:
        outputs_root = output_dir.parent
        experiment_name = output_dir.name

    weights_dir = outputs_root / "weights" / experiment_name
    results_dir = outputs_root / "results" / experiment_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir, results_dir


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access a GPU. "
                "Install a CUDA-enabled PyTorch build and verify that torch.cuda.is_available() returns True."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_num_workers(requested_workers: Optional[int], device: torch.device) -> int:
    if requested_workers is not None:
        return max(requested_workers, 0)

    cpu_count = os.cpu_count() or 0
    if device.type == "cuda" and cpu_count > 1:
        return min(4, cpu_count)
    return 0


def resolve_amp_enabled(requested_amp: Optional[bool], device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if requested_amp is None:
        return True
    return requested_amp


def autocast_context(device: torch.device, amp_enabled: bool):
    if device.type == "cuda" and amp_enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def resolve_default_data_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "闄勪欢" / "Data",
        repo_root / "闄勪欢" / "Data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_training_output_dir(
    output_dir: Optional[Path],
    repo_root: Path,
    device: torch.device,
    run_mode: str,
    model_name: str,
    pretrained: bool,
    class_names: Sequence[str],
) -> Path:
    if output_dir is not None:
        return output_dir

    backbone_suffix = build_backbone_suffix(model_name)

    if run_mode == "expert":
        return repo_root / "outputs" / f"expert_{build_output_slug(class_names)}{backbone_suffix}"

    if device.type == "cuda":
        default_output_name = "v2.0_pretrained_ce" if pretrained else "v1.1_scratch_ce_cuda"
    else:
        default_output_name = "v1.0_scratch_ce_cpu"
    return repo_root / "outputs" / f"{default_output_name}{backbone_suffix}"
