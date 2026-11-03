"""Shared helpers: paths, config loading, environment capture, JSON I/O, logging."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "experiments.yaml"
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"     # large, git-ignored
META_DIR = RESULTS / "run_meta"        # small, committed
METRIC_DIR = RESULTS / "metrics"       # small, committed
TABLE_DIR = RESULTS / "tables"
FIG_DIR = RESULTS / "figures"
LOG_PATH = RESULTS / "run_log.jsonl"

# The command line as launched. Captured once at import because run_all.py rewrites sys.argv
# before calling each stage's main().
_ARGV = " ".join(sys.argv)


def scrub(text: str) -> str:
    """Replace the absolute repo path with '<repo>' so committed files carry no personal paths."""
    return text.replace(str(ROOT), "<repo>").replace(ROOT.as_posix(), "<repo>")


def p(rel: str | Path) -> Path:
    """Resolve a repo-relative path."""
    rel = Path(rel)
    return rel if rel.is_absolute() else ROOT / rel


def load_cfg(path: Path = CFG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("DETBENCH_DEVICE"):          # set by run_all.py --device
        cfg["device"] = os.environ["DETBENCH_DEVICE"]
    return cfg


def save_json(obj, path: Path, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_name(model: str, dataset: str, tag: str | None = None) -> str:
    return f"{model}__{dataset}" + (f"__{tag}" if tag else "")


def resolve_device(cfg_device: str) -> str:
    import torch
    if str(cfg_device).lower() == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return str(cfg_device)


def torch_device(dev: str):
    import torch
    return torch.device("cpu" if dev == "cpu" else f"cuda:{dev}" if dev.isdigit() else dev)


def sync(dev: str) -> None:
    if dev != "cpu":
        import torch
        torch.cuda.synchronize()


def env_info() -> dict:
    """Everything the 'Reproducibility' section needs about hardware/software."""
    import torch
    import torchvision
    info = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import pycocotools  # noqa: F401
        from importlib.metadata import version
        info["pycocotools"] = version("pycocotools")
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info["gpu"] = prop.name
        info["gpu_mem_gb"] = round(prop.total_memory / 1024**3, 1)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            info["driver"] = out.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return info


def log_event(event: str, **kw) -> None:
    """Append one line to results/run_log.jsonl (the 'command log' deliverable)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "event": event,
           "argv": _ARGV, "cwd": scrub(os.getcwd()), **kw}
    line = json.dumps(rec)
    # absolute paths inside kw values appear JSON-escaped, so scrub the escaped form too
    line = line.replace(json.dumps(str(ROOT))[1:-1], "<repo>")
    line = line.replace(json.dumps(ROOT.as_posix())[1:-1], "<repo>")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]

# Contiguous 80-class index -> official COCO category_id (1..90 with gaps)
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
                75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
