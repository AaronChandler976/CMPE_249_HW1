"""Is the YOLO11s-COCO latency outlier a one-time per-input-shape cost? Time the same 200 images twice.

  python scripts/diagnostics/diag_bench.py [model=yolo11s]

Result that motivated `benchmark.warmup: 200` (2026-11-01, warmup 20 here on purpose):
  yolo11s pass 1: mean 51.5 p50 5.2 max 1046 ms, 9 imgs >20 ms = 90 % of time | pass 2: mean 5.3, max 7 ms
"""
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import cv2  # noqa: E402
from common import load_cfg, resolve_device  # noqa: E402
from detectors import build_detector  # noqa: E402
from predict import dataset_images  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "yolo11s"
cfg = load_cfg(); dev = resolve_device(cfg["device"])
imgs, d = dataset_images(cfg, "coco", 200)
arrs = [cv2.imread(str(d / r["file_name"])) for r in imgs]
det = build_detector(model, "coco", cfg, dev)
for i in range(20):
    det(arrs[i])
for p in (1, 2):
    t = np.array([det(a)[3]["total_ms"] for a in arrs])
    print(f"{model} pass {p}: mean {t.mean():.1f} p50 {np.median(t):.1f} max {t.max():.0f} ms, "
          f">20ms: {(t > 20).sum()} imgs, share of total time in those: {t[t > 20].sum() / t.sum():.0%}")
