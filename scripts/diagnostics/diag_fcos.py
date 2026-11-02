"""Diagnose FCOS COCO mAP gap: FP16 vs FP32 x eval thresholds vs torchvision defaults.
Runs on the first N COCO val images; no pipeline files are modified.

  python scripts/diagnostics/diag_fcos.py [N=500] [model=fcos_r50]

Result that motivated `models.fcos_r50.half: false` (500 imgs, 2026-11-01):
  fp16 0.001/300 AP 36.2 | fp32 0.001/300 AP 43.1 | fp16 0.2/100 AP 36.1 | fp32 0.2/100 AP 42.8
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common import load_cfg, p, resolve_device  # noqa: E402
from detectors import build_detector  # noqa: E402
from evaluate import coco_eval  # noqa: E402
from predict import dataset_images  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MODEL = sys.argv[2] if len(sys.argv) > 2 else "fcos_r50"
cfg0 = load_cfg()
dev = resolve_device(cfg0["device"])
imgs, img_dir = dataset_images(cfg0, "coco", N)
ids = [r["id"] for r in imgs]

variants = {
    "fp16 conf0.001 max300 (original)":(True, {"conf": 0.001, "nms_iou": 0.7, "max_det": 300}),
    "fp32 conf0.001 max300":            (False, {"conf": 0.001, "nms_iou": 0.7, "max_det": 300}),
    "fp16 tv-default 0.2/100":          (True, {"conf": 0.2, "nms_iou": 0.7, "max_det": 100}),
    "fp32 tv-default 0.2/100":          (False, {"conf": 0.2, "nms_iou": 0.7, "max_det": 100}),
}
for name, (half, thr) in variants.items():
    cfg = copy.deepcopy(cfg0)
    cfg["half"] = half
    cfg["models"][MODEL]["half"] = half   # per-model override (fcos_r50 has half: false) must be set too
    det = build_detector(MODEL, "coco", cfg, dev, thr=thr)
    dets = []
    for r in imgs:
        xyxy, s, c, _ = det(str(img_dir / r["file_name"]))
        for (x1, y1, x2, y2), sc, cc in zip(xyxy, s, c):
            dets.append({"image_id": r["id"], "category_id": int(cc),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(sc)})
    E, _, _ = coco_eval(p(cfg["datasets"]["coco"]["gt"]), dets, ids)
    print(f"{MODEL} | {name:34s} | AP={E.stats[0]*100:.1f} AP50={E.stats[1]*100:.1f} "
          f"dets/img={len(dets)/len(imgs):.0f}", flush=True)
    del det
