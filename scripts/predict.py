"""Run one detector over one dataset and save COCO-format detections + run metadata.

  python scripts/predict.py --model yolo11s --dataset kitti [--imgsz 1280 --tag res1280] [--limit 25]

Writes
  results/predictions/<run>.json   COCO results list (category ids in the model's label space)
  results/run_meta/<run>.json      everything needed for the reproducibility table
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from tqdm import tqdm

from common import (META_DIR, PRED_DIR, env_info, load_cfg, load_json, log_event, p, resolve_device,
                    run_name, save_json)
from detectors import build_detector


def dataset_images(cfg: dict, dataset: str, limit: int | None):
    d = cfg["datasets"][dataset]
    gt = load_json(p(d["gt"] if dataset == "coco" else d["gt_native"]))
    imgs = sorted(gt["images"], key=lambda r: r["id"])
    img_dir = p(d["images"])
    imgs = [r for r in imgs if (img_dir / r["file_name"]).exists()]   # quick mode = partial download
    if limit:
        imgs = imgs[:limit]
    return imgs, img_dir


def run(model: str, dataset: str, imgsz=None, tag=None, limit=None, cfg=None) -> str:
    cfg = cfg or load_cfg()
    dev = resolve_device(cfg["device"])
    name = run_name(model, dataset, tag)
    det = build_detector(model, dataset, cfg, dev, imgsz=imgsz)
    imgs, img_dir = dataset_images(cfg, dataset, limit)

    results, times = [], []
    t_start = time.time()
    for i, r in enumerate(tqdm(imgs, desc=name)):
        xyxy, scores, cats, timing = det(str(img_dir / r["file_name"]))
        if i >= 5:                      # skip first iterations (CUDA warm-up) in the mean
            times.append(timing)
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, cats):
            results.append({"image_id": r["id"], "category_id": int(c),
                            "bbox": [round(float(x1), 2), round(float(y1), 2),
                                     round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                            "score": round(float(s), 5)})
    wall = time.time() - t_start

    mcfg = cfg["models"][model]
    mean_t = {k: float(np.mean([t[k] for t in times])) for k in times[0]} if times else {}
    meta = {
        "run": name, "model": model, "family": mcfg["family"], "impl": mcfg["impl"],
        "weights": mcfg.get("weights") or getattr(det, "weights_name", mcfg.get("arch")),
        "dataset": dataset, "split": "val2017" if dataset == "coco" else "kitti val (Ultralytics 1,496)",
        "n_images": len(imgs), "n_detections": len(results),
        "imgsz": imgsz or mcfg.get("imgsz"),
        "thresholds": {**cfg["eval_thresholds"], "nms_iou": det.nms_iou},   # actual per-model NMS IoU
        "nms": det.nms, "half": getattr(det, "half", False), "device": dev,
        "label_space": det.label_space, "prompts": getattr(det, "prompts", None),
        "image_ids": [r["id"] for r in imgs],
        "wall_time_s": round(wall, 1), "mean_timing_ms": mean_t, "env": env_info(),
        "command": f"python scripts/predict.py --model {model} --dataset {dataset}"
                   + (f" --imgsz {imgsz}" if imgsz else "") + (f" --tag {tag}" if tag else "")
                   + (f" --limit {limit}" if limit else ""),
    }
    save_json(results, PRED_DIR / f"{name}.json", indent=None)
    save_json(meta, META_DIR / f"{name}.json")
    log_event("predict", run=name, n_images=len(imgs), wall_time_s=meta["wall_time_s"])
    print(f"[predict] {name}: {len(imgs)} images, {len(results)} dets, {wall:.1f}s")
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["coco", "kitti"])
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.dataset, a.imgsz, a.tag, a.limit)


if __name__ == "__main__":
    main()
