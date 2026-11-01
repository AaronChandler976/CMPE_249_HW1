"""Score a prediction file with pycocotools against one GT taxonomy.

  python scripts/evaluate.py --run yolo11s__kitti --target kitti_cocomap

targets
  coco           COCO val2017, 80 classes
  kitti_cocomap  KITTI val relabelled to COCO ids {person, car, truck}; Van / Person_sitting /
                 Cyclist are iscrowd ignore-regions (see configs/experiments.yaml: taxonomy)
  kitti_native   KITTI val, 7 native classes (only for models that output KITTI labels)

Writes results/metrics/<run>__<target>.json : 12 COCO stats + per-class AP / AP50 / AP_S/M/L
"""
from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np

from common import META_DIR, METRIC_DIR, PRED_DIR, load_cfg, load_json, log_event, p, save_json

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def gt_path(cfg: dict, target: str):
    if target == "coco":
        return p(cfg["datasets"]["coco"]["gt"])
    if target == "kitti_cocomap":
        return p(cfg["datasets"]["kitti"]["gt_cocomap"])
    if target == "kitti_native":
        return p(cfg["datasets"]["kitti"]["gt_native"])
    raise ValueError(target)


def convert(dets: list, src: str, target: str, cfg: dict) -> list:
    """Move detections from the model's label space into the target GT's label space."""
    tax = cfg["taxonomy"]
    kitti = tax["kitti_classes"]
    if target == "coco":
        assert src == "coco", f"cannot score {src}-label predictions on COCO"
        return dets
    if target == "kitti_cocomap":
        if src == "coco":
            keep = {1, 3, 8}                       # person, car, truck
            return [d for d in dets if d["category_id"] in keep]
        m = {kitti.index(k) + 1: v for k, v in tax["kitti_pred_to_coco"].items()}
        return [{**d, "category_id": m[d["category_id"]]} for d in dets if d["category_id"] in m]
    if target == "kitti_native":
        assert src == "kitti", "native KITTI eval needs a model that outputs KITTI classes"
        native = tax["native_eval_classes"]
        m = {kitti.index(n) + 1: i + 1 for i, n in enumerate(native)}
        return [{**d, "category_id": m[d["category_id"]]} for d in dets if d["category_id"] in m]
    raise ValueError(target)


def coco_eval(gt_file, dets: list, img_ids: list | None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_file))
    if not dets:
        return None, gt, ""
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
    E = COCOeval(gt, dt, "bbox")
    if img_ids:
        E.params.imgIds = sorted(img_ids)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.evaluate()
        E.accumulate()
        E.summarize()
    return E, gt, buf.getvalue()


def per_class(E, gt) -> dict:
    """precision array: [T(iou thr), R(recall pts), K(classes), A(area: all,S,M,L), M(maxdets)]."""
    prec = E.eval["precision"]
    out = {}

    def ap(x):
        x = x[x > -1]
        return float(np.mean(x)) * 100 if x.size else float("nan")

    for k, cid in enumerate(E.params.catIds):
        name = gt.cats[cid]["name"]
        n_gt = sum(1 for a in gt.loadAnns(gt.getAnnIds(catIds=[cid], imgIds=E.params.imgIds))
                   if not a.get("iscrowd", 0))
        out[name] = {"AP": ap(prec[:, :, k, 0, -1]), "AP50": ap(prec[0, :, k, 0, -1]),
                     "AP75": ap(prec[5, :, k, 0, -1]), "AP_small": ap(prec[:, :, k, 1, -1]),
                     "AP_medium": ap(prec[:, :, k, 2, -1]), "AP_large": ap(prec[:, :, k, 3, -1]),
                     "n_gt": n_gt}
    return out


def run(run_name: str, target: str, cfg=None) -> dict:
    cfg = cfg or load_cfg()
    meta = load_json(META_DIR / f"{run_name}.json")
    dets = convert(load_json(PRED_DIR / f"{run_name}.json"), meta["label_space"], target, cfg)
    E, gt, summary = coco_eval(gt_path(cfg, target), dets, meta.get("image_ids"))
    if E is None:
        stats = {k: 0.0 for k in STAT_NAMES}
        pc = {}
    else:
        # pycocotools returns -1 when a bucket has no GT (e.g. no 'large' objects) -> NaN, not -100
        stats = {k: (round(float(v) * 100, 2) if v >= 0 else float("nan"))
                 for k, v in zip(STAT_NAMES, E.stats)}
        pc = per_class(E, gt)
    res = {"run": run_name, "target": target, "model": meta["model"], "dataset": meta["dataset"],
           "imgsz": meta["imgsz"], "n_images": meta["n_images"], "n_dets_scored": len(dets),
           "stats": stats, "per_class": pc, "pycocotools_summary": summary}
    out = METRIC_DIR / f"{run_name}__{target}.json"
    save_json(res, out)
    log_event("evaluate", run=run_name, target=target, AP=stats["AP"])
    print(f"[eval] {run_name} on {target}: AP={stats['AP']:.1f} AP50={stats['AP50']:.1f} "
          f"APs={stats['AP_small']:.1f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", required=True, choices=["coco", "kitti_cocomap", "kitti_native"])
    a = ap.parse_args()
    run(a.run, a.target)


if __name__ == "__main__":
    main()
