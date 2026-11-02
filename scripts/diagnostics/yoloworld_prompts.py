"""Prompt SENSITIVITY for YOLO-World on KITTI (tests surprise 3: worst person AP, 12.8).
Same weights, same thresholds; only the text prompts change. Predictions stay in memory - the
pipeline's results/predictions/yoloworld_s__kitti.json and configs are NOT touched.

  python scripts/diagnostics/yoloworld_prompts.py

Each prompt is mapped to a KITTI class index, then to COCO ids via taxonomy.kitti_pred_to_coco,
and scored on kitti_cocomap exactly like the main run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common import load_cfg, p, resolve_device  # noqa: E402
from detectors import build_detector  # noqa: E402
from evaluate import coco_eval, convert, per_class  # noqa: E402
from predict import dataset_images  # noqa: E402

# prompt -> KITTI class it stands for
VARIANTS = {
    "baseline (pipeline)": [("car", "car"), ("van", "van"), ("truck", "truck"), ("pedestrian", "pedestrian"),
                            ("person sitting", "person_sitting"), ("cyclist", "cyclist"), ("tram", "tram")],
    "pedestrian->person":  [("car", "car"), ("van", "van"), ("truck", "truck"), ("person", "pedestrian"),
                            ("person sitting", "person_sitting"), ("cyclist", "cyclist"), ("tram", "tram")],
    "4 prompts, person":   [("car", "car"), ("van", "van"), ("truck", "truck"), ("person", "pedestrian")],
}


def main() -> None:
    cfg = load_cfg()
    dev = resolve_device(cfg["device"])
    kitti = cfg["taxonomy"]["kitti_classes"]
    imgs, img_dir = dataset_images(cfg, "kitti", None)
    ids = [r["id"] for r in imgs]
    det = build_detector("yoloworld_s", "kitti", cfg, dev)
    for name, pairs in VARIANTS.items():
        det.model.set_classes([pr for pr, _ in pairs])
        det.map = np.array([kitti.index(k) + 1 for _, k in pairs])
        dets = []
        for r in imgs:
            xyxy, s, c, _ = det(str(img_dir / r["file_name"]))
            for (x1, y1, x2, y2), sc, cc in zip(xyxy, s, c):
                dets.append({"image_id": r["id"], "category_id": int(cc), "score": float(sc),
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
        E, gt, _ = coco_eval(p(cfg["datasets"]["kitti"]["gt_cocomap"]),
                             convert(dets, "kitti", "kitti_cocomap", cfg), ids)
        pc = per_class(E, gt)
        print(f"{name:22s} | AP={E.stats[0]*100:.1f} | person {pc['person']['AP']:.1f} "
              f"(AP50 {pc['person']['AP50']:.1f}) car {pc['car']['AP']:.1f} truck {pc['truck']['AP']:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
