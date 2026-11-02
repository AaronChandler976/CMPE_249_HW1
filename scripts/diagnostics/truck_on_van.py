"""How many confident 'truck' predictions (COCO id 8, or KITTI truck for YOLO-World) land on KITTI *vans*?
Vans are ignore regions only for the car category, so such boxes count as truck FPs.

  python scripts/diagnostics/truck_on_van.py
Needs results/predictions/*.json (git-ignored -> run the predict stage first) + KITTI native GT.
Source of finding F3 in docs/findings.md.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
gt = json.load(open(ROOT / "data" / "kitti" / "annotations" / "kitti_val_native.json"))
cats = {c["id"]: c["name"] for c in gt["categories"]}
boxes = defaultdict(lambda: defaultdict(list))
for a in gt["annotations"]:
    boxes[a["image_id"]][cats[a["category_id"]]].append(a["bbox"])


def overlap_frac(d, g):  # intersection / area(d), xywh
    ix = max(0, min(d[0] + d[2], g[0] + g[2]) - max(d[0], g[0]))
    iy = max(0, min(d[1] + d[3], g[1] + g[3]) - max(d[1], g[1]))
    return ix * iy / max(d[2] * d[3], 1e-9)


def iou(d, g):
    ix = max(0, min(d[0] + d[2], g[0] + g[2]) - max(d[0], g[0]))
    iy = max(0, min(d[1] + d[3], g[1] + g[3]) - max(d[1], g[1]))
    i = ix * iy
    return i / (d[2] * d[3] + g[2] * g[3] - i)


for run, truck_id in [("yolo11s__kitti", 8), ("rtdetr_l__kitti", 8), ("frcnn_r50_v2__kitti", 8),
                      ("fcos_r50__kitti", 8), ("yoloworld_s__kitti", 3)]:  # kitti idx: car1 van2 truck3
    dets = json.load(open(ROOT / "results" / "predictions" / f"{run}.json"))
    t = [d for d in dets if d["category_id"] == truck_id and d["score"] >= 0.3]
    on_van = on_truck = 0
    for d in t:
        g = boxes[d["image_id"]]
        if any(iou(d["bbox"], x) >= 0.5 for x in g.get("truck", [])):
            on_truck += 1
        elif any(overlap_frac(d["bbox"], x) >= 0.5 for x in g.get("van", [])):
            on_van += 1
    print(f"{run:22s} truck preds@0.3: {len(t):5d}  on a real truck (IoU>=.5): {on_truck:4d}  "
          f"on a van instead: {on_van:4d} ({on_van / max(len(t), 1):.0%})")
