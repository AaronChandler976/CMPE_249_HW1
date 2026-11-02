"""Taxonomy SENSITIVITY check: how much do KITTI scores depend on how Van is treated?
The main mapping (configs/experiments.yaml -> taxonomy) is NOT changed; alternative GT files are
built in a temp folder from data/kitti/annotations/kitti_val_cocomap.json and the existing
predictions are re-scored.

  python scripts/diagnostics/sensitivity_taxonomy.py

Variants
  main        the pipeline's mapping: Van = ignore region for car only (must reproduce committed APs)
  van_ign_trk Van = ignore region for car AND truck   (tests finding F3: van->"truck" FPs)
  van_as_car  Van scored as car (iscrowd=0)            (tests surprise 2: car AP higher on KITTI)

Vans are identified as category car + iscrowd=1 (the only KITTI class mapped to an ignored car).
Writes results/metrics/sensitivity_taxonomy.json and results/tables/sensitivity_taxonomy.md.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common import META_DIR, METRIC_DIR, PRED_DIR, TABLE_DIR, load_cfg, load_json, p, save_json  # noqa: E402
from evaluate import coco_eval, convert, per_class  # noqa: E402

RUNS = ["yolo11s__kitti", "rtdetr_l__kitti", "frcnn_r50_v2__kitti", "fcos_r50__kitti",
        "yoloworld_s__kitti", "yolo11s_kitti_ft__kitti"]


def variants(gt: dict) -> dict:
    vans = [a for a in gt["annotations"] if a["category_id"] == 3 and a.get("iscrowd")]
    assert len(vans) == 555, f"expected 555 van ignore boxes, got {len(vans)}"
    out = {"main": gt}
    a = copy.deepcopy(gt)
    nid = max(x["id"] for x in a["annotations"]) + 1
    for i, v in enumerate(vans):
        a["annotations"].append({**v, "id": nid + i, "category_id": 8})   # also ignore for truck
    out["van_ign_trk"] = a
    b = copy.deepcopy(gt)
    for x in b["annotations"]:
        if x["category_id"] == 3 and x.get("iscrowd"):
            x["iscrowd"] = 0                                             # van becomes a scored car
    out["van_as_car"] = b
    return out


def main() -> None:
    cfg = load_cfg()
    gt = load_json(p(cfg["datasets"]["kitti"]["gt_cocomap"]))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        files = {}
        for name, g in variants(gt).items():
            f = Path(td) / f"{name}.json"
            f.write_text(json.dumps(g))
            files[name] = f
        for run in RUNS:
            if not (META_DIR / f"{run}.json").exists():
                continue
            meta = load_json(META_DIR / f"{run}.json")
            dets = convert(load_json(PRED_DIR / f"{run}.json"), meta["label_space"], "kitti_cocomap", cfg)
            for name, f in files.items():
                E, g, _ = coco_eval(f, dets, meta["image_ids"])
                pc = per_class(E, g)
                rows.append({"run": run, "variant": name, "AP": round(E.stats[0] * 100, 1),
                             **{f"AP_{k}": round(v["AP"], 1) for k, v in pc.items()}})
                print(rows[-1], flush=True)
    save_json({"note": "sensitivity only; main results use variant 'main'", "rows": rows},
              METRIC_DIR / "sensitivity_taxonomy.json")
    import pandas as pd
    df = pd.DataFrame(rows)
    (TABLE_DIR / "sensitivity_taxonomy.md").write_text(
        "# Taxonomy sensitivity (KITTI val, 3-class AP@[.5:.95]) - NOT the main results\n\n"
        "main = pipeline mapping; van_ign_trk = Van also ignored for truck; van_as_car = Van scored as car\n\n"
        + df.to_markdown(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
