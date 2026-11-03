"""Error breakdown (a light TIDE-style taxonomy) + annotated failure images for KITTI runs.

  python scripts/failure_analysis.py --run yolo11s__kitti [--target kitti_cocomap] [--k 8]

At a fixed operating point (vis conf, IoU 0.5) every prediction / GT box is classified:
  TP            IoU>=0.5 with an unmatched GT of the same class
  FP-duplicate  IoU>=0.5 with an already-matched GT of the same class  (NMS failure)
  FP-class      IoU>=0.5 with a GT of a different class, or >=50% inside an ignore region of a
                different class (confusion, e.g. a 'truck' on a Van, whose ignore region is 'car')
  FP-loc        0.1<=IoU<0.5 with a GT of the same class                (bad box)
  FP-background IoU<0.1 with every GT                                    (hallucination)
  FN            GT never matched, reported per class x scale
Predictions on iscrowd (ignore) regions are skipped, exactly like pycocotools.

Writes results/failures/<run>/{summary.json, *.jpg}
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from common import META_DIR, PRED_DIR, RESULTS, load_cfg, load_json, log_event, p, save_json
from evaluate import convert, gt_path


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [N,4], b [M,4] in xywh -> IoU [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(ax1[:, None], bx1), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(ay1[:, None], by1), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def scale_of(area: float) -> str:
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def analyze(run: str, target: str, k: int, conf: float, cfg: dict) -> dict:
    import cv2
    meta = load_json(META_DIR / f"{run}.json")
    gt = load_json(gt_path(cfg, target))
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    dets = [d for d in convert(load_json(PRED_DIR / f"{run}.json"), meta["label_space"], target, cfg)
            if d["score"] >= conf]
    keep_ids = set(meta["image_ids"])
    imgs = {r["id"]: r for r in gt["images"] if r["id"] in keep_ids}
    g_by, d_by = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        if a["image_id"] in imgs:
            g_by[a["image_id"]].append(a)
    for d in dets:
        d_by[d["image_id"]].append(d)

    counts, fn_cs, gt_cs = Counter(), Counter(), Counter()
    per_image = []
    for iid in imgs:
        G = [a for a in g_by[iid] if not a["iscrowd"]]
        C = [a for a in g_by[iid] if a["iscrowd"]]
        D = sorted(d_by[iid], key=lambda d: -d["score"])
        gb = np.array([a["bbox"] for a in G]).reshape(-1, 4)
        cb = np.array([a["bbox"] for a in C]).reshape(-1, 4)
        db = np.array([d["bbox"] for d in D]).reshape(-1, 4)
        I = iou_matrix(db, gb)
        # crowd: pycocotools uses intersection / det-area
        Ic = np.zeros((len(D), len(C)))
        if len(C) and len(D):
            for j, c in enumerate(cb):
                x1 = np.maximum(db[:, 0], c[0]); y1 = np.maximum(db[:, 1], c[1])
                x2 = np.minimum(db[:, 0] + db[:, 2], c[0] + c[2]); y2 = np.minimum(db[:, 1] + db[:, 3], c[1] + c[3])
                inter_j = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                Ic[:, j] = inter_j / np.maximum(db[:, 2] * db[:, 3], 1e-9)
        matched = [False] * len(G)
        labels = []
        for i, d in enumerate(D):
            same = [j for j, g in enumerate(G) if g["category_id"] == d["category_id"]]
            best_j, best = -1, 0.0
            for j in same:
                if not matched[j] and I[i, j] >= 0.5 and I[i, j] > best:
                    best_j, best = j, I[i, j]
            if best_j >= 0:
                matched[best_j] = True
                labels.append("TP")
                continue
            crowd_same = [j for j, c in enumerate(C) if c["category_id"] == d["category_id"]]
            if any(Ic[i, j] >= 0.5 for j in crowd_same):
                labels.append("ignored")
                continue
            if any(I[i, j] >= 0.5 for j in same):
                labels.append("FP-duplicate")
            elif any(I[i, j] >= 0.5 for j in range(len(G))) or \
                    any(Ic[i, j] >= 0.5 for j, c in enumerate(C) if c["category_id"] != d["category_id"]):
                # ignore regions of another class count too (e.g. 'truck' on a Van, whose ignore region
                # is 'car'); same overlap rule as pycocotools crowd matching and truck_on_van.py
                labels.append("FP-class")
            elif any(I[i, j] >= 0.1 for j in same):
                labels.append("FP-loc")
            else:
                labels.append("FP-background")
        for lab in labels:
            counts[lab] += 1
        for j, g in enumerate(G):
            key = f"{cats[g['category_id']]}/{scale_of(g['area'])}"
            gt_cs[key] += 1
            if not matched[j]:
                counts["FN"] += 1
                fn_cs[key] += 1
        n_err = labels.count("FP-background") + labels.count("FP-duplicate") + labels.count("FP-class") \
            + labels.count("FP-loc") + matched.count(False)
        per_image.append((n_err, iid, G, matched, D, labels))

    out_dir = RESULTS / "failures" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = p(cfg["datasets"][meta["dataset"]]["images"])
    colors = {"TP": (214, 120, 42), "FP": (72, 73, 227), "FN": (0, 161, 237), "GT": (122, 175, 27)}
    worst = sorted(per_image, key=lambda t: -t[0])[:k]
    saved = []
    for n_err, iid, G, matched, D, labels in worst:
        im = cv2.imread(str(img_dir / imgs[iid]["file_name"]))
        for g, m in zip(G, matched):
            x, y, w, h = map(int, g["bbox"])
            cv2.rectangle(im, (x, y), (x + w, y + h), colors["GT"] if m else colors["FN"], 1 if m else 3)
        for d, lab in zip(D, labels):
            if lab == "ignored":
                continue
            x, y, w, h = map(int, d["bbox"])
            c = colors["TP"] if lab == "TP" else colors["FP"]
            cv2.rectangle(im, (x, y), (x + w, y + h), c, 1)
            if lab != "TP":
                cv2.putText(im, f"{lab[3:]} {cats.get(d['category_id'], '?')} {d['score']:.2f}",
                            (x, max(y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        f = out_dir / f"{n_err:03d}err_{imgs[iid]['file_name'].rsplit('.', 1)[0]}.jpg"
        cv2.imwrite(str(f), im)
        saved.append(f.name)

    recall_cs = {kk: round(1 - fn_cs[kk] / v, 3) for kk, v in sorted(gt_cs.items())}
    summary = {"run": run, "target": target, "conf": conf, "iou": 0.5, "counts": dict(counts),
               "fn_by_class_scale": dict(fn_cs), "gt_by_class_scale": dict(gt_cs),
               "recall_by_class_scale": recall_cs, "images_saved": saved,
               "legend": "thick orange=missed GT (FN), thin green=matched GT, blue=TP, red=FP (label = type)"}
    save_json(summary, out_dir / "summary.json")
    log_event("failure_analysis", run=run, target=target, counts=dict(counts))
    print(f"[failures] {run}: {dict(counts)}")
    return summary


def main() -> None:
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", default="kitti_cocomap")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--conf", type=float, default=cfg["vis_thresholds"]["conf"])
    a = ap.parse_args()
    analyze(a.run, a.target, a.k, a.conf, cfg)


if __name__ == "__main__":
    main()
