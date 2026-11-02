"""95 % bootstrap confidence intervals for KITTI 3-class AP (and per-class AP) by resampling images.

  python scripts/diagnostics/bootstrap_ci.py [B=1000]

pycocotools' per-image matching (evaluate) runs once per model. Each bootstrap replicate resamples
the 1,496 val images with replacement and re-runs only accumulate + summarize on the corresponding
per-image results, so B=1000 takes minutes. Also reports paired differences vs RT-DETR-L (same
resampled images for both models), which is the right test for "is model A better than B?".
Writes results/metrics/diagnostics/bootstrap_ci.json and results/tables/bootstrap_ci.md.
"""
from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common import META_DIR, METRIC_DIR, PRED_DIR, TABLE_DIR, load_cfg, load_json, p, save_json  # noqa: E402
from evaluate import convert  # noqa: E402

RUNS = {"yolo11s__kitti": "YOLO11s", "rtdetr_l__kitti": "RT-DETR-L", "frcnn_r50_v2__kitti": "Faster R-CNN v2",
        "fcos_r50__kitti": "FCOS", "yoloworld_s__kitti": "YOLO-World-S", "yolo11s__kitti__res1280": "YOLO11s @1280",
        "yolo11s_kitti_ft__kitti": "YOLO11s fine-tuned"}
REF = "rtdetr_l__kitti"


def evaluated(run, cfg, gt):
    from pycocotools.cocoeval import COCOeval
    meta = load_json(META_DIR / f"{run}.json")
    dets = convert(load_json(PRED_DIR / f"{run}.json"), meta["label_space"], "kitti_cocomap", cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.params.imgIds = sorted(meta["image_ids"])
        E.evaluate()
    return E


def stats_for(E, base_eval_imgs, idx, n_img):
    """Recompute AP (and per-class AP) on resampled image indices `idx`."""
    K, A = len(E.params.catIds), len(E.params.areaRng)
    E.evalImgs = [base_eval_imgs[k * A * n_img + a * n_img + j] for k in range(K) for a in range(A) for j in idx]
    E.params.imgIds = list(range(len(idx)))
    E._paramsEval = copy.deepcopy(E.params)
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate()
    prec = E.eval["precision"][:, :, :, 0, -1]          # T x R x K, area=all, maxDets=100
    per = [float(np.mean(x[x > -1])) * 100 if (x > -1).any() else np.nan for x in np.moveaxis(prec, 2, 0)]
    valid = prec[prec > -1]
    return float(np.mean(valid)) * 100, per


def one_run(run: str, B: int):
    """Worker: point estimate + B bootstrap replicates for one model. Resamples come from a fixed
    seed, so every worker draws the SAME image indices -> paired comparisons stay valid."""
    from pycocotools.coco import COCO
    cfg = load_cfg()
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(p(cfg["datasets"]["kitti"]["gt_cocomap"])))
    E = evaluated(run, cfg, gt)
    base, n = list(E.evalImgs), len(E.params.imgIds)
    rng = np.random.default_rng(0)
    idx_sets = [rng.integers(0, n, n) for _ in range(B)]
    cats = [gt.cats[c]["name"] for c in E.params.catIds]
    return run, n, cats, stats_for(E, base, np.arange(n), n), [stats_for(E, base, ix, n) for ix in idx_sets]


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    boots, point, ns, cats = {}, {}, set(), None
    with ProcessPoolExecutor(max_workers=len(RUNS)) as ex:          # one process per model
        for run, n, cats, pt, bs in ex.map(one_run, RUNS, [B] * len(RUNS)):
            ns.add(n)
            point[run], boots[run] = pt, bs
            print(f"{RUNS[run]:20s} AP {pt[0]:.1f}", flush=True)
    assert len(ns) == 1, "all runs must cover the same images for paired bootstrap"

    rows, out = [], {"B": B, "seed": 0, "classes": cats, "runs": {}}
    ref = np.array([b[0] for b in boots[REF]])
    for run, name in RUNS.items():
        ap = np.array([b[0] for b in boots[run]])
        lo, hi = np.percentile(ap, [2.5, 97.5])
        d = ap - ref
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        pc = np.array([b[1] for b in boots[run]])
        pc_ci = {c: [round(float(v), 1) for v in np.nanpercentile(pc[:, i], [2.5, 97.5])] for i, c in enumerate(cats)}
        out["runs"][run] = {"AP": round(point[run][0], 2), "ci95": [round(lo, 2), round(hi, 2)],
                            "diff_vs_rtdetr_ci95": [round(dlo, 2), round(dhi, 2)], "per_class_ci95": pc_ci}
        rows.append(f"| {name} | {point[run][0]:.1f} | {lo:.1f}-{hi:.1f} | "
                    + ("-" if run == REF else f"{np.mean(d):+.1f} [{dlo:+.1f}, {dhi:+.1f}]") + " | "
                    + " | ".join(f"{point[run][1][i]:.1f} ({pc_ci[c][0]:.1f}-{pc_ci[c][1]:.1f})" for i, c in enumerate(cats))
                    + " |")
    save_json(out, METRIC_DIR / "diagnostics" / "bootstrap_ci.json")
    md = [f"# KITTI val 3-class AP with 95 % image-bootstrap CIs (B={B}, seed 0)", "",
          "Paired difference vs RT-DETR-L uses the same resampled images for both models; "
          "an interval containing 0 means no significant difference.", "",
          "| model | AP | 95 % CI | AP - RT-DETR-L [95 % CI] | " + " | ".join(f"{c} AP (CI)" for c in cats) + " |",
          "|---|---|---|---|" + "---|" * len(cats)] + rows
    (TABLE_DIR / "bootstrap_ci.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
