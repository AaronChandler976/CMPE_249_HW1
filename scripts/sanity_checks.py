"""The pipeline's evaluation harness: automated checks that the numbers are trustworthy.

  python scripts/sanity_checks.py           # full-run expectations
  python scripts/sanity_checks.py --quick   # smoke-test expectations

Exit code 0 = all PASS (WARNs allowed), 1 = at least one FAIL.  Writes results/sanity_report.md.
An agent (or you) should not write up results until this passes.

The key idea: before trusting a NEW measurement (COCO->KITTI), reproduce a KNOWN one.
If our COCO mAP for YOLO11s is far from the published 47.0, the pipeline is broken
(wrong category-id mapping, conf threshold too high, boxes in wrong coordinates, ...),
and every KITTI number would be wrong too.
"""
from __future__ import annotations

import argparse
import math
import sys

from common import FIG_DIR, METRIC_DIR, META_DIR, PRED_DIR, RESULTS, TABLE_DIR, load_cfg, load_json, run_name


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status: str, check: str, detail: str = ""):
        self.rows.append((status, check, detail))
        print(f"[{status}] {check} {('- ' + detail) if detail else ''}")

    def ok(self, cond: bool, check: str, detail: str = "", warn_only=False):
        self.add("PASS" if cond else ("WARN" if warn_only else "FAIL"), check, detail)

    @property
    def failed(self):
        return any(r[0] == "FAIL" for r in self.rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tol", type=float, default=2.0, help="allowed |AP - published| on COCO")
    a = ap.parse_args()
    cfg = load_cfg()
    R = Report()

    # 1. datasets
    stats_f = RESULTS / "dataset_stats.json"
    R.ok(stats_f.exists(), "dataset_stats.json exists")
    if stats_f.exists():
        s = load_json(stats_f)
        R.ok(s["n_val_images"] == 1496, "KITTI val has 1,496 images", str(s["n_val_images"]))
        R.ok(s["n_train_images"] == 5985, "KITTI train has 5,985 images", str(s["n_train_images"]))
        R.ok(len(s["cocomap_ignored"]) > 0, "KITTI->COCO mapping produced ignore regions",
             str(s["cocomap_ignored"]))

    # 2. every configured run produced predictions, metadata and metrics
    for r in cfg["runs"]:
        name = run_name(r["model"], r["dataset"], r.get("tag"))
        has_pred = (PRED_DIR / f"{name}.json").exists() and (META_DIR / f"{name}.json").exists()
        R.ok(has_pred, f"predictions+meta: {name}", warn_only=r.get("needs_finetune", False) and a.quick)
        if not has_pred:
            continue
        meta = load_json(META_DIR / f"{name}.json")
        if not a.quick:
            want = 5000 if r["dataset"] == "coco" else 1496
            R.ok(meta["n_images"] == want, f"{name} evaluated on full split", f"{meta['n_images']}/{want}")
            R.ok(meta["env"].get("cuda_available", False), f"{name} ran on GPU", meta["env"].get("gpu", "cpu"))
        R.ok(meta["thresholds"]["conf"] <= 0.01, f"{name} used a low conf threshold for mAP",
             str(meta["thresholds"]["conf"]))
        R.ok(meta["n_detections"] > 0, f"{name} produced detections", str(meta["n_detections"]))
        for tgt in r["evals"]:
            mf = METRIC_DIR / f"{name}__{tgt}.json"
            R.ok(mf.exists(), f"metrics: {name} on {tgt}")
            if not mf.exists():
                continue
            st = load_json(mf)["stats"]
            sane = all(not math.isnan(st[k]) for k in ["AP", "AP50", "AP75"]) and \
                0 <= st["AP"] <= st["AP50"] <= 100 and st["AP75"] <= st["AP50"]
            R.ok(sane, f"{name} on {tgt}: AP <= AP50, AP75 <= AP50, no NaN",
                 f"AP={st['AP']} AP50={st['AP50']} AP75={st['AP75']}")
            # 3. reproduce published COCO numbers
            ref = cfg["models"][r["model"]].get("reference_coco_map")
            if tgt == "coco" and ref is not None and not a.quick:
                d = st["AP"] - ref
                R.ok(abs(d) <= a.tol, f"{name} reproduces published COCO mAP ({ref})",
                     f"ours {st['AP']:.1f}, diff {d:+.1f}")

    # 4. expected direction of effects (WARN only: a surprise is a finding to explain, not a bug)
    def ap(run, tgt):
        f = METRIC_DIR / f"{run}__{tgt}.json"
        return load_json(f)["stats"]["AP"] if f.exists() else None
    zs, ft = ap("yolo11s__kitti", "kitti_cocomap"), ap("yolo11s_kitti_ft__kitti", "kitti_cocomap")
    if zs is not None and ft is not None and not a.quick:
        R.ok(ft > zs, "fine-tuned YOLO11s beats zero-shot on KITTI", f"{ft:.1f} vs {zs:.1f}", warn_only=True)
    lo, hi = ap("yolo11s__kitti", "kitti_cocomap"), ap("yolo11s__kitti__res1280", "kitti_cocomap")
    if lo is not None and hi is not None and not a.quick:
        R.ok(hi >= lo, "imgsz 1280 >= 640 on KITTI (small objects)", f"{hi:.1f} vs {lo:.1f}", warn_only=True)

    # 5. benchmark + artefacts
    bf = METRIC_DIR / "benchmark.json"
    R.ok(bf.exists(), "benchmark.json exists")
    if bf.exists():
        rows = load_json(bf)["rows"]
        R.ok(all(r["latency_ms_mean"] > 0 for r in rows), "all latencies > 0", f"{len(rows)} rows")
        if not a.quick:
            R.ok(all(r["peak_mem_mb"] for r in rows), "peak GPU memory recorded for every model")
    for f in ["main_results.md", "domain_shift.md", "benchmark.md"]:
        R.ok((TABLE_DIR / f).exists(), f"table {f}")
    for f in ["speed_accuracy.png", "domain_shift_per_class.png", "domain_shift_scale.png"]:
        R.ok((FIG_DIR / f).exists(), f"figure {f}")
    R.ok(any((RESULTS / "failures").glob("*/summary.json")), "failure analysis summary exists")

    n_fail = sum(r[0] == "FAIL" for r in R.rows)
    n_warn = sum(r[0] == "WARN" for r in R.rows)
    md = ["# Sanity report", "", f"Mode: {'quick' if a.quick else 'full'} | "
          f"FAIL: {n_fail} | WARN: {n_warn} | PASS: {len(R.rows) - n_fail - n_warn}", "",
          "| status | check | detail |", "|---|---|---|"]
    md += [f"| {s} | {c} | {d.replace('|', '/')} |" for s, c, d in R.rows]
    (RESULTS / "sanity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nFAIL={n_fail} WARN={n_warn} -> results/sanity_report.md")
    return 1 if R.failed else 0


if __name__ == "__main__":
    sys.exit(main())
