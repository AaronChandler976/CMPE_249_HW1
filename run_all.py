"""One command for the whole experiment pipeline.

  python run_all.py --quick                     # ~5 min smoke test (25 images, 1-epoch fine-tune)
  python run_all.py                             # full run (~2-3 h on an RTX 5070 Ti, mostly fine-tuning)
  python run_all.py --stages predict evaluate   # re-run selected stages
  python run_all.py --skip-existing             # resume after a crash without redoing finished runs

Stages: prepare predict finetune predict_ft evaluate benchmark failures analyze sanity
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "scripts")

ALL = ["prepare", "predict", "finetune", "predict_ft", "evaluate", "benchmark", "failures",
       "analyze", "sanity"]


def main() -> int:
    from common import META_DIR, METRIC_DIR, load_cfg, log_event, p, run_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default=None, help="override configs device, e.g. cpu")
    a = ap.parse_args()
    if a.device:
        import os
        os.environ["DETBENCH_DEVICE"] = a.device
    cfg = load_cfg()
    limit = cfg["quick"]["n_images"] if a.quick else None
    t0 = time.time()
    log_event("run_all_start", quick=a.quick, stages=a.stages)

    import benchmark, evaluate, failure_analysis, predict, prepare_data, sanity_checks  # noqa: E401

    if "prepare" in a.stages:
        sys.argv = ["prepare_data.py"] + (["--quick"] if a.quick else [])
        prepare_data.main()

    def do_predict(r):
        name = run_name(r["model"], r["dataset"], r.get("tag"))
        if a.skip_existing and (META_DIR / f"{name}.json").exists():
            print(f"[skip] {name}")
            return
        try:
            predict.run(r["model"], r["dataset"], r.get("imgsz"), r.get("tag"), limit, cfg)
        except Exception as e:  # keep going; sanity_checks will FAIL on the missing run
            import traceback
            traceback.print_exc()
            log_event("ERROR", run=name, error=repr(e))
            print(f"[ERROR] {name} failed: {e!r} - continuing with the next run")

    if "predict" in a.stages:
        for r in cfg["runs"]:
            if not r.get("needs_finetune"):
                do_predict(r)

    ft_weights = p(cfg["models"]["yolo11s_kitti_ft"]["weights"])
    if "finetune" in a.stages and not (a.skip_existing and ft_weights.exists()):
        import finetune
        sys.argv = ["finetune.py"] + (["--quick"] if a.quick else [])
        finetune.main()

    if "predict_ft" in a.stages:
        for r in cfg["runs"]:
            if r.get("needs_finetune"):
                if ft_weights.exists():
                    do_predict(r)
                else:
                    print(f"[warn] {ft_weights} missing - run the finetune stage first")

    if "evaluate" in a.stages:
        for r in cfg["runs"]:
            name = run_name(r["model"], r["dataset"], r.get("tag"))
            if not (META_DIR / f"{name}.json").exists():
                continue
            for tgt in r["evals"]:
                try:
                    evaluate.run(name, tgt, cfg)
                except Exception as e:
                    log_event("ERROR", run=name, target=tgt, error=repr(e))
                    print(f"[ERROR] eval {name} on {tgt}: {e!r}")

    if "benchmark" in a.stages and not (a.skip_existing and (METRIC_DIR / "benchmark.json").exists()):
        q = cfg["quick"]
        benchmark.main(["--n", str(q["benchmark_images"]), "--warmup", str(q["benchmark_warmup"])]
                       if a.quick else [])

    if "failures" in a.stages:
        for name in ["yolo11s__kitti", "rtdetr_l__kitti", "frcnn_r50_v2__kitti", "yolo11s_kitti_ft__kitti"]:
            if (META_DIR / f"{name}.json").exists():
                failure_analysis.analyze(name, "kitti_cocomap", 8, cfg["vis_thresholds"]["conf"], cfg)

    if "analyze" in a.stages:
        import analyze
        analyze.main()

    code = 0
    if "sanity" in a.stages:
        sys.argv = ["sanity_checks.py"] + (["--quick"] if a.quick else [])
        code = sanity_checks.main()

    log_event("run_all_done", minutes=round((time.time() - t0) / 60, 1), sanity_exit=code)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Sanity exit code: {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
