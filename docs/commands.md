    git init
    conda env export --no-builds > environment.yml
    pip freeze > requirements.txt
    nvidia-smi > results\hardware.txt
## Stage 0 - setup (2026-11-01)
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True NVIDIA GeForce RTX 5070 Ti
    pip install git+https://github.com/ultralytics/CLIP.git tabulate
    gh repo create CMPE_249_HW1 --private --source=. --remote=origin
    pip install pyyaml        # ultralytics 8.4.197 no longer pulls PyYAML in; scripts/common.py needs it

## Stage 1 - smoke test
    python run_all.py --quick

## Stage 2 - data
    python run_all.py --stages prepare
    rm -rf results/predictions results/run_meta results/metrics results/failures runs/finetune   # (run by Aaron) clear smoke-test artefacts

## Stage 3 - zero-shot predictions + evaluation
    python run_all.py --stages predict evaluate
    python scripts/diagnostics/diag_fcos.py 500 fcos_r50      # diagnosis: FCOS FP16 vs FP32 x thresholds, 500 COCO imgs
    # fix: configs/experiments.yaml models.fcos_r50.half: false  (+ per-model `half` override in scripts/detectors.py)
    python scripts/predict.py --model fcos_r50 --dataset coco
    python scripts/predict.py --model fcos_r50 --dataset kitti
    python scripts/evaluate.py --run fcos_r50__coco --target coco
    python scripts/evaluate.py --run fcos_r50__kitti --target kitti_cocomap
    python scripts/analyze.py      # regenerate tables/figures from full-run metrics

## Stage 4 - fine-tuning
    python run_all.py --stages finetune predict_ft evaluate

## Stage 5 - benchmark, failures, tables, sanity
    python run_all.py --stages benchmark failures analyze sanity
    python scripts/diagnostics/diag_bench.py yolo11s   # diagnosis: per-shape first-call cost
    # fix: configs/experiments.yaml benchmark.warmup 20 -> 200
    python run_all.py --stages benchmark analyze sanity
    # fix: benchmark.repeats 3 (median of pass means) - scripts/benchmark.py
    python run_all.py --stages benchmark analyze sanity
    python scripts/analyze.py   # + pass_range_ms column

## Stage 6 - findings
    python scripts/diagnostics/truck_on_van.py   # diagnostic: confident truck boxes landing on KITTI vans
    # wrote docs/findings.md

## Stage 7 - verification / sensitivity experiments (main results unchanged)
    python scripts/diagnostics/sensitivity_taxonomy.py                  # Van ignored for truck too / Van scored as car
    python scripts/diagnostics/diag_fcos.py 5000 fcos_r50 --per-class   # FP16 loss by class, full COCO
    python scripts/diagnostics/yoloworld_prompts.py                     # "pedestrian" vs "person" prompt
    # outputs saved to results/metrics/diagnostics/*.txt and results/tables/sensitivity_taxonomy.md

## Report
    # docs/report/report.html -> PDF via headless Edge:
    msedge --headless=new --no-pdf-header-footer --print-to-pdf=docs/report/report.pdf docs/report/report.html
    python scripts/diagnostics/bootstrap_ci.py 1000      # 95% image-bootstrap CIs + paired diffs vs RT-DETR-L
    python scripts/diagnostics/diag_bench.py yolo11s     # re-run, saved to results/metrics/diagnostics/bench_warmup.txt
    # report revised after rubric-reviewer pass; PDF re-printed with headless Edge (see Report above)
    python scripts/analyze.py                            # adds results/figures/finetune_curve.png (Fig. 4b)
    # second revision after a first-pass review against the assignment (10 fixes); PDF re-printed, 9 pages
    # repo cleanup: README rewrite, MIT LICENSE, assistant tooling and study notes untracked, repo renamed to zero-shot-detection-transfer
