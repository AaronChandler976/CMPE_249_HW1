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
    python <scratchpad>/diag_fcos.py 500 fcos_r50      # diagnosis: FCOS FP16 vs FP32 x thresholds, 500 COCO imgs (not in repo)
    # fix: configs/experiments.yaml models.fcos_r50.half: false  (+ per-model `half` override in scripts/detectors.py)
    python scripts/predict.py --model fcos_r50 --dataset coco
    python scripts/predict.py --model fcos_r50 --dataset kitti
    python scripts/evaluate.py --run fcos_r50__coco --target coco
    python scripts/evaluate.py --run fcos_r50__kitti --target kitti_cocomap
    python scripts/analyze.py      # regenerate tables/figures from full-run metrics

## Stage 4 - fine-tuning
    python run_all.py --stages finetune predict_ft evaluate
