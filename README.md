# CMPE 249 HW1 — Do COCO detectors survive the road? Zero-shot COCO → KITTI evaluation

Track A (experimental). Five detector configurations from four families are evaluated on
COCO val2017 and, without retraining, on KITTI. YOLO11s is then fine-tuned on KITTI as a comparison.

| Model | Family | Framework | Checkpoint |
|---|---|---|---|
| YOLO11s | one-stage, anchor-free, NMS | Ultralytics | `yolo11s.pt` |
| RT-DETR-L | query-based transformer, NMS-free | Ultralytics | `rtdetr-l.pt` |
| Faster R-CNN R50-FPN v2 | two-stage, anchor-based | torchvision | `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` |
| FCOS R50-FPN | one-stage, anchor-free, dense | torchvision | `FCOS_ResNet50_FPN_Weights.COCO_V1` |
| YOLO-World-S v2 | open-vocabulary, text-prompted | Ultralytics | `yolov8s-worldv2.pt` |
| YOLO11s-KITTI | YOLO11s fine-tuned on KITTI train | Ultralytics | `runs/finetune/yolo11s_kitti/weights/best.pt` |

Datasets: COCO val2017 (5,000 images), and KITTI 2D Object, using the Ultralytics release of the 7,481
labelled training images split 5,985 train / 1,496 val. KITTI → COCO label mapping: `docs/taxonomy_mapping.md`.

## Reproduce
```bash
conda env create -f environment.yml        # or: conda activate CMPE249-hw1
pip install git+https://github.com/ultralytics/CLIP.git tabulate
python run_all.py --quick                  # 5-minute smoke test
python run_all.py                          # full pipeline
python scripts/sanity_checks.py            # evaluation harness: must print FAIL=0
```
Every setting (thresholds, resolutions, runs) is in `configs/experiments.yaml`.

## Where things are
- `results/tables/*.md` — every number in the report
- `results/figures/*.png` — plots
- `results/failures/<run>/` — error taxonomy + annotated worst images
- `results/run_meta/*.json` — per-run model, checkpoint, split, resolution, thresholds, hardware, versions, command
- `results/run_log.jsonl`, `commands.md` — command log
- `results/sanity_report.md` — automated checks, including reproduction of published COCO mAP

## How this was built
Pipeline code was written with AI assistance (Claude) and run and verified locally. Each published COCO
mAP is reproduced as a correctness check before any KITTI number is trusted.

## Report
`docs/report/CMPE249_HW1_Report.pdf` (source: `docs/report/report.html`). Findings with sources: `docs/findings.md`.

## Data and licences
No datasets are stored in this repo; `scripts/prepare_data.py` downloads them.
- KITTI Vision Benchmark Suite (Geiger, Lenz, Urtasun, CVPR 2012), licensed CC BY-NC-SA 3.0.
  The annotated example images in `results/failures/` are derived from KITTI images and are shared under
  the same licence, for non-commercial academic use.
- COCO (Lin et al., ECCV 2014). Annotations are CC BY 4.0; images are subject to Flickr terms.
