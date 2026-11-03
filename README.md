# Zero-Shot Detection Transfer: COCO → KITTI

How much of a COCO-trained object detector's accuracy survives on a driving dataset, which classes and
object sizes break, and is the fix a better architecture or in-domain data?

Five COCO-pretrained detectors from four families are evaluated on COCO val2017 and, **without
retraining**, on KITTI through a documented label mapping. A YOLO11s fine-tuned on KITTI is the
in-domain reference. Every published COCO mAP is first reproduced to within 1.4 AP, so the KITTI
numbers rest on a validated evaluation harness.

**Report:** [`docs/report/report.pdf`](docs/report/report.pdf)

## Key results

KITTI val (1,496 images), AP@[.5:.95] on the shared classes person / car / truck. Latency is batch 1 on an
RTX 5070 Ti.

| Model | Family | COCO 3-class AP | KITTI AP (95% CI) | KITTI latency |
|---|---|---:|---:|---:|
| YOLO11s | one-stage, anchor-free | 46.4 | 24.6 (23.6–26.4) | 5.0 ms |
| RT-DETR-L | DETR-style, NMS-free | 52.6 | 29.0 (27.5–31.1) | 12.2 ms |
| Faster R-CNN R50-FPN v2 | two-stage | 50.8 | 28.3 (26.9–30.3) | 16.4 ms |
| FCOS R50-FPN | one-stage, dense | 43.4 | 28.7 (27.1–30.7) | 18.8 ms |
| YOLO-World-S v2 | open-vocabulary | 41.6 | 28.8 (27.4–31.0) | 5.4 ms |
| YOLO11s fine-tuned on KITTI | one-stage | – | **71.6** (70.3–72.9) | – |

- **Transfer is class-specific.** Car AP barely moves, while person drops 33–44 points for every model.
- **Architecture matters less than data.** The 11-point spread between models on COCO shrinks to 4.4 on
  KITTI, where three of the other four models are not significantly different from RT-DETR. Twenty-five minutes of fine-tuning adds 47 AP.
- **The gap is not mainly about small objects.** Medium and large objects lose 32–60% of their AP.

Sources: `results/tables/main_results.md`, `results/tables/bootstrap_ci.md`, `results/tables/benchmark.md`.

## Repository layout

```
├── configs/experiments.yaml   # single source of truth: models, thresholds, label mapping, runs
├── run_all.py                 # runs the pipeline stages in order
├── scripts/
│   ├── prepare_data.py        # downloads COCO val2017 + KITTI, builds COCO-format KITTI ground truth
│   ├── detectors.py           # uniform wrapper around Ultralytics and torchvision detectors
│   ├── predict.py             # inference -> COCO-format detections
│   ├── evaluate.py            # pycocotools evaluation (COCO, KITTI mapped, KITTI native)
│   ├── finetune.py            # YOLO11s fine-tuning on KITTI
│   ├── benchmark.py           # latency and GPU memory
│   ├── failure_analysis.py    # TIDE-style error breakdown + annotated examples
│   ├── analyze.py             # tables and figures
│   ├── sanity_checks.py       # automated checks that gate every result
│   └── diagnostics/           # targeted verification experiments cited in the report
├── results/
│   ├── tables/                # every number in the report (Markdown + CSV)
│   ├── figures/               # report figures
│   ├── metrics/               # raw evaluation output per run
│   ├── failures/              # error taxonomy and annotated images per run
│   ├── run_meta/              # per-run checkpoint, split, thresholds, hardware, versions, command
│   └── sanity_report.md
└── docs/
    ├── report/                # report.pdf and its HTML source
    ├── findings.md            # findings with the files that support them
    ├── taxonomy_mapping.md    # KITTI -> COCO label mapping and rationale
    └── commands.md            # command log
```

## Reproduce

Requires an NVIDIA GPU with CUDA 12.8+ (tested on Windows 11, RTX 5070 Ti).

```bash
conda env create -f environment.yml
conda activate detection-transfer
pip install git+https://github.com/ultralytics/CLIP.git tabulate

python run_all.py --quick          # ~5-minute smoke test on a few images
python run_all.py                  # full pipeline (~1-1.5 h): data, inference, fine-tuning, evaluation
python scripts/sanity_checks.py    # must report FAIL=0
```

Add `--skip-existing` to resume after an interruption, or `--stages <names>` to run part of the pipeline
(`prepare predict finetune predict_ft evaluate benchmark failures analyze sanity`).

## Method notes

- **Label mapping.** Car → car, Truck → truck, Pedestrian → person are scored. Van, Person_sitting and
  Cyclist are ignore regions (`iscrowd=1`), following the KITTI devkit's neighbouring-class convention.
  Tram and Misc are dropped. See [`docs/taxonomy_mapping.md`](docs/taxonomy_mapping.md).
- **Fair comparison.** COCO → KITTI comparisons use the same three classes on both sides, never the
  80-class COCO mAP.
- **Evaluation.** pycocotools, conf = 0.001 and at most 300 detections per image for every model. NMS IoU
  is each framework's default: 0.7 for the Ultralytics models (YOLO11s, YOLO-World, fine-tuned YOLO11s),
  0.5 for Faster R-CNN's final per-class NMS (its RPN applies 0.7 to proposals), and 0.6 for FCOS.
  RT-DETR uses no NMS.

## Data and licences

Code is released under the [MIT License](LICENSE). No datasets are stored in this repository;
`scripts/prepare_data.py` downloads them.

- **KITTI** Vision Benchmark Suite (Geiger, Lenz, Urtasun, CVPR 2012) is licensed CC BY-NC-SA 3.0. The
  annotated example images in `results/failures/` are derived from KITTI and shared under the same
  licence, for non-commercial academic use.
- **COCO** (Lin et al., ECCV 2014): annotations are CC BY 4.0; images are subject to Flickr terms.

Developed as a course project for CMPE 249 at San José State University.
