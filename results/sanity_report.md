# Sanity report

Mode: full | FAIL: 2 | WARN: 0 | PASS: 105

| status | check | detail |
|---|---|---|
| PASS | dataset_stats.json exists |  |
| PASS | KITTI val has 1,496 images | 1496 |
| PASS | KITTI train has 5,985 images | 5985 |
| PASS | KITTI->COCO mapping produced ignore regions | {'cyclist->person': 352, 'person_sitting->person': 36, 'van->car': 555} |
| PASS | predictions+meta: yolo11s__coco |  |
| PASS | yolo11s__coco evaluated on full split | 5000/5000 |
| PASS | yolo11s__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s__coco used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s__coco produced detections | 536331 |
| PASS | metrics: yolo11s__coco on coco |  |
| PASS | yolo11s__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=45.63 AP50=62.29 AP75=49.3 |
| PASS | yolo11s__coco reproduces published COCO mAP (47.0) | ours 45.6, diff -1.4 |
| PASS | predictions+meta: rtdetr_l__coco |  |
| PASS | rtdetr_l__coco evaluated on full split | 5000/5000 |
| PASS | rtdetr_l__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | rtdetr_l__coco used a low conf threshold for mAP | 0.001 |
| PASS | rtdetr_l__coco produced detections | 1500000 |
| PASS | metrics: rtdetr_l__coco on coco |  |
| PASS | rtdetr_l__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=52.75 AP50=71.23 AP75=57.11 |
| PASS | rtdetr_l__coco reproduces published COCO mAP (53.0) | ours 52.8, diff -0.2 |
| PASS | predictions+meta: frcnn_r50_v2__coco |  |
| PASS | frcnn_r50_v2__coco evaluated on full split | 5000/5000 |
| PASS | frcnn_r50_v2__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | frcnn_r50_v2__coco used a low conf threshold for mAP | 0.001 |
| PASS | frcnn_r50_v2__coco produced detections | 955624 |
| PASS | metrics: frcnn_r50_v2__coco on coco |  |
| PASS | frcnn_r50_v2__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=47.01 AP50=68.14 AP75=51.27 |
| PASS | frcnn_r50_v2__coco reproduces published COCO mAP (46.7) | ours 47.0, diff +0.3 |
| PASS | predictions+meta: fcos_r50__coco |  |
| PASS | fcos_r50__coco evaluated on full split | 5000/5000 |
| PASS | fcos_r50__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | fcos_r50__coco used a low conf threshold for mAP | 0.001 |
| PASS | fcos_r50__coco produced detections | 1500000 |
| PASS | metrics: fcos_r50__coco on coco |  |
| PASS | fcos_r50__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=39.33 AP50=58.66 AP75=42.14 |
| PASS | fcos_r50__coco reproduces published COCO mAP (39.2) | ours 39.3, diff +0.1 |
| PASS | predictions+meta: yoloworld_s__coco |  |
| PASS | yoloworld_s__coco evaluated on full split | 5000/5000 |
| PASS | yoloworld_s__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yoloworld_s__coco used a low conf threshold for mAP | 0.001 |
| PASS | yoloworld_s__coco produced detections | 745608 |
| PASS | metrics: yoloworld_s__coco on coco |  |
| PASS | yoloworld_s__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=36.77 AP50=51.29 AP75=39.83 |
| PASS | yoloworld_s__coco reproduces published COCO mAP (37.7) | ours 36.8, diff -0.9 |
| PASS | predictions+meta: yolo11s__kitti |  |
| PASS | yolo11s__kitti evaluated on full split | 1496/1496 |
| PASS | yolo11s__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s__kitti used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s__kitti produced detections | 156642 |
| PASS | metrics: yolo11s__kitti on kitti_cocomap |  |
| PASS | yolo11s__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=24.59 AP50=48.25 AP75=22.54 |
| PASS | predictions+meta: rtdetr_l__kitti |  |
| PASS | rtdetr_l__kitti evaluated on full split | 1496/1496 |
| PASS | rtdetr_l__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | rtdetr_l__kitti used a low conf threshold for mAP | 0.001 |
| PASS | rtdetr_l__kitti produced detections | 448800 |
| PASS | metrics: rtdetr_l__kitti on kitti_cocomap |  |
| PASS | rtdetr_l__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=28.99 AP50=53.76 AP75=27.39 |
| PASS | predictions+meta: frcnn_r50_v2__kitti |  |
| PASS | frcnn_r50_v2__kitti evaluated on full split | 1496/1496 |
| PASS | frcnn_r50_v2__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | frcnn_r50_v2__kitti used a low conf threshold for mAP | 0.001 |
| PASS | frcnn_r50_v2__kitti produced detections | 281842 |
| PASS | metrics: frcnn_r50_v2__kitti on kitti_cocomap |  |
| PASS | frcnn_r50_v2__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=28.32 AP50=54.42 AP75=27.34 |
| PASS | predictions+meta: fcos_r50__kitti |  |
| PASS | fcos_r50__kitti evaluated on full split | 1496/1496 |
| PASS | fcos_r50__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | fcos_r50__kitti used a low conf threshold for mAP | 0.001 |
| PASS | fcos_r50__kitti produced detections | 448800 |
| PASS | metrics: fcos_r50__kitti on kitti_cocomap |  |
| PASS | fcos_r50__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=28.73 AP50=54.53 AP75=27.37 |
| PASS | predictions+meta: yoloworld_s__kitti |  |
| PASS | yoloworld_s__kitti evaluated on full split | 1496/1496 |
| PASS | yoloworld_s__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yoloworld_s__kitti used a low conf threshold for mAP | 0.001 |
| PASS | yoloworld_s__kitti produced detections | 104153 |
| PASS | metrics: yoloworld_s__kitti on kitti_cocomap |  |
| PASS | yoloworld_s__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=28.8 AP50=49.97 AP75=29.5 |
| PASS | metrics: yoloworld_s__kitti on kitti_native |  |
| PASS | yoloworld_s__kitti on kitti_native: AP <= AP50, AP75 <= AP50, no NaN | AP=15.35 AP50=26.89 AP75=15.6 |
| PASS | predictions+meta: yolo11s__kitti__res1280 |  |
| PASS | yolo11s__kitti__res1280 evaluated on full split | 1496/1496 |
| PASS | yolo11s__kitti__res1280 ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s__kitti__res1280 used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s__kitti__res1280 produced detections | 241441 |
| PASS | metrics: yolo11s__kitti__res1280 on kitti_cocomap |  |
| PASS | yolo11s__kitti__res1280 on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=26.86 AP50=50.98 AP75=25.09 |
| PASS | predictions+meta: yolo11s_kitti_ft__kitti |  |
| PASS | yolo11s_kitti_ft__kitti evaluated on full split | 1496/1496 |
| PASS | yolo11s_kitti_ft__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s_kitti_ft__kitti used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s_kitti_ft__kitti produced detections | 36225 |
| PASS | metrics: yolo11s_kitti_ft__kitti on kitti_cocomap |  |
| PASS | yolo11s_kitti_ft__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=71.56 AP50=93.03 AP75=80.25 |
| PASS | metrics: yolo11s_kitti_ft__kitti on kitti_native |  |
| PASS | yolo11s_kitti_ft__kitti on kitti_native: AP <= AP50, AP75 <= AP50, no NaN | AP=67.16 AP50=90.31 AP75=75.53 |
| PASS | fine-tuned YOLO11s beats zero-shot on KITTI | 71.6 vs 24.6 |
| PASS | imgsz 1280 >= 640 on KITTI (small objects) | 26.9 vs 24.6 |
| FAIL | benchmark.json exists |  |
| PASS | table main_results.md |  |
| PASS | table domain_shift.md |  |
| PASS | table benchmark.md |  |
| PASS | figure speed_accuracy.png |  |
| PASS | figure domain_shift_per_class.png |  |
| PASS | figure domain_shift_scale.png |  |
| FAIL | failure analysis summary exists |  |
