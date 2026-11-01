# Sanity report

Mode: full | FAIL: 16 | WARN: 1 | PASS: 91

| status | check | detail |
|---|---|---|
| PASS | dataset_stats.json exists |  |
| PASS | KITTI val has 1,496 images | 1496 |
| PASS | KITTI train has 5,985 images | 5985 |
| PASS | KITTI->COCO mapping produced ignore regions | {'cyclist->person': 352, 'person_sitting->person': 36, 'van->car': 555} |
| PASS | predictions+meta: yolo11s__coco |  |
| FAIL | yolo11s__coco evaluated on full split | 25/5000 |
| PASS | yolo11s__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s__coco used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s__coco produced detections | 2580 |
| PASS | metrics: yolo11s__coco on coco |  |
| PASS | yolo11s__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=57.6 AP50=77.37 AP75=58.71 |
| FAIL | yolo11s__coco reproduces published COCO mAP (47.0) | ours 57.6, diff +10.6 |
| PASS | predictions+meta: rtdetr_l__coco |  |
| FAIL | rtdetr_l__coco evaluated on full split | 25/5000 |
| PASS | rtdetr_l__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | rtdetr_l__coco used a low conf threshold for mAP | 0.001 |
| PASS | rtdetr_l__coco produced detections | 7500 |
| PASS | metrics: rtdetr_l__coco on coco |  |
| PASS | rtdetr_l__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=64.67 AP50=78.57 AP75=69.66 |
| FAIL | rtdetr_l__coco reproduces published COCO mAP (53.0) | ours 64.7, diff +11.7 |
| PASS | predictions+meta: frcnn_r50_v2__coco |  |
| FAIL | frcnn_r50_v2__coco evaluated on full split | 25/5000 |
| PASS | frcnn_r50_v2__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | frcnn_r50_v2__coco used a low conf threshold for mAP | 0.001 |
| PASS | frcnn_r50_v2__coco produced detections | 4717 |
| PASS | metrics: frcnn_r50_v2__coco on coco |  |
| PASS | frcnn_r50_v2__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=60.86 AP50=81.75 AP75=63.3 |
| FAIL | frcnn_r50_v2__coco reproduces published COCO mAP (46.7) | ours 60.9, diff +14.2 |
| PASS | predictions+meta: fcos_r50__coco |  |
| FAIL | fcos_r50__coco evaluated on full split | 25/5000 |
| PASS | fcos_r50__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | fcos_r50__coco used a low conf threshold for mAP | 0.001 |
| PASS | fcos_r50__coco produced detections | 7500 |
| PASS | metrics: fcos_r50__coco on coco |  |
| PASS | fcos_r50__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=47.05 AP50=63.52 AP75=51.51 |
| FAIL | fcos_r50__coco reproduces published COCO mAP (39.2) | ours 47.0, diff +7.8 |
| PASS | predictions+meta: yoloworld_s__coco |  |
| FAIL | yoloworld_s__coco evaluated on full split | 25/5000 |
| PASS | yoloworld_s__coco ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yoloworld_s__coco used a low conf threshold for mAP | 0.001 |
| PASS | yoloworld_s__coco produced detections | 3401 |
| PASS | metrics: yoloworld_s__coco on coco |  |
| PASS | yoloworld_s__coco on coco: AP <= AP50, AP75 <= AP50, no NaN | AP=53.15 AP50=69.46 AP75=57.9 |
| PASS | predictions+meta: yolo11s__kitti |  |
| FAIL | yolo11s__kitti evaluated on full split | 25/1496 |
| PASS | yolo11s__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s__kitti used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s__kitti produced detections | 2228 |
| PASS | metrics: yolo11s__kitti on kitti_cocomap |  |
| PASS | yolo11s__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=33.94 AP50=53.74 AP75=44.04 |
| PASS | predictions+meta: rtdetr_l__kitti |  |
| FAIL | rtdetr_l__kitti evaluated on full split | 25/1496 |
| PASS | rtdetr_l__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | rtdetr_l__kitti used a low conf threshold for mAP | 0.001 |
| PASS | rtdetr_l__kitti produced detections | 7500 |
| PASS | metrics: rtdetr_l__kitti on kitti_cocomap |  |
| PASS | rtdetr_l__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=35.43 AP50=58.27 AP75=44.45 |
| PASS | predictions+meta: frcnn_r50_v2__kitti |  |
| FAIL | frcnn_r50_v2__kitti evaluated on full split | 25/1496 |
| PASS | frcnn_r50_v2__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | frcnn_r50_v2__kitti used a low conf threshold for mAP | 0.001 |
| PASS | frcnn_r50_v2__kitti produced detections | 4388 |
| PASS | metrics: frcnn_r50_v2__kitti on kitti_cocomap |  |
| PASS | frcnn_r50_v2__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=37.7 AP50=59.19 AP75=49.15 |
| PASS | predictions+meta: fcos_r50__kitti |  |
| FAIL | fcos_r50__kitti evaluated on full split | 25/1496 |
| PASS | fcos_r50__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | fcos_r50__kitti used a low conf threshold for mAP | 0.001 |
| PASS | fcos_r50__kitti produced detections | 7500 |
| PASS | metrics: fcos_r50__kitti on kitti_cocomap |  |
| PASS | fcos_r50__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=38.47 AP50=64.0 AP75=45.8 |
| PASS | predictions+meta: yoloworld_s__kitti |  |
| FAIL | yoloworld_s__kitti evaluated on full split | 25/1496 |
| PASS | yoloworld_s__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yoloworld_s__kitti used a low conf threshold for mAP | 0.001 |
| PASS | yoloworld_s__kitti produced detections | 1502 |
| PASS | metrics: yoloworld_s__kitti on kitti_cocomap |  |
| PASS | yoloworld_s__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=41.55 AP50=61.41 AP75=46.58 |
| PASS | metrics: yoloworld_s__kitti on kitti_native |  |
| PASS | yoloworld_s__kitti on kitti_native: AP <= AP50, AP75 <= AP50, no NaN | AP=24.01 AP50=34.34 AP75=27.69 |
| PASS | predictions+meta: yolo11s__kitti__res1280 |  |
| FAIL | yolo11s__kitti__res1280 evaluated on full split | 25/1496 |
| PASS | yolo11s__kitti__res1280 ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s__kitti__res1280 used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s__kitti__res1280 produced detections | 3918 |
| PASS | metrics: yolo11s__kitti__res1280 on kitti_cocomap |  |
| PASS | yolo11s__kitti__res1280 on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=41.05 AP50=63.7 AP75=54.15 |
| PASS | predictions+meta: yolo11s_kitti_ft__kitti |  |
| FAIL | yolo11s_kitti_ft__kitti evaluated on full split | 25/1496 |
| PASS | yolo11s_kitti_ft__kitti ran on GPU | NVIDIA GeForce RTX 5070 Ti |
| PASS | yolo11s_kitti_ft__kitti used a low conf threshold for mAP | 0.001 |
| PASS | yolo11s_kitti_ft__kitti produced detections | 7500 |
| PASS | metrics: yolo11s_kitti_ft__kitti on kitti_cocomap |  |
| PASS | yolo11s_kitti_ft__kitti on kitti_cocomap: AP <= AP50, AP75 <= AP50, no NaN | AP=12.68 AP50=21.28 AP75=13.26 |
| PASS | metrics: yolo11s_kitti_ft__kitti on kitti_native |  |
| PASS | yolo11s_kitti_ft__kitti on kitti_native: AP <= AP50, AP75 <= AP50, no NaN | AP=6.35 AP50=10.72 AP75=6.65 |
| WARN | fine-tuned YOLO11s beats zero-shot on KITTI | 12.7 vs 33.9 |
| PASS | imgsz 1280 >= 640 on KITTI (small objects) | 41.0 vs 33.9 |
| PASS | benchmark.json exists |  |
| PASS | all latencies > 0 | 11 rows |
| PASS | peak GPU memory recorded for every model |  |
| PASS | table main_results.md |  |
| PASS | table domain_shift.md |  |
| PASS | table benchmark.md |  |
| PASS | figure speed_accuracy.png |  |
| PASS | figure domain_shift_per_class.png |  |
| PASS | figure domain_shift_scale.png |  |
| PASS | failure analysis summary exists |  |
