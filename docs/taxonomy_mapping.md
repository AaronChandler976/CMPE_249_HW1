# KITTI -> COCO label mapping (zero-shot evaluation)

COCO-pretrained detectors output 80 COCO classes. KITTI 2D Object has 8 classes (plus DontCare,
which the Ultralytics release removes). To score a COCO model on KITTI **without retraining** we
re-label the KITTI ground truth into COCO category ids.

| KITTI class | COCO class (id) | Role in scoring | Why |
|---|---|---|---|
| Car | car (3) | scored | same concept |
| Van | car (3) | **ignore region** (`iscrowd=1`) | COCO has no van; COCO models call vans "car" or "truck". KITTI's own devkit treats Van as a neighbouring class that is not penalised when evaluating Car. |
| Truck | truck (8) | scored | same concept |
| Pedestrian | person (1) | scored | same concept |
| Person_sitting | person (1) | **ignore region** | devkit treats it as neighbouring class of Pedestrian |
| Cyclist | person (1) | **ignore region** | COCO models detect the rider as `person` (+ a separate `bicycle`); the KITTI cyclist box covers both. Counting those `person` boxes as false positives would punish correct behaviour; counting them as true positives would reward a box that does not match the KITTI definition. |
| Tram | — | dropped | no COCO equivalent (COCO `train` is a different concept) |
| Misc | — | dropped | undefined category |

How ignore regions work (pycocotools): a detection matched to an `iscrowd=1` box of the same
category (overlap = intersection / detection-area >= IoU threshold) is neither a TP nor a FP, and the
crowd box does not count toward recall.

**Scored classes: person, car, truck.** For fair COCO-vs-KITTI comparison, COCO is also summarised on
the same three classes (`COCO_AP_3cls` in `results/tables/domain_shift.md`).

Models that predict KITTI classes (fine-tuned YOLO11s, YOLO-World with KITTI text prompts) are mapped
into the same space with `taxonomy.kitti_pred_to_coco` (car, van -> car; truck -> truck; pedestrian,
person_sitting -> person; cyclist/tram dropped), and are additionally scored on all 7 native KITTI classes.

The mapping is defined once in `configs/experiments.yaml` (`taxonomy:`) and applied in
`scripts/prepare_data.py::build_kitti_gt` (ground truth) and `scripts/evaluate.py::convert` (predictions).
