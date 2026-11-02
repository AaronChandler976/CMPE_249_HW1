# Key findings (draft for the report)

Every number below comes from a file in `results/` produced by this pipeline (sanity report:
109 PASS / 0 WARN / 0 FAIL). "KITTI AP" = AP@[.5:.95] on KITTI val (1,496 imgs) mapped to COCO
{person, car, truck}; "COCO 3-cls AP" = the same three classes on COCO val2017, so the two are
comparable (`docs/taxonomy_mapping.md`). Two numbers (F4, F7) come from one-off diagnostic
scripts. They are marked as such, and the scripts are in `scripts/diagnostics/`, commands in `docs/commands.md`.

---

## F1. The pipeline reproduces published COCO numbers, so the KITTI numbers can be trusted
COCO val2017 mAP, ours vs published: YOLO11s 45.6/47.0, RT-DETR-L 52.8/53.0, Faster R-CNN v2
47.0/46.7, FCOS 39.3/39.2, YOLO-World-S v2 36.8/37.7. All five are within 1.4 AP.
**Source:** `results/sanity_report.md`, `results/tables/main_results.md`.
**Why it matters:** the same id mapping, coordinates and thresholds score KITTI, so a KITTI drop
is a property of the models, not a bug. FCOS initially failed the check (31.3) because of FP16
autocast and now runs FP32; see Surprise 1.

## F2. Domain shift costs 13-24 AP, and it is class-specific: person and truck collapse, car does not
COCO 3-cls -> KITTI AP: YOLO11s 46.4 -> 24.6, RT-DETR-L 52.6 -> 29.0, Faster R-CNN 50.8 -> 28.3,
FCOS 43.4 -> 28.7, YOLO-World 41.6 -> 28.8 (drops 12.8-23.6 pts).
Per class, **car hardly moves** (YOLO11s 44.9 -> 45.3; range across models -3.6 to +6.7). **Person
drops 33-44 pts** for every model (e.g. RT-DETR 61.7 -> 20.1). **Truck drops up to 29 pts** for
the closed-set models (RT-DETR 46.1 -> 17.4).
**Source:** `results/tables/domain_shift.md`, `results/figures/domain_shift_per_class.png`.
**Interpretation:** COCO "car" and KITTI "Car" are the same concept, seen from a similar viewpoint.
"Person" and "truck" are not: see F3 (truck) and the hypotheses on person below.

## F3. Much of the truck drop comes from KITTI's Van class, a side effect of the taxonomy
Of the confident (score >= 0.3) `truck` boxes the closed-set models output on KITTI,
**34-48 % land on a KITTI Van** (YOLO11s 157/328, RT-DETR 399/1,172, Faster R-CNN 312/819;
FCOS 551/3,298 = 17 %, but FCOS outputs far more truck boxes overall). Only 6-20 % hit a real
truck. Vans are ignore regions for `car` only (pycocotools ignores per category), so every van
called "truck" is a truck false positive.
**Sensitivity check (main mapping unchanged):** making Van an ignore region for truck as well raises
truck AP for **every** zero-shot model: YOLO11s 9.0 -> 14.7, RT-DETR 17.4 -> 28.5, Faster R-CNN
18.1 -> 25.0, FCOS 19.7 -> 27.9, YOLO-World 27.7 -> 36.9 (3-class AP +1.9 to +3.7). The fine-tuned
model is unchanged (83.9): it learned van vs truck.
**Correction to an earlier hypothesis:** YOLO-World's explicit "van" prompt does *not* explain its
truck lead. It gains as much as the others (+9.2) and still leads afterwards (36.9 vs 28.5).
Its truck advantage has another, untested cause.
**Even with vans ignored, truck still drops a lot** for the closed-set models (RT-DETR 46.1 -> 28.5).
Van confusion is a large part of the truck drop, not all of it.
**Source:** truck AP from `results/tables/domain_shift.md`, sensitivity from
`results/tables/sensitivity_taxonomy.md` (`scripts/diagnostics/sensitivity_taxonomy.py`). Box
counts from `scripts/diagnostics/truck_on_van.py` over `results/predictions/*.json` + KITTI native GT.

## F4. Medium and large objects drop sharply for every model, so the shift is not mainly "KITTI objects are small"
(Corrected after review: an earlier version said large objects drop "nearly as much as small ones".)
Relative COCO 3-cls -> KITTI drop: **medium 45-60 %, large 32-47 % for every model** (e.g. YOLO11s
large 69.6 -> 36.8). Small objects vary widely: 2 % for YOLO-World (24.0 -> 23.7) up to 58 % for Faster
R-CNN (36.3 -> 15.3). Also, most KITTI GT is **medium** by COCO area
thresholds (e.g. cars 17 % small / 53 % medium / 30 % large, `results/dataset_stats.json`).
**Source:** `results/tables/domain_shift_scale.md`, `results/figures/domain_shift_scale.png`.
**Resolution ablation (YOLO11s):** imgsz 640 -> 1280 lifts AP_small 14.0 -> 21.9 and AP 24.6 -> 26.9.
AP_large stays flat (36.8 -> 36.3), and latency rises only 5.0 -> 5.6 ms (`results/tables/resolution.md`,
`benchmark.md`). Pixels help small objects. They do not fix the large-object drop, which points to
annotation/definition differences (F2, F3, F6) rather than resolution.

## F5. COCO rankings do not transfer: an 11-pt spread on COCO's 3 classes shrinks to 4.4 pts on KITTI
(Corrected after review: an earlier version compared the 80-class COCO spread, 16 pts, with 3-class
KITTI, which is exactly the unfair comparison the method rules out.)
COCO 3-cls AP ranges 41.6 (YOLO-World) to 52.6 (RT-DETR-L). KITTI AP ranges 24.6 (YOLO11s) to
29.0 (RT-DETR-L). A paired image bootstrap (B=1000) finds RT-DETR, Faster R-CNN, FCOS and YOLO-World
**statistically tied** on KITTI (all differences vs RT-DETR have 95 % CIs containing 0, e.g. YOLO-World
-0.1 [-1.7, +1.4]). Only YOLO11s is significantly worse, at -4.3 [-5.5, -3.3]. YOLO-World is last on
COCO's 3 classes but tied for best on KITTI.
**Source:** `results/tables/domain_shift.md`, `results/tables/bootstrap_ci.md`.
**Interpretation:** benchmark rank on the training distribution is a poor predictor of
out-of-domain rank. Truck AP CIs are 6-10 pts wide (223 boxes), and the 3-class mean gives truck
the same weight as car.

## F6. 25 minutes of fine-tuning nearly triples KITTI AP (24.6 -> 71.6)
YOLO11s fine-tuned 50 epochs on KITTI train: AP 24.6 -> 71.6, AP50 48.2 -> 93.0. Per class car
45.3 -> 80.9, person 19.5 -> 49.8, **truck 9.0 -> 83.9**. On all 7 native KITTI classes, AP is 67.2,
vs 15.3 for YOLO-World zero-shot with KITTI text prompts. At conf 0.3, missed GT falls 1,862 -> 484
and background FPs 1,182 -> 281.
**Source:** `results/tables/finetune.md`, `results/figures/finetune_native.png`,
`results/failures/{yolo11s__kitti,yolo11s_kitti_ft__kitti}/summary.json`,
`results/metrics/finetune/results.csv`.
**Interpretation:** the same 9.5 M-parameter network goes from worst to far best, so the zero-shot
gap is mostly *label definitions + domain* (what counts as a truck or a pedestrian, how boxes are
drawn), not model capacity. **Caveats:** train/val share 0 files, and best.pt = last epoch (no
selection gain). But KITTI's image-level split very likely places frames of the same drives in
both, so 71.6 is optimistic vs. a sequence-level split (not verified per file).

## F7. Failure modes: misses and "background" FPs dominate, and many of those FPs are real but unlabelled objects
At conf 0.3 / IoU 0.5, zero-shot YOLO11s: TP 4,998, FN 1,862, FP-background 1,182, FP-loc 760,
FP-duplicate 134, FP-class 55. Class confusion is rare. Worst recall: truck/small 1/34, person/small
0.14. Faster R-CNN logs 8,193 FP-background at the same threshold.
Visual inspection shows many "background" FPs are **real, unlabelled objects**: a pedestrian scored
0.74 in `results/failures/yolo11s__kitti/019err_000535.jpg`, and a row of parked cars behind a hedge
in `results/failures/frcnn_r50_v2__kitti/044err_000845.jpg` (KITTI's DontCare regions are missing
from the Ultralytics release). Occluded rows of parked cars are the typical miss (same image, left).
**Source:** `results/failures/*/summary.json` and images.
**Caveat:** a fixed conf 0.3 is not the same operating point for every model, because scores are
calibrated differently. Compare FP counts *within* a model (zero-shot vs fine-tuned), not across models.

## F8. Speed-accuracy: on this GPU all models are real-time, and the one-stage YOLOs dominate on KITTI
KITTI, batch 1 (median of 3 passes x 200 imgs): YOLO11s 5.0 ms, YOLO-World 5.4, RT-DETR-L 12.2,
Faster R-CNN 16.4, FCOS 18.8 (FP32, see below). Peak memory 72-173 MB for the 640-px models vs
391-488 MB for the torchvision R50-FPN models (~800 px input).
YOLO-World gives about RT-DETR's KITTI AP (28.8 vs 29.0) at less than half the latency.
**Source:** `results/tables/benchmark.md`, `results/figures/speed_accuracy.png` (COCO).
**Caveats:** FCOS runs FP32 (Surprise 1) and the others FP16, so its latency is not like-for-like.
COCO latencies vary between passes by up to 5 ms on Windows/WDDM (RT-DETR 13.7-18.8), while KITTI
passes agree within 0.5 ms, so quote KITTI latencies. Pipeline differences (Ultralytics vs
torchvision preprocessing, input size) are part of what is measured.

---

## Surprises, with the verification experiments run on 2026-11-01
Raw outputs: `results/metrics/diagnostics/*.txt`, `results/tables/sensitivity_taxonomy.md`.

1. **FP16 cost FCOS 7.9 AP on COCO (31.3 vs 39.3) but ~0 on KITTI (28.6 vs 28.7). CONFIRMED:
   the loss is entirely outside person/car/truck.** Full COCO val, FP16 vs FP32: person 53.4/53.5,
   car 42.2/42.4, truck 34.3/34.3, while the mean AP of the other 77 classes is **30.8 vs 39.2**
   (-8.4). Thresholds change < 0.3 AP. KITTI scores only the three unaffected classes, so it did not
   move. *Still untested:* why only those classes suffer. A plausible guess is that half precision
   reorders near-tied low scores, which matter most for classes with weaker evidence.
   (`python scripts/diagnostics/diag_fcos.py 5000 fcos_r50 --per-class`)
2. **Car AP is *higher* on KITTI than COCO for FCOS (+4.0) and YOLO-World (+6.7). MOSTLY NOT
   explained by ignoring vans.** Scoring Van as car lowers KITTI car AP by only 0.6-1.8 (FCOS
   46.4 -> 45.3, YOLO-World 45.9 -> 45.3), so both stay above their COCO car AP (42.4, 39.2).
   *Remaining hypothesis (untested):* KITTI cars are canonical (rear/side views, road context,
   little clutter), while COCO "car" covers odd views, crowds and partial cars.
   (`results/tables/sensitivity_taxonomy.md`)
3. **YOLO-World's worst-in-class person AP (12.8) is partly the prompt word. CONFIRMED, partly.**
   Same weights, prompt "person" instead of "pedestrian": person AP 12.8 -> **17.5** (AP50 32.9 ->
   45.7), 3-class AP 28.8 -> 30.1. Removing the competing "person sitting"/"cyclist" prompts adds
   only +0.5 more (18.0). It still trails the closed-set models (~20), and the pipeline keeps the
   original prompts (changing them after seeing results would be tuning on the test set).
   (`python scripts/diagnostics/yoloworld_prompts.py`)
4. **The first benchmark reported YOLO11s at 56.5 ms on COCO, 10x its real cost.** This was a
   one-time setup cost for each new input shape (9/200 imgs up to 1,046 ms, gone on a second
   pass), fixed with warmup = 200 + 3 repeats. Worth one sentence in the report's method section:
   latency measurement needs a protocol, not one loop.

## Limitations to state
No DontCare regions (F7). COCO's absolute area thresholds are applied to 1242x375 images (F4).
Truck AP rests on 223 GT boxes. One train/val split, likely correlated by drive (F6).
Windows/WDDM latency is noisy and quoted as median of 3 (F8). FCOS FP32 vs others FP16.
