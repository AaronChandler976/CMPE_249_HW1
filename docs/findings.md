# Key findings (draft for the report)

Every number below comes from a file in `results/` produced by this pipeline (sanity report:
109 PASS / 0 WARN / 0 FAIL). "KITTI AP" = AP@[.5:.95] on KITTI val (1,496 imgs) mapped to COCO
{person, car, truck}; "COCO 3-cls AP" = the same three classes on COCO val2017, so the two are
comparable (`docs/taxonomy_mapping.md`). Two numbers (F4, F7) come from one-off diagnostic
scripts. They are marked as such, and the scripts are in `scripts/diagnostics/`, commands in `commands.md`.

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
FCOS 551/3,298 = 17 %, but FCOS outputs far more truck boxes overall). Only 6-20 % hit a real truck. Vans are ignore regions for `car` only (pycocotools ignores per
category), so every van called "truck" is a truck false positive. YOLO-World, prompted with an
explicit "van" class, puts 68 boxes on vans and has the best truck precision (99/220 = 45 %) and the
best KITTI truck AP (27.7, almost no drop from 29.2 on COCO).
**Source:** truck AP from `results/tables/domain_shift.md`. Box counts from the diagnostic
`scripts/diagnostics/truck_on_van.py` over `results/predictions/*.json` + KITTI native GT.
**Open decision for Aaron (taxonomy change, so not applied):** also marking Van as an ignore region
for `truck` would measure "truck skill" without the van ambiguity. Report it as a sensitivity check
if we run it; it must not silently replace the main mapping.

## F4. Drops happen at every object scale, so the shift is not only "KITTI objects are small"
COCO 3-cls -> KITTI AP for **large** objects drops nearly as much as for small ones: YOLO11s 69.6 ->
36.8, RT-DETR 75.4 -> 40.9, Faster R-CNN 67.4 -> 40.0. Small objects drop 27.9 -> 14.0 (YOLO11s) but
barely at all for YOLO-World (24.0 -> 23.7). Also, most KITTI GT is **medium** by COCO area
thresholds (e.g. cars 17 % small / 53 % medium / 30 % large, `results/dataset_stats.json`).
**Source:** `results/tables/domain_shift_scale.md`, `results/figures/domain_shift_scale.png`.
**Resolution ablation (YOLO11s):** imgsz 640 -> 1280 lifts AP_small 14.0 -> 21.9 and AP 24.6 -> 26.9.
AP_large stays flat (36.8 -> 36.3), and latency rises only 5.0 -> 5.6 ms (`results/tables/resolution.md`,
`benchmark.md`). Pixels help small objects. They do not fix the large-object drop, which points to
annotation/definition differences (F2, F3, F6) rather than resolution.

## F5. COCO rankings do not transfer: a 16-pt spread on COCO shrinks to 4.4 pts on KITTI
COCO 80-cls AP ranges 36.8 (YOLO-World) to 52.8 (RT-DETR-L). KITTI AP ranges 24.6 (YOLO11s) to
29.0 (RT-DETR-L). YOLO-World is **last on COCO but ties for best on KITTI** (28.8) and has the
smallest drop (12.8). YOLO11s is mid-pack on COCO but last on KITTI.
**Source:** `results/tables/domain_shift.md`, `main_results.md`.
**Interpretation:** benchmark rank on the training distribution is a poor predictor of
out-of-domain rank. Differences of 1-2 KITTI AP between models (one val split, truck AP from only
223 boxes) should not be over-read. With car at 83 % of the boxes, the 3-class mean gives truck
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

## Surprises, with explanations you can verify

1. **FP16 cost FCOS 7.9 AP on COCO (31.3 vs 39.3) but ~0 on KITTI (28.6 vs 28.7).** On 500 COCO
   images FP16 vs FP32 was 36.2 vs 43.1, and thresholds changed < 0.3 AP. *Hypothesis:* the damage
   is concentrated in the other 77 COCO classes. FCOS scores are sqrt(cls x centerness), and
   half precision may reorder the many near-tied low scores. *Verify:* run
   `python scripts/diagnostics/diag_fcos.py 500`, extended to print per-class AP, and compare
   person/car/truck vs the other 77 classes for FP16 vs FP32.
2. **Car AP is *higher* on KITTI than COCO for FCOS (+4.0) and YOLO-World (+6.7).** *Hypothesis:*
   KITTI cars are canonical (rear/side views, road context, little clutter), and vans, the main
   ambiguous case, are ignored. *Verify:* rerun the car eval with Van scored as car instead of
   ignored (sensitivity only). Car AP should then drop.
3. **YOLO-World has the worst KITTI person AP (12.8) but the best truck AP (27.7).** *Hypothesis:*
   the prompt word "pedestrian" is a weaker CLIP text embedding than "person", which COCO
   captions use far more often. *Verify:* change `prompts.kitti` to use "person" instead of
   "pedestrian" and re-run `yoloworld_s` on KITTI (~1 min).
4. **The first benchmark reported YOLO11s at 56.5 ms on COCO, 10x its real cost.** This was a
   one-time setup cost for each new input shape (9/200 imgs up to 1,046 ms, gone on a second
   pass), fixed with warmup = 200 + 3 repeats. Worth one sentence in the report's method section:
   latency measurement needs a protocol, not one loop.

## Limitations to state
No DontCare regions (F7). COCO's absolute area thresholds are applied to 1242x375 images (F4).
Truck AP rests on 223 GT boxes. One train/val split, likely correlated by drive (F6).
Windows/WDDM latency is noisy and quoted as median of 3 (F8). FCOS FP32 vs others FP16.
