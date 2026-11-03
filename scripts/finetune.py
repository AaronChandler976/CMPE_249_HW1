"""Fine-tune YOLO11s (COCO-pretrained) on KITTI train (5,985 imgs), validate on KITTI val (1,496).

  python scripts/finetune.py             # settings from configs/experiments.yaml -> finetune:
  python scripts/finetune.py --quick     # 1 epoch on 5% of data (smoke test)

Output: runs/finetune/yolo11s_kitti/weights/best.pt (+ Ultralytics curves, results.csv)
Then:   python scripts/predict.py --model yolo11s_kitti_ft --dataset kitti
"""
from __future__ import annotations

import argparse
import shutil

from common import METRIC_DIR, ROOT, env_info, load_cfg, log_event, p, resolve_device, save_json, scrub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_cfg()
    ft = cfg["finetune"]
    from ultralytics import YOLO
    dev = resolve_device(cfg["device"])
    base = p(ft["base_weights"])
    base.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base))
    kw = dict(data=str(p(cfg["datasets"]["kitti"]["yolo_yaml"])), epochs=a.epochs or ft["epochs"],
              imgsz=ft["imgsz"], batch=ft["batch"], workers=ft["workers"], patience=ft["patience"],
              device=dev, seed=cfg["seed"], deterministic=True, project=str(ROOT / "runs" / "finetune"),
              name=ft["name"], exist_ok=True, plots=True, verbose=True)
    if a.quick:
        q = cfg["quick"]
        kw.update(epochs=q["finetune_epochs"], fraction=q["finetune_fraction"], batch=4, workers=0,
                  plots=False)
    log_event("finetune_start", **{k: str(v) for k, v in kw.items()})
    model.train(**kw)
    save_dir = ROOT / "runs" / "finetune" / ft["name"]
    # keep the small artefacts (curves, results.csv, args.yaml) in results/ so they get committed
    keep = METRIC_DIR / "finetune"
    keep.mkdir(parents=True, exist_ok=True)
    for f in ["results.csv", "args.yaml", "results.png", "confusion_matrix_normalized.png",
              "BoxPR_curve.png", "PR_curve.png"]:
        if (save_dir / f).exists():
            shutil.copy(save_dir / f, keep / f)
    # Ultralytics writes absolute paths into args.yaml; keep the committed copy path-free
    args_yaml = keep / "args.yaml"
    if args_yaml.exists():
        args_yaml.write_text(scrub(args_yaml.read_text(encoding="utf-8")), encoding="utf-8")
    save_json({"train_kwargs": {k: scrub(str(v)) for k, v in kw.items()}, "env": env_info()},
              keep / "finetune_meta.json")
    log_event("finetune_done", save_dir=str(save_dir))


if __name__ == "__main__":
    main()
