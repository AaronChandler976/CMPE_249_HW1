"""Latency / FPS / peak GPU memory, batch size 1, images pre-loaded in RAM (no disk I/O timed).

  python scripts/benchmark.py                 # all models x datasets in configs (benchmark:)
  python scripts/benchmark.py --models yolo11s --datasets kitti --n 50

Protocol (state this in the report):
  * batch 1, FP16 on GPU, same eval thresholds as accuracy runs (conf 0.001 -> worst-case NMS load)
  * `warmup` untimed iterations, then `n_images` timed images
  * torch.cuda.synchronize() before/after each call; end-to-end = preprocess + model + postprocess
  * peak memory = torch.cuda.max_memory_allocated() after reset (PyTorch allocator, not nvidia-smi)
"""
from __future__ import annotations

import argparse

import numpy as np

from common import METRIC_DIR, env_info, load_cfg, log_event, resolve_device, save_json
from detectors import build_detector
from predict import dataset_images


def bench_one(model: str, dataset: str, cfg: dict, n: int, warmup: int, imgsz=None) -> dict:
    import cv2
    import torch
    dev = resolve_device(cfg["device"])
    imgs, img_dir = dataset_images(cfg, dataset, n)
    arrays = [cv2.imread(str(img_dir / r["file_name"])) for r in imgs]
    if dev != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    det = build_detector(model, dataset, cfg, dev, imgsz=imgsz)
    for i in range(warmup):
        det(arrays[i % len(arrays)])
    if dev != "cpu":
        torch.cuda.reset_peak_memory_stats()
    timings = [det(a)[3] for a in arrays]
    peak = torch.cuda.max_memory_allocated() / 1024**2 if dev != "cpu" else None
    tot = np.array([t["total_ms"] for t in timings])
    n_params = None
    m = getattr(det, "model", None)
    try:
        n_params = sum(x.numel() for x in m.parameters()) / 1e6
    except Exception:  # noqa: BLE001
        pass
    res = {"model": model, "dataset": dataset, "imgsz": imgsz or cfg["models"][model].get("imgsz"),
           "n_images": len(arrays), "warmup": warmup, "device": dev, "half": getattr(det, "half", False),
           "latency_ms_mean": float(tot.mean()), "latency_ms_p50": float(np.percentile(tot, 50)),
           "latency_ms_p90": float(np.percentile(tot, 90)), "fps": float(1000 / tot.mean()),
           "breakdown_ms": {k: float(np.mean([t[k] for t in timings])) for k in timings[0]},
           "peak_mem_mb": peak, "params_m": n_params}
    del det
    if dev != "cpu":
        torch.cuda.empty_cache()
    print(f"[bench] {model} on {dataset}: {res['latency_ms_mean']:.1f} ms  "
          f"({res['fps']:.1f} FPS)  peak {peak if peak is None else round(peak)} MB")
    return res


def main(argv=None) -> None:
    cfg = load_cfg()
    b = cfg["benchmark"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=b["models"])
    ap.add_argument("--datasets", nargs="*", default=b["datasets"])
    ap.add_argument("--n", type=int, default=b["n_images"])
    ap.add_argument("--warmup", type=int, default=b["warmup"])
    ap.add_argument("--extra-kitti-1280", action="store_true", default=True,
                    help="also benchmark yolo11s on KITTI at imgsz 1280 (resolution ablation)")
    a = ap.parse_args(argv)
    rows = []
    for d in a.datasets:
        for m in a.models:
            try:
                rows.append(bench_one(m, d, cfg, a.n, a.warmup))
            except Exception as e:  # one broken model must not kill the whole benchmark
                log_event("ERROR", stage="benchmark", model=m, dataset=d, error=repr(e))
                print(f"[ERROR] benchmark {m} on {d}: {e!r}")
    if a.extra_kitti_1280 and "kitti" in a.datasets and "yolo11s" in a.models:
        r = bench_one("yolo11s", "kitti", cfg, a.n, a.warmup, imgsz=1280)
        r["tag"] = "res1280"
        rows.append(r)
    save_json({"env": env_info(), "rows": rows}, METRIC_DIR / "benchmark.json")
    log_event("benchmark", n=a.n, models=a.models, datasets=a.datasets)


if __name__ == "__main__":
    main()
