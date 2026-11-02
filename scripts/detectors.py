"""Uniform wrappers around every detector so predict.py / benchmark.py treat them identically.

Each wrapper:  det(img_path_or_bgr_ndarray) -> (xyxy[N,4] np.float32, scores[N], cat_ids[N]), timing
cat_ids are already in the wrapper's `label_space`:
    'coco'  -> official COCO category ids (1..90, with gaps)
    'kitti' -> 1-based index into taxonomy.kitti_classes
"""
from __future__ import annotations

import os
import time

import numpy as np

from common import COCO80_NAMES, COCO80_TO_91, p, sync, torch_device


class UltralyticsDetector:
    """YOLO11 / RT-DETR / YOLO-World via the Ultralytics API."""

    def __init__(self, mcfg: dict, dataset: str, cfg: dict, dev: str, imgsz: int, thr: dict):
        from ultralytics import RTDETR, YOLO, YOLOWorld
        impl = mcfg["impl"]
        w = p(mcfg["weights"])
        w.parent.mkdir(parents=True, exist_ok=True)
        cls = {"ultralytics": YOLO, "ultralytics_rtdetr": RTDETR, "ultralytics_world": YOLOWorld}[impl]
        self.model = cls(str(w))
        self.dev, self.imgsz, self.thr = dev, imgsz, thr
        self.half = bool(mcfg.get("half", cfg["half"])) and dev != "cpu"   # per-model override
        kitti_names = cfg["taxonomy"]["kitti_classes"]

        if impl == "ultralytics_world":
            prompts = cfg["prompts"][dataset]
            prompts = COCO80_NAMES if prompts == "coco80" else list(prompts)
            self.model.set_classes(prompts)
            self.prompts = prompts
            if dataset == "coco":
                self.label_space = "coco"
                self.map = np.array(COCO80_TO_91)
            else:
                self.label_space = "kitti"
                self.map = np.array([kitti_names.index(n.replace(" ", "_")) + 1 for n in prompts])
        elif mcfg["label_space"] == "coco":
            self.label_space = "coco"
            self.map = np.array(COCO80_TO_91)
        else:  # model trained on KITTI classes (fine-tuned)
            self.label_space = "kitti"
            self.map = np.arange(1, len(kitti_names) + 1)
        self.nms = impl != "ultralytics_rtdetr"

    def __call__(self, img):
        sync(self.dev)
        t0 = time.perf_counter()
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.thr["conf"], iou=self.thr["nms_iou"],
                               max_det=self.thr["max_det"], device=self.dev,
                               quantize=16 if self.half else None,   # 8.4.x name for half=True
                               verbose=False)[0]
        sync(self.dev)
        total_ms = (time.perf_counter() - t0) * 1000
        b = r.boxes
        xyxy = b.xyxy.float().cpu().numpy()
        scores = b.conf.float().cpu().numpy()
        cats = self.map[b.cls.int().cpu().numpy()] if len(b) else np.zeros(0, dtype=int)
        timing = {"total_ms": total_ms, **{f"{k}_ms": v for k, v in r.speed.items()}}
        return xyxy, scores, cats, timing


class TorchvisionDetector:
    """Faster R-CNN v2 / FCOS from torchvision.models.detection (COCO_V1 weights)."""

    def __init__(self, mcfg: dict, dataset: str, cfg: dict, dev: str, imgsz, thr: dict):
        import torch
        import torchvision
        from torchvision.models import detection as D
        arch = mcfg["arch"]
        builder = getattr(D, arch)
        kw = {}
        if arch.startswith("fasterrcnn"):
            kw = {"box_score_thresh": thr["conf"], "box_detections_per_img": thr["max_det"]}
        elif arch.startswith("fcos") or arch.startswith("retinanet"):
            kw = {"score_thresh": thr["conf"], "detections_per_img": thr["max_det"]}
        if os.environ.get("DETBENCH_RANDOM_TV") == "1":   # offline smoke tests only
            self.model = builder(weights=None, weights_backbone=None, num_classes=91, **kw)
            self.weights_name = "RANDOM (smoke test)"
        else:
            W = torchvision.models.get_model_weights(arch).DEFAULT
            self.model = builder(weights=W, **kw)
            self.weights_name = str(W)
        self.torch = torch
        self.device = torch_device(dev)
        self.dev = dev
        self.model.eval().to(self.device)
        self.half = bool(mcfg.get("half", cfg["half"])) and dev != "cpu"   # per-model override
        self.label_space = "coco"           # torchvision labels ARE COCO category ids
        self.nms = True

    def _load(self, img):
        torch = self.torch
        if isinstance(img, np.ndarray):     # BGR uint8 (cv2) -> RGB float
            t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1)
        else:
            import torchvision
            t = torchvision.io.read_image(str(img), mode=torchvision.io.ImageReadMode.RGB)
        return t

    def __call__(self, img):
        torch = self.torch
        sync(self.dev)
        t0 = time.perf_counter()
        t = self._load(img).to(self.device).float().div_(255.0)
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                    enabled=self.half):
            out = self.model([t])[0]
        sync(self.dev)
        t2 = time.perf_counter()
        xyxy = out["boxes"].float().cpu().numpy()
        scores = out["scores"].float().cpu().numpy()
        cats = out["labels"].cpu().numpy()
        t3 = time.perf_counter()
        timing = {"total_ms": (t3 - t0) * 1000, "preprocess_ms": (t1 - t0) * 1000,
                  "inference_ms": (t2 - t1) * 1000,   # includes internal resize + NMS
                  "postprocess_ms": (t3 - t2) * 1000}
        return xyxy, scores, cats, timing


def build_detector(model_key: str, dataset: str, cfg: dict, dev: str, imgsz=None, thr=None):
    mcfg = cfg["models"][model_key]
    thr = thr or cfg["eval_thresholds"]
    imgsz = imgsz or mcfg.get("imgsz")
    if mcfg["impl"].startswith("ultralytics"):
        return UltralyticsDetector(mcfg, dataset, cfg, dev, int(imgsz), thr)
    if mcfg["impl"] == "torchvision":
        return TorchvisionDetector(mcfg, dataset, cfg, dev, imgsz, thr)
    raise ValueError(mcfg["impl"])
