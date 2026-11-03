"""Turn results/metrics/*.json into report-ready tables (CSV + Markdown) and figures (PNG).

  python scripts/analyze.py

Tables  results/tables/{main_results,benchmark,domain_shift,domain_shift_scale,finetune,resolution}.{csv,md}
Figures results/figures/{speed_accuracy,domain_shift_per_class,domain_shift_scale,finetune_native,finetune_curve}.png

Fairness note baked in: COCO "3-class AP" = mean of person/car/truck AP on COCO, so the
COCO->KITTI comparison uses the same classes on both sides (80-class mAP vs 3-class mAP is unfair).
"""
from __future__ import annotations

import math

import pandas as pd

from common import FIG_DIR, METRIC_DIR, TABLE_DIR, load_cfg, load_json, log_event

# Reference categorical palette (light mode), fixed order. Text stays in ink colours.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SHARED = ["person", "car", "truck"]
SHORT = {"YOLO11s": "YOLO11s", "RT-DETR-L": "RT-DETR-L", "Faster R-CNN R50-FPN v2": "Faster R-CNN",
         "FCOS R50-FPN": "FCOS", "YOLO-World-S v2": "YOLO-World"}
ZERO_SHOT = ["yolo11s", "rtdetr_l", "frcnn_r50_v2", "fcos_r50", "yoloworld_s"]
PRETTY = {"yolo11s": "YOLO11s", "rtdetr_l": "RT-DETR-L", "frcnn_r50_v2": "Faster R-CNN R50-FPN v2",
          "fcos_r50": "FCOS R50-FPN", "yoloworld_s": "YOLO-World-S v2",
          "yolo11s_kitti_ft": "YOLO11s (KITTI fine-tuned)"}


def nan_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def load_metrics() -> dict:
    out = {}
    for f in METRIC_DIR.glob("*__*.json"):
        m = load_json(f)
        out[(m["run"], m["target"])] = m
    return out


def write(df: pd.DataFrame, name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (TABLE_DIR / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED)


def shared_ap(m: dict, key="AP") -> float | None:
    vals = [m["per_class"][c][key] for c in SHARED if c in m["per_class"]]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = load_cfg()
    M = load_metrics()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- main results
    rows = []
    for (run, tgt), m in sorted(M.items()):
        s = m["stats"]
        rows.append({"run": run, "model": PRETTY.get(m["model"], m["model"]),
                     "family": cfg["models"][m["model"]]["family"], "eval": tgt, "imgsz": m["imgsz"],
                     "n_img": m["n_images"], "AP": s["AP"], "AP50": s["AP50"], "AP75": s["AP75"],
                     "AP_S": s["AP_small"], "AP_M": s["AP_medium"], "AP_L": s["AP_large"],
                     "AR100": s["AR100"]})
    if rows:
        write(pd.DataFrame(rows), "main_results")

    # ---------------- benchmark
    bench = None
    if (METRIC_DIR / "benchmark.json").exists():
        bench = load_json(METRIC_DIR / "benchmark.json")
        b = pd.DataFrame([{"model": PRETTY.get(r["model"], r["model"]), "key": r["model"],
                           "dataset": r["dataset"], "imgsz": r["imgsz"],
                           "latency_ms": r["latency_ms_mean"],
                           "pass_range_ms": (f"{min(r['pass_means_ms']):.1f}-{max(r['pass_means_ms']):.1f}"
                                             if r.get("pass_means_ms") else ""),
                           "p90_ms": r["latency_ms_p90"],
                           "FPS": r["fps"], "peak_mem_MB": r["peak_mem_mb"], "params_M": r["params_m"]}
                          for r in bench["rows"]])
        write(b.drop(columns="key"), "benchmark")

    # ---------------- domain shift (COCO -> KITTI, same 3 classes)
    ds, scale = [], []
    for k in ZERO_SHOT:
        c, kt = M.get((f"{k}__coco", "coco")), M.get((f"{k}__kitti", "kitti_cocomap"))
        if not (c and kt):
            continue
        row = {"model": PRETTY[k], "COCO_AP_80cls": c["stats"]["AP"], "COCO_AP_3cls": shared_ap(c),
               "KITTI_AP_3cls": kt["stats"]["AP"]}
        row["drop_pts"] = (row["COCO_AP_3cls"] or 0) - row["KITTI_AP_3cls"]
        for cls in SHARED:
            row[f"COCO_{cls}"] = c["per_class"].get(cls, {}).get("AP")
            row[f"KITTI_{cls}"] = kt["per_class"].get(cls, {}).get("AP")
        ds.append(row)
        for sz, key in [("small", "AP_small"), ("medium", "AP_medium"), ("large", "AP_large")]:
            scale.append({"model": PRETTY[k], "scale": sz, "COCO_3cls": shared_ap(c, key),
                          "KITTI_3cls": nan_none(kt["stats"][key])})
    if ds:
        dsf = pd.DataFrame(ds)
        write(dsf, "domain_shift")
        write(pd.DataFrame(scale), "domain_shift_scale")

        # figure: per-class COCO vs KITTI, small multiples (one panel per class)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
        names = [SHORT[r["model"]] for r in ds]
        x = range(len(ds))
        for ax, cls in zip(axes, SHARED):
            ax.bar([i - 0.2 for i in x], dsf[f"COCO_{cls}"], 0.38, color=BLUE, label="COCO val2017",
                   edgecolor=SURFACE, linewidth=2)
            ax.bar([i + 0.2 for i in x], dsf[f"KITTI_{cls}"], 0.38, color=ORANGE, label="KITTI val (zero-shot)",
                   edgecolor=SURFACE, linewidth=2)
            style(ax, cls, "", "AP@[.5:.95] (%)" if cls == "person" else "")
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=20, ha="right", color=INK2)
        axes[0].legend(frameon=False, labelcolor=INK2, loc="upper right")
        fig.suptitle("Per-class AP, COCO vs KITTI (no retraining)", x=0.01, ha="left", color=INK,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "domain_shift_per_class.png", dpi=200)
        plt.close(fig)

        # figure: scale breakdown
        sc = pd.DataFrame(scale)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
        for ax, sz in zip(axes, ["small", "medium", "large"]):
            d = sc[sc.scale == sz].reset_index(drop=True)
            xi = range(len(d))
            ax.bar([i - 0.2 for i in xi], d["COCO_3cls"], 0.38, color=BLUE, label="COCO (person/car/truck)",
                   edgecolor=SURFACE, linewidth=2)
            ax.bar([i + 0.2 for i in xi], d["KITTI_3cls"].fillna(0), 0.38, color=ORANGE,
                   label="KITTI (zero-shot)", edgecolor=SURFACE, linewidth=2)
            style(ax, f"{sz} objects", "", "AP (%)" if sz == "small" else "")
            ax.set_xticks(list(xi))
            ax.set_xticklabels([SHORT[m] for m in d["model"]], rotation=20, ha="right", color=INK2)
        axes[0].legend(frameon=False, labelcolor=INK2)
        fig.suptitle("AP by object scale (COCO area thresholds 32² / 96² px)", x=0.01, ha="left",
                     color=INK, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "domain_shift_scale.png", dpi=200)
        plt.close(fig)

    # ---------------- speed vs accuracy (KITTI: its latencies are stable across repeats, COCO's
    # varied by up to 5 ms on Windows/WDDM because of variable input shapes)
    if bench:
        pts = []
        for r in bench["rows"]:
            m = M.get((f"{r['model']}__kitti", "kitti_cocomap"))
            if r["dataset"] == "kitti" and m and not r.get("tag"):
                pts.append((PRETTY[r["model"]], r["latency_ms_mean"], m["stats"]["AP"], r["peak_mem_mb"]))
        if pts:
            fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
            for i, (name, lat, ap_, mem) in enumerate(sorted(pts, key=lambda t: t[1])):
                ax.scatter(lat, ap_, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
                lab = name + (f"\n{mem:.0f} MB peak" if mem else "")
                above = i % 2 == 0                      # alternate so neighbouring labels never collide
                ax.annotate(lab, (lat, ap_), xytext=(8, 6 if above else -8), textcoords="offset points",
                            fontsize=9, color=INK2, va="bottom" if above else "top")
            style(ax, "Speed vs accuracy on KITTI val, zero-shot (batch 1, median of 3 passes)",
                  "End-to-end latency (ms / image)", "KITTI AP@[.5:.95], person/car/truck (%)")
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.margins(x=0.25, y=0.25)     # room for the point labels
            fig.tight_layout()
            fig.savefig(FIG_DIR / "speed_accuracy.png", dpi=200)
            plt.close(fig)

    # ---------------- fine-tuning / closed vs open vocabulary
    ft_rows = []
    for run, label in [("yolo11s__kitti", "YOLO11s zero-shot (COCO weights)"),
                       ("yoloworld_s__kitti", "YOLO-World zero-shot (text prompts)"),
                       ("yolo11s_kitti_ft__kitti", "YOLO11s fine-tuned on KITTI")]:
        m = M.get((run, "kitti_cocomap"))
        if m:
            ft_rows.append({"model": label, "eval": "kitti_cocomap (3 cls)", "AP": m["stats"]["AP"],
                            "AP50": m["stats"]["AP50"], "AP_S": m["stats"]["AP_small"],
                            **{f"AP_{c}": m["per_class"].get(c, {}).get("AP") for c in SHARED}})
    native = {}
    for run, label in [("yoloworld_s__kitti", "YOLO-World zero-shot"),
                       ("yolo11s_kitti_ft__kitti", "YOLO11s fine-tuned")]:
        m = M.get((run, "kitti_native"))
        if m:
            ft_rows.append({"model": label, "eval": "kitti_native (7 cls)", "AP": m["stats"]["AP"],
                            "AP50": m["stats"]["AP50"], "AP_S": m["stats"]["AP_small"]})
            native[label] = {c: v["AP"] for c, v in m["per_class"].items()}
    if ft_rows:
        write(pd.DataFrame(ft_rows), "finetune")
    if native:
        classes = cfg["taxonomy"]["native_eval_classes"]
        fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
        colors = [BLUE, ORANGE, AQUA]
        n = len(native)
        w = 0.8 / n
        for j, (label, d) in enumerate(native.items()):
            ax.bar([i + (j - (n - 1) / 2) * w for i in range(len(classes))],
                   [d.get(c, 0) or 0 for c in classes], w * 0.95, color=colors[j], label=label,
                   edgecolor=SURFACE, linewidth=2)
        style(ax, "KITTI native classes: open-vocabulary zero-shot vs fine-tuned", "", "AP@[.5:.95] (%)")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, color=INK2)
        ax.legend(frameon=False, labelcolor=INK2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "finetune_native.png", dpi=200)
        plt.close(fig)

    # ---------------- fine-tuning curve (Ultralytics' own per-epoch validation, all 8 trained classes incl. misc)
    curve = METRIC_DIR / "finetune" / "results.csv"
    if curve.exists():
        h = pd.read_csv(curve)
        h.columns = [c.strip() for c in h.columns]
        fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
        ax.plot(h["epoch"], 100 * h["metrics/mAP50-95(B)"], color=BLUE, lw=2, label="mAP@[.5:.95]")
        ax.plot(h["epoch"], 100 * h["metrics/mAP50(B)"], color=ORANGE, lw=2, label="mAP@.5")
        style(ax, "YOLO11s fine-tuning on KITTI: validation mAP per epoch", "epoch", "val mAP (%)")
        ax.legend(frameon=False, labelcolor=INK2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "finetune_curve.png", dpi=200)
        plt.close(fig)

    # ---------------- resolution ablation
    r640, r1280 = M.get(("yolo11s__kitti", "kitti_cocomap")), M.get(("yolo11s__kitti__res1280", "kitti_cocomap"))
    if r640 and r1280:
        write(pd.DataFrame([{"imgsz": m["imgsz"], **{k: m["stats"][k] for k in
                             ["AP", "AP50", "AP_small", "AP_medium", "AP_large"]}} for m in (r640, r1280)]),
              "resolution")

    log_event("analyze", n_metric_files=len(M))
    print(f"[analyze] {len(M)} metric files -> {TABLE_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
