"""Đo phân phối short-side + landmark residual của probe RMFRD.

Mục đích: chọn ngưỡng `min_probe_short_side` (Lever #1) và `max_landmark_residual`
(Lever #5) DỰA TRÊN PHÂN PHỐI thực tế, không hardcode tùy tiện.

Chỉ chạy detector (RetinaFace) — KHÔNG embed → nhanh hơn run_experiment nhiều
lần (chỉ vài phút trên toàn probe set).

Usage:
    python scripts/diagnose_probe_quality.py --config configs/rmfrd_clean.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataset import build_dataset
from src.detector import FaceDetector
from src.utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rmfrd_clean.yaml",
                   help="Config xác định dataset + detector. Allowlist (nếu có) "
                        "được áp → diagnose trên cùng subset với eval.")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output", default="experiments/rmfrd_probe_quality.json")
    return p.parse_args()


def percentiles(values, label: str) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        print(f"  {label}: (rỗng)")
        return {}
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    qs = np.percentile(arr, pcts)
    cells = "  ".join(f"p{p}={q:.2f}" for p, q in zip(pcts, qs))
    print(f"  {label} (n={arr.size}): {cells}")
    return {f"p{p}": float(q) for p, q in zip(pcts, qs)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg["experiment"]["seed"])

    # Detector KHÔNG gate (max_landmark_residual=None) để mọi sample có residual
    # đo được — ta đang sample phân phối, chưa filter.
    det_cfg = dict(cfg["detector"])
    det_cfg["max_landmark_residual"] = None
    # Bỏ fallback để phân biệt "detect được" vs "detect fail" rõ.
    det_cfg["fallback_align_mode"] = None
    detector = FaceDetector(
        det_size=tuple(det_cfg["det_size"]),
        ctx_id=det_cfg["ctx_id"],
        det_thresh=det_cfg["det_thresh"],
        align_mode=det_cfg.get("align_mode", "detect"),
        fallback_align_mode=None,
        pad_ratio=det_cfg.get("pad_ratio", 0.0),
        max_landmark_residual=None,
    )

    probe_ds = build_dataset(split="probe", **cfg["dataset"])
    print(f"Probe samples: {len(probe_ds)}")

    rows = []
    short_sides = []
    residuals = []   # chỉ ghi khi detect thành công
    det_scores = []
    n_skip = 0
    for i in tqdm(range(len(probe_ds)), desc="Probe"):
        img, identity = probe_ds[i]
        h, w = img.shape[:2]
        ss = min(h, w)
        short_sides.append(ss)
        face, source, score = detector.detect_and_align_best(img)
        res = detector.last_landmark_residual
        path = str(getattr(probe_ds, "samples", [(None,)])[i][0])
        if face is None:
            n_skip += 1
            rows.append({
                "path": path, "identity": identity, "short_side": int(ss),
                "source": source, "det_score": float(score), "residual": None,
            })
            continue
        det_scores.append(float(score))
        if res is not None:
            residuals.append(float(res))
        rows.append({
            "path": path, "identity": identity, "short_side": int(ss),
            "source": source, "det_score": float(score),
            "residual": float(res) if res is not None else None,
        })

    print(f"\nDetector skip: {n_skip}/{len(probe_ds)} ({100*n_skip/len(probe_ds):.1f}%)")
    print("\n=== Phân phối ===")
    p_short = percentiles(short_sides, "short_side (px)")
    p_resid = percentiles(residuals, "landmark_residual (px)")
    p_score = percentiles(det_scores, "det_score        ")

    # Sweep cutoff để xem trade-off
    print("\n=== Sweep min_short_side (loại nếu < ngưỡng) ===")
    arr_ss = np.asarray(short_sides)
    for t in [60, 70, 80, 90, 100, 120]:
        loss = int((arr_ss < t).sum())
        print(f"  short_side < {t:3d}: loại {loss:4d}/{arr_ss.size} "
              f"({100*loss/arr_ss.size:.1f}%)")
    print("\n=== Sweep max_landmark_residual (gate nếu > ngưỡng) ===")
    arr_r = np.asarray(residuals)
    for t in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]:
        if arr_r.size == 0:
            break
        loss = int((arr_r > t).sum())
        print(f"  residual > {t:4.1f}: gate {loss:4d}/{arr_r.size} "
              f"({100*loss/arr_r.size:.1f}%) trên số detect thành công")

    report = {
        "config_path": args.config,
        "n_probe": len(probe_ds),
        "n_detect_skip": n_skip,
        "percentiles": {
            "short_side": p_short, "residual": p_resid, "det_score": p_score,
        },
        "rows": rows,
    }
    out = Path(args.output)
    ensure_dir(out.parent)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
