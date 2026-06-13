"""Đo chất lượng alignment qua intra-identity cosine — kiểm chứng Lever #4.

Tăng Rank-1 không đủ: nó có thể đến từ bất kỳ thay đổi nào trong pipeline.
Để xác minh alignment thật sự cải thiện, đo trực tiếp:
    - intra-identity cosine (cùng người, khác ảnh) → CÀNG CAO càng tốt
    - inter-identity cosine (khác người)             → tham chiếu
    - margin = intra - inter                          → CÀNG CAO càng tốt

So sánh 2 config (vd: pad_ratio=0 vs pad_ratio=0.2) trên cùng dataset.
Nếu intra tăng + margin tăng → alignment thực sự tốt hơn, không phải Rank-1
ăn may.

Usage:
    python scripts/measure_alignment_quality.py --config configs/rmfrd_lvface.yaml
    python scripts/measure_alignment_quality.py --config configs/rmfrd_lvface.yaml \\
        --override detector.pad_ratio=0.2 detector.det_size=[416,416]
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
from src.embedder import build_embedder
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rmfrd_lvface.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--split", default="probe", choices=["gallery", "probe"],
                   help="Đo trên gallery (unmasked) hay probe (masked).")
    p.add_argument("--output", default=None,
                   help="Optional JSON path để lưu kết quả.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg["experiment"]["seed"])

    detector = FaceDetector(
        det_size=tuple(cfg["detector"]["det_size"]),
        ctx_id=cfg["detector"]["ctx_id"],
        det_thresh=cfg["detector"]["det_thresh"],
        align_mode=cfg["detector"].get("align_mode", "detect"),
        fallback_align_mode=cfg["detector"].get("fallback_align_mode"),
        pad_ratio=cfg["detector"].get("pad_ratio", 0.0),
    )

    if cfg["model"]["name"] == "ensemble":
        members = cfg["model"]["members"]
        embedder = build_embedder(**members[0])
        print(f"[info] Ensemble config → dùng member đầu '{embedder.name}'")
    else:
        embedder = build_embedder(**cfg["model"])

    # Force K=1 cho đo intra/inter — multi-shot aggregate sẽ ép intra cao giả tạo.
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["gallery_shots"] = 1
    ds = build_dataset(split=args.split, **ds_cfg)
    print(f"[info] Encode {args.split}: {len(ds)} samples")

    embs: list[np.ndarray] = []
    ids: list[str] = []
    skipped = 0
    for i in tqdm(range(len(ds)), desc=f"Encode {args.split}"):
        img, identity = ds[i]
        face, _source, _score = detector.detect_and_align_best(img)
        if face is None:
            skipped += 1
            continue
        embs.append(embedder.embed(face))
        ids.append(identity)
    if not embs:
        sys.exit(f"0/{len(ds)} faces encoded.")
    E = np.stack(embs)
    ids_arr = np.asarray(ids)

    # Build identity → row indices.
    unique_ids, inverse = np.unique(ids_arr, return_inverse=True)

    # Full cosine matrix (N, N).
    sim = E @ E.T
    same_id = inverse[:, None] == inverse[None, :]
    # Exclude diagonal (self-similarity = 1).
    np.fill_diagonal(same_id, False)

    intra_mask = same_id
    inter_mask = ~same_id
    np.fill_diagonal(inter_mask, False)

    # Pull off-diagonal entries only.
    iu = np.triu_indices_from(sim, k=1)
    sim_pairs = sim[iu]
    intra_pairs = intra_mask[iu]

    intra_vals = sim_pairs[intra_pairs]
    inter_vals = sim_pairs[~intra_pairs]

    def _stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    intra_stats = _stats(intra_vals)
    inter_stats = _stats(inter_vals)
    margin = (intra_stats["mean"] - inter_stats["mean"]
              if intra_stats["mean"] is not None and inter_stats["mean"] is not None
              else None)

    report = {
        "config_path": args.config,
        "overrides": args.override,
        "embedder": embedder.name,
        "split": args.split,
        "pad_ratio": cfg["detector"].get("pad_ratio", 0.0),
        "det_size": cfg["detector"]["det_size"],
        "n_total": len(ds),
        "n_encoded": len(embs),
        "n_skipped": skipped,
        "n_identities": int(unique_ids.size),
        "intra": intra_stats,
        "inter": inter_stats,
        "margin_mean": margin,
    }

    print("\n=== Alignment quality ===")
    print(f"  pad_ratio:  {report['pad_ratio']}")
    print(f"  det_size:   {report['det_size']}")
    print(f"  encoded:    {report['n_encoded']}/{report['n_total']} "
          f"(skipped={report['n_skipped']})")
    print(f"  intra mean: {intra_stats['mean']:.4f}  median: {intra_stats['median']:.4f}")
    print(f"  inter mean: {inter_stats['mean']:.4f}  median: {inter_stats['median']:.4f}")
    print(f"  margin:     {margin:.4f}  (CÀNG CAO càng tốt)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
