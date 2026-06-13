"""Audit-only: phát hiện identity / probe nghi ngờ là label noise trong RMFRD.

KHÔNG sửa, move, hay delete dataset gốc. Chỉ sinh report JSON + CSV để bạn
quyết định xem có muốn chạy clean-subset experiment riêng hay không.

Heuristic:
  1. Per-identity centroid của K gallery (unmasked) — coi như "ground-truth"
     embedding của identity đó.
  2. Mỗi masked probe của identity X: tính cosine với centroid X (intra) và
     với centroid identity gần nhất Y != X (inter).
  3. Probe đáng ngờ khi:
        cos(intra) < intra_low_thresh  (kém giống chính mình)
     HOẶC
        cos(inter) - cos(intra) > margin_thresh
        (giống người khác hơn người gán nhãn)
  4. Identity đáng ngờ khi tỉ lệ probe đáng ngờ vượt ratio_thresh.

Usage:
    python scripts/audit_rmfrd_label_noise.py \
        --config configs/rmfrd.yaml \
        --output experiments/rmfrd_label_noise_audit.json
"""

from __future__ import annotations

import argparse
import csv
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
from src.utils import aggregate_by_identity, ensure_dir, l2_normalize, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rmfrd.yaml",
                   help="YAML config; mặc định dùng configs/rmfrd.yaml.")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output", default="experiments/rmfrd_label_noise_audit.json",
                   help="Đường dẫn JSON report.")
    p.add_argument("--csv-output", default=None,
                   help="Tùy chọn: thêm CSV report các probe đáng ngờ.")
    p.add_argument("--intra-low-thresh", type=float, default=0.25,
                   help="Probe cosine với centroid identity gán nhãn thấp hơn "
                        "ngưỡng này → đáng ngờ.")
    p.add_argument("--margin-thresh", type=float, default=0.10,
                   help="Nếu cos(inter best) - cos(intra) > margin → đáng ngờ.")
    p.add_argument("--ratio-thresh", type=float, default=0.5,
                   help="Identity có tỉ lệ probe đáng ngờ > ngưỡng này được "
                        "đánh dấu suspect identity.")
    return p.parse_args()


def encode_split(dataset, detector: FaceDetector, embedder) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode all samples → (embeddings, identity_per_sample, path_per_sample)."""
    embs: list[np.ndarray] = []
    ids: list[str] = []
    paths: list[str] = []
    for i in tqdm(range(len(dataset)), desc=f"Encode {dataset.split}"):
        img, identity = dataset[i]
        face, source = detector.detect_and_align_best(img)
        if face is None:
            continue
        embs.append(embedder.embed(face))
        ids.append(identity)
        sample_path = getattr(dataset, "samples", [(None, None)])[i][0]
        paths.append(str(sample_path) if sample_path is not None else "")
    return np.stack(embs), ids, paths


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
    )

    # Audit dùng embedder duy nhất — ưu tiên model headline. Nếu config là
    # ensemble thì dùng member đầu (LVFace nếu có ưu tiên hơn).
    if cfg["model"]["name"] == "ensemble":
        members = cfg["model"]["members"]
        chosen = next((m for m in members if m["name"] == "lvface"), members[0])
        embedder = build_embedder(**chosen)
        print(f"Audit dùng embedder: {embedder.name} (chọn từ ensemble members)")
    else:
        embedder = build_embedder(**cfg["model"])
        print(f"Audit dùng embedder: {embedder.name}")

    # --- Encode gallery (unmasked) + probe (masked) ---
    gallery_ds = build_dataset(split="gallery", **cfg["dataset"])
    probe_ds = build_dataset(split="probe", **cfg["dataset"])

    g_emb, g_ids, g_paths = encode_split(gallery_ds, detector, embedder)
    p_emb, p_ids, p_paths = encode_split(probe_ds, detector, embedder)

    # Aggregate gallery → 1 centroid / identity, L2-norm.
    centroids, centroid_ids = aggregate_by_identity(g_emb, g_ids)
    id_to_centroid_idx = {idn: i for i, idn in enumerate(centroid_ids)}
    print(f"Centroids: {len(centroid_ids)} identities | Probes encoded: {len(p_ids)}")

    # Cosine matrix: (n_probe, n_id).
    sim = p_emb @ centroids.T

    # Per-probe diagnostics.
    suspect_probes: list[dict] = []
    per_identity_total: dict[str, int] = {}
    per_identity_suspect: dict[str, int] = {}

    for i, identity in enumerate(p_ids):
        per_identity_total[identity] = per_identity_total.get(identity, 0) + 1
        if identity not in id_to_centroid_idx:
            # probe identity không có centroid → coi như suspect (gallery rỗng).
            per_identity_suspect[identity] = per_identity_suspect.get(identity, 0) + 1
            suspect_probes.append({
                "path": p_paths[i],
                "identity": identity,
                "reason": "identity_missing_in_gallery",
                "intra": None,
                "best_inter_id": None,
                "best_inter": None,
                "margin": None,
            })
            continue

        intra_idx = id_to_centroid_idx[identity]
        intra = float(sim[i, intra_idx])

        # Best inter: top similarity với centroid khác.
        masked_row = sim[i].copy()
        masked_row[intra_idx] = -np.inf
        best_inter_idx = int(np.argmax(masked_row))
        best_inter = float(masked_row[best_inter_idx])
        best_inter_id = centroid_ids[best_inter_idx]
        margin = best_inter - intra

        reasons: list[str] = []
        if intra < args.intra_low_thresh:
            reasons.append(f"intra<{args.intra_low_thresh}")
        if margin > args.margin_thresh:
            reasons.append(f"inter-intra>{args.margin_thresh}")
        if reasons:
            per_identity_suspect[identity] = per_identity_suspect.get(identity, 0) + 1
            suspect_probes.append({
                "path": p_paths[i],
                "identity": identity,
                "reason": ",".join(reasons),
                "intra": intra,
                "best_inter_id": best_inter_id,
                "best_inter": best_inter,
                "margin": margin,
            })

    # Identity-level suspect: tỉ lệ probe đáng ngờ > ngưỡng.
    suspect_identities: list[dict] = []
    for identity, total in per_identity_total.items():
        nsus = per_identity_suspect.get(identity, 0)
        ratio = nsus / total if total else 0.0
        if ratio > args.ratio_thresh:
            suspect_identities.append({
                "identity": identity,
                "n_probe": total,
                "n_suspect": nsus,
                "ratio": ratio,
            })
    suspect_identities.sort(key=lambda r: r["ratio"], reverse=True)

    report = {
        "config_path": args.config,
        "embedder": embedder.name,
        "thresholds": {
            "intra_low_thresh": args.intra_low_thresh,
            "margin_thresh": args.margin_thresh,
            "ratio_thresh": args.ratio_thresh,
        },
        "n_centroids": len(centroid_ids),
        "n_probes_encoded": len(p_ids),
        "n_suspect_probes": len(suspect_probes),
        "n_suspect_identities": len(suspect_identities),
        "suspect_identities": suspect_identities,
        "suspect_probes": suspect_probes,
    }

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Suspect probes:     {len(suspect_probes)}/{len(p_ids)}")
    print(f"Suspect identities: {len(suspect_identities)}/{len(per_identity_total)}")
    print(f"Report saved:       {out_path}")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        ensure_dir(csv_path.parent)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["identity", "path", "reason", "intra", "best_inter_id",
                            "best_inter", "margin"],
            )
            writer.writeheader()
            for row in suspect_probes:
                writer.writerow(row)
        print(f"CSV saved:          {csv_path}")


if __name__ == "__main__":
    main()
