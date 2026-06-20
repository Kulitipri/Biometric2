"""CLI entry point: chạy 1 combo (model, dataset) → metrics + plots.

Usage:
    # Chạy với default config
    python eval/run_experiment.py

    # Đổi model/dataset nhanh qua override
    python eval/run_experiment.py --override model.name=lvface dataset.name=rmfrd

    # Dùng config riêng cho 1 experiment
    python eval/run_experiment.py --config configs/exp_ablation.yaml

Output:
    experiments/{model}_{dataset}_{timestamp}.json   # metrics
    experiments/{model}_{dataset}_{timestamp}_cmc.png
    experiments/{model}_{dataset}_{timestamp}_roc.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataset import build_dataset
from src.detector import FaceDetector
from src.embedder import build_embedder
from src.matcher import Matcher
from src.restorer import build_restorer
from src.utils import (
    aggregate_by_identity,
    compute_quality_weights,
    ensure_dir,
    laplacian_sharpness,
    set_seed,
)
from eval.metrics import cmc_curve, equal_error_rate, rank_k_accuracy, tar_at_far


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml",
                   help="Đường dẫn YAML config.")
    p.add_argument("--override", nargs="*", default=[],
                   help="Override field: key.subkey=value (vd: model.name=lvface).")
    return p.parse_args()


def encode_dataset(
    dataset,
    detector: FaceDetector,
    embedders: list,
    desc: str,
    collect_quality: bool = False,
    min_short_side: int | None = None,
    max_landmark_residual: float | None = None,
    restorer=None,
) -> tuple[list[np.ndarray], list[str], dict]:
    """Detect → align → embed mỗi sample với MỖI embedder trong list.

    Single-model = list of 1 (zero-overhead). Ensemble = list of N → detector
    chỉ chạy 1 lần / ảnh, mỗi member embed crop đã align (tiết kiệm detector cost).

    Best-face: detector.detect() đã sort theo score desc → lấy face đầu = face
    có confidence cao nhất, không phải face đầu danh sách "ngẫu nhiên".

    Fallback: nếu detector fail và detector.fallback_align_mode='resize',
    sample vẫn được encode bằng resize ảnh gốc. Số lượng fallback được count
    riêng để biết tỉ lệ ảnh không có alignment proper.

    Lever #1 (min_short_side): pre-filter ảnh quá nhỏ TRƯỚC khi detect/encode.
    Tín hiệu độc lập với model → áp được cho probe RMFRD mà không circular. Đếm
    riêng n_too_small để tách khỏi detection failures.

    Lever #5 (max_landmark_residual): probe-only gate, ÁP TẠI ĐÂY (không trong
    detector) — gallery K-shot enrollment có thể cần threshold khác (lenient
    hơn) hoặc tắt hẳn để giữ đủ ảnh prototype. Caller chỉ truyền giá trị này
    cho probe pass.

    Returns (list of (N,512) matrices, ids list, stats dict).
    """
    per_model_embs: list[list] = [[] for _ in embedders]
    ids: list[str] = []
    det_scores: list[float] = []
    sharpness: list[float] = []
    n_too_small = 0
    n_skip_detect = 0
    n_reject_landmark = 0
    n_fallback = 0
    n_restored = 0
    skipped_samples: list = []  # giữ vài path đầu để debug khi all-skipped
    for i in tqdm(range(len(dataset)), desc=desc):
        img, identity = dataset[i]
        # Lever A: phục chế mặt nhỏ TRƯỚC mọi bước. Sau restore ảnh thành 512x512
        # nên gate min_short_side bên dưới sẽ không loại nữa (đúng ý đồ: rescue
        # thay vì drop). Không bật chung min_short_side + restorer trong 1 config.
        if restorer is not None:
            img, was_restored = restorer.restore(img)
            if was_restored:
                n_restored += 1
        if min_short_side is not None and min(img.shape[:2]) < min_short_side:
            n_too_small += 1
            continue
        face, source, det_score = detector.detect_and_align_best(img)
        if face is None:
            n_skip_detect += 1
            if len(skipped_samples) < 3:
                skipped_samples.append(getattr(dataset, "samples", [(None,)])[i])
            continue
        # Lever #5: gate sau detect, dùng residual detector vừa đo. fallback_resize
        # → không có residual (last_landmark_residual=None) → bỏ qua gate cho
        # case đó (đã không có alignment proper, không cần gate thêm).
        if max_landmark_residual is not None:
            res = detector.last_landmark_residual
            if res is not None and res > max_landmark_residual:
                n_reject_landmark += 1
                continue
        if source == "fallback_resize":
            n_fallback += 1
        for j, e in enumerate(embedders):
            per_model_embs[j].append(e.embed(face))
        ids.append(identity)
        if collect_quality:
            det_scores.append(det_score)
            sharpness.append(laplacian_sharpness(face))
    if not ids:
        raise RuntimeError(
            f"{desc}: 0/{len(dataset)} faces encoded — kiểm tra detector hoặc "
            f"data path. Sample paths bị skip: {skipped_samples}"
        )
    n_skipped = n_too_small + n_skip_detect + n_reject_landmark
    stats = {
        "n_total": len(dataset),
        "n_encoded": len(ids),
        "n_skipped": n_skipped,
        "n_too_small": n_too_small,
        "n_skip_detect": n_skip_detect,
        "n_reject_landmark": n_reject_landmark,
        "n_fallback": n_fallback,
        "n_restored": n_restored,
    }
    if collect_quality:
        stats["det_scores"] = det_scores
        stats["sharpness"] = sharpness
    return [np.stack(m) for m in per_model_embs], ids, stats


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg["experiment"]["seed"])

    # --- Build pipeline ---
    detector = FaceDetector(
        det_size=tuple(cfg["detector"]["det_size"]),
        ctx_id=cfg["detector"]["ctx_id"],
        det_thresh=cfg["detector"]["det_thresh"],
        align_mode=cfg["detector"].get("align_mode", "detect"),
        fallback_align_mode=cfg["detector"].get("fallback_align_mode"),
        pad_ratio=cfg["detector"].get("pad_ratio", 0.0),
        # Lever mới: align 3pt/auto cho mặt đeo mask (khóe miệng bị che).
        align_landmarks=cfg["detector"].get("align_landmarks", "5pt"),
        auto_residual_thresh=cfg["detector"].get("auto_residual_thresh", 3.0),
    )
    # Lever mới: occlusion_mask khai báo ở model-level, propagate xuống mỗi member.
    occlusion = cfg["model"].get("occlusion_mask", "none")
    # Ensemble: list of N embedders. Single model: list of 1 (cùng code path).
    if cfg["model"]["name"] == "ensemble":
        members = [
            {**m, "occlusion_mask": m.get("occlusion_mask", occlusion)}
            for m in cfg["model"]["members"]
        ]
        embedders = [build_embedder(**m) for m in members]
        model_save_name = "ensemble_" + "+".join(e.name for e in embedders)
        print(f"Ensemble: {[e.name for e in embedders]} (score-level mean fusion)")
    else:
        embedders = [build_embedder(**{**cfg["model"], "occlusion_mask": occlusion})]
        model_save_name = embedders[0].name

    # Lever A: face restoration (CodeFormer) — None nếu restorer.enabled=false.
    restorer = build_restorer(**cfg.get("restorer", {"enabled": False}))
    if restorer is not None:
        print(
            f"Restorer: CodeFormer (min_short_side={restorer.min_short_side}, "
            f"fidelity={restorer.fidelity})"
        )

    # --- Encode gallery + probe ---
    gallery_ds = build_dataset(split="gallery", **cfg["dataset"])
    probe_ds = build_dataset(split="probe", **cfg["dataset"])
    print(f"Gallery: {len(gallery_ds)} samples | Probe: {len(probe_ds)} samples")

    # Lever #2: chỉ collect quality (det_score + sharpness) cho gallery khi
    # K>1 và bật quality_weighted — tiết kiệm cost cho K=1 (weight không có
    # tác dụng vì mỗi identity chỉ 1 ảnh).
    quality_weighted = cfg["dataset"].get("quality_weighted", False)
    gallery_shots = cfg["dataset"].get("gallery_shots", 1)
    collect_q_gallery = quality_weighted and gallery_shots > 1
    # Probe-only filters (Lever #1 + #5). Gallery enrolment giữ lenient để không
    # mất identity ở bước aggregate K-shot.
    min_probe_short = cfg["dataset"].get("min_probe_short_side")
    max_probe_res = cfg["dataset"].get("max_probe_landmark_residual")
    g_embs, g_ids, g_stats = encode_dataset(
        gallery_ds, detector, embedders, "Gallery",
        collect_quality=collect_q_gallery,
        restorer=restorer,
    )
    p_embs, p_ids, p_stats = encode_dataset(
        probe_ds, detector, embedders, "Probe",
        min_short_side=min_probe_short,
        max_landmark_residual=max_probe_res,
        restorer=restorer,
    )
    print(
        f"Gallery: total={g_stats['n_total']} encoded={g_stats['n_encoded']} "
        f"skipped={g_stats['n_skipped']} fallback={g_stats['n_fallback']}"
    )
    print(
        f"Probe:   total={p_stats['n_total']} encoded={p_stats['n_encoded']} "
        f"skipped={p_stats['n_skipped']} "
        f"(too_small={p_stats['n_too_small']} detect={p_stats['n_skip_detect']} "
        f"landmark={p_stats['n_reject_landmark']}) "
        f"fallback={p_stats['n_fallback']} restored={p_stats['n_restored']}"
    )

    # Multi-shot gallery: aggregate per-model (mỗi model có gallery embs riêng).
    # K=1 thì no-op. ids list giống nhau cho mọi model (cùng dataset, cùng skip).
    if gallery_shots > 1:
        n_before = len(g_ids)
        # Lever #2: quality weights = det_score * sharpness_normalized.
        # Khi quality_weighted off → weights=None → uniform mean (behavior cũ).
        if collect_q_gallery:
            q_weights = compute_quality_weights(
                g_stats["det_scores"], g_stats["sharpness"]
            )
        else:
            q_weights = None
        new_g_embs, new_g_ids = [], g_ids
        for g_emb in g_embs:
            agg, new_g_ids = aggregate_by_identity(g_emb, g_ids, weights=q_weights)
            new_g_embs.append(agg)
        g_embs, g_ids = new_g_embs, new_g_ids
        wlabel = "weighted" if q_weights is not None else "uniform"
        print(
            f"Multi-shot K={gallery_shots} ({wlabel}): "
            f"{n_before} gallery embs -> {len(g_ids)} prototypes"
        )

    # --- Match + metrics ---
    # Score fusion: tính sim per-model. Single model = 1 ma trận.
    sims = [Matcher(g, g_ids).score(p) for g, p in zip(g_embs, p_embs)]
    # fusion mode (chỉ có tác dụng khi ensemble):
    #   "mean"       = mean cosine thô (mặc định, giữ nguyên kết quả cũ)
    #   "calibrated" = src.fusion.weighted_fuse (calibrate per-model + ưu tiên LVFace)
    fusion_mode = cfg["model"].get("fusion", "mean")
    if fusion_mode == "calibrated" and len(embedders) > 1:
        from src.fusion import weighted_fuse
        sims_named = {e.name: s for e, s in zip(embedders, sims)}
        sim = weighted_fuse(sims_named)  # (n_probe, n_gallery), calibrated [0,1]
        print(f"Fusion: calibrated weighted ({[e.name for e in embedders]})")
    else:
        sim = np.mean(sims, axis=0)  # (n_probe, n_gallery)

    cmc_max = cfg["eval"]["cmc_max_rank"]
    cmc = cmc_curve(sim, p_ids, g_ids, max_k=cmc_max)

    # Verification: mỗi cell (probe_i, gallery_j) = 1 trial.
    # Genuine (label=1) khi cùng identity; Impostor (label=0) khi khác.
    # 1-shot gallery → mỗi probe có đúng 1 genuine + (n_gallery-1) impostor pairs.
    p_arr = np.asarray(p_ids)
    g_arr = np.asarray(g_ids)
    labels = (p_arr[:, None] == g_arr[None, :]).astype(np.int32).ravel()
    scores = sim.ravel().astype(np.float64)
    far_targets = cfg["eval"]["far_targets"]
    tar_far = {f"tar@far={t}": tar_at_far(scores, labels, t) for t in far_targets}
    eer, eer_thresh = equal_error_rate(scores, labels)

    # Two flavors of Rank-K:
    #  - *_encoded: chỉ tính trên probe đã encode được (gốc, fair so với
    #    literature). Skipped probes biến mất khỏi mẫu số.
    #  - *_all_probes: skipped probes coi là miss → mẫu số = tổng probe gốc.
    #    Đây là metric headline cho RMFRD vì skip rate là 1 phần chất lượng
    #    pipeline (detector + alignment), không nên giấu bằng cách loại trừ.
    rank_1_encoded = rank_k_accuracy(sim, p_ids, g_ids, k=1)
    rank_5_encoded = rank_k_accuracy(sim, p_ids, g_ids, k=5)
    rank_10_encoded = rank_k_accuracy(sim, p_ids, g_ids, k=10)
    n_probe_total = p_stats["n_total"]
    n_probe_encoded = p_stats["n_encoded"]
    encoded_to_all = (n_probe_encoded / n_probe_total) if n_probe_total else 0.0
    rank_1_all_probes = rank_1_encoded * encoded_to_all
    rank_5_all_probes = rank_5_encoded * encoded_to_all
    rank_10_all_probes = rank_10_encoded * encoded_to_all
    probe_skip_rate = (
        p_stats["n_skipped"] / n_probe_total if n_probe_total else 0.0
    )

    metrics = {
        # Encoded-only (literature-comparable): mẫu số = probe encode được.
        "rank_1_encoded": rank_1_encoded,
        "rank_5_encoded": rank_5_encoded,
        "rank_10_encoded": rank_10_encoded,
        # All-probes (headline cho RMFRD): skipped probes tính là fail.
        "rank_1_all_probes": rank_1_all_probes,
        "rank_5_all_probes": rank_5_all_probes,
        "rank_10_all_probes": rank_10_all_probes,
        # Backward-compat: rank_1/5/10 alias trỏ về encoded để code/notebook cũ chạy.
        "rank_1": rank_1_encoded,
        "rank_5": rank_5_encoded,
        "rank_10": rank_10_encoded,
        "cmc": cmc.tolist(),
        "eer": eer,
        "eer_threshold": eer_thresh,
        **tar_far,
        "n_genuine_pairs": int(labels.sum()),
        "n_impostor_pairs": int((labels == 0).sum()),
        "n_probe_total": n_probe_total,
        "n_probe_encoded": n_probe_encoded,
        "probe_skip_rate": probe_skip_rate,
        "n_probe_fallback": p_stats["n_fallback"],
        "n_probe_too_small": p_stats.get("n_too_small", 0),
        "n_probe_skip_detect": p_stats.get("n_skip_detect", 0),
        "n_probe_reject_landmark": p_stats.get("n_reject_landmark", 0),
        "n_probe_restored": p_stats.get("n_restored", 0),
        "n_gallery_fallback": g_stats["n_fallback"],
        "n_gallery_restored": g_stats.get("n_restored", 0),
    }

    # --- Save ---
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    tag = cfg["experiment"].get("tag") or ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model_save_name}_{cfg['dataset']['name']}{('_' + tag) if tag else ''}_{stamp}"
    # Lọc các list quality nội bộ (det_scores/sharpness) khỏi JSON — không
    # cần trong report, chỉ dùng để tính weights.
    g_stats_out = {k: v for k, v in g_stats.items() if k not in {"det_scores", "sharpness"}}
    p_stats_out = {k: v for k, v in p_stats.items() if k not in {"det_scores", "sharpness"}}
    result = {
        "config": cfg,
        "n_gallery": len(g_ids),
        "n_probe": len(p_ids),
        "gallery_stats": g_stats_out,
        "probe_stats": p_stats_out,
        # Backward-compat alias.
        "n_skipped_gallery": g_stats_out["n_skipped"],
        "n_skipped_probe": p_stats_out["n_skipped"],
        "metrics": metrics,
        "timestamp": stamp,
    }
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRank-1  (encoded):    {metrics['rank_1_encoded']:.4f}")
    print(f"Rank-1  (all probes): {metrics['rank_1_all_probes']:.4f}  "
          f"<- headline (skip = miss, rate={metrics['probe_skip_rate']:.3f})")
    print(f"Rank-5  (encoded):    {metrics['rank_5_encoded']:.4f}")
    print(f"Rank-5  (all probes): {metrics['rank_5_all_probes']:.4f}")
    print(f"Rank-10 (encoded):    {metrics['rank_10_encoded']:.4f}")
    print(f"EER:     {metrics['eer']:.4f} (thresh={metrics['eer_threshold']:.4f})")
    for k in (f"tar@far={t}" for t in far_targets):
        print(f"{k:14s} {metrics[k]:.4f}")
    print(f"Saved:   {out_path}")


if __name__ == "__main__":
    main()
