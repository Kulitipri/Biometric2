"""Build a "clean" RMFRD identity subset via gallery self-consistency.

Mục tiêu: lọc các identity mà folder unmasked KHÔNG thuần một người (label
noise của RMFRD) — DỰA TRÊN tín hiệu unmasked-vs-unmasked, hoàn toàn độc lập
với probe masked. Nhờ vậy Rank-1 đo trên probe sau khi clean là HỢP LỆ, không
circular (khác với audit_rmfrd_label_noise.py vốn flag theo cos(probe, gallery)
— chính đại lượng quyết định Rank-1).

Heuristic (per identity):
  1. Lấy mẫu đều M ảnh unmasked (trải theo sorted name → tránh near-dup cùng
     session dồn về đầu list).
  2. Encode → centroid = L2norm(mean(embs)).
  3. intra_i = emb_i · centroid.
       mean_intra   = mean(intra_i)        # folder thuần 1 người → cao
       outlier_rate = frac(intra_i < tau)  # bắt cả ca lẫn ít lẫn trộn 50/50
  4. Identity "clean" khi mean_intra >= --min-mean-intra VÀ
     outlier_rate <= --max-outlier-rate.

Hai chế độ:
  * Audit (mặc định, không truyền cutoff): chỉ in phân phối percentile + ghi
    report JSON. Dùng để CHỌN cutoff tại gap tự nhiên, không đặt số tùy tiện.
  * Write allowlist (truyền --min-mean-intra): thêm ghi file allowlist
    (1 identity / dòng) cho RMFRDDataset.identity_allowlist đọc.

Usage:
    # B1: audit, xem phân phối
    python scripts/build_clean_rmfrd_subset.py --config configs/rmfrd_k5_ablation.yaml

    # B2: chốt cutoff từ phân phối rồi ghi allowlist
    python scripts/build_clean_rmfrd_subset.py --config configs/rmfrd_k5_ablation.yaml \
        --min-mean-intra 0.45 --max-outlier-rate 0.30 \
        --allowlist-output data/rmfrd/clean_identities.txt
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
from src.dataset import RMFRDDataset, _iter_images, _normalize_extensions
from src.detector import FaceDetector
from src.embedder import build_embedder
from src.utils import ensure_dir, l2_normalize, read_image_bgr, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rmfrd_k5_ablation.yaml",
                   help="YAML config — chỉ dùng để lấy dataset.root, detector và "
                        "(nếu cần) weights path của embedder.")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--embedder", default="arcface",
                   help="Embedder dùng để audit consistency. Mặc định arcface — "
                        "KHÔNG dùng lvface để tránh circular với ensemble headline.")
    p.add_argument("--max-imgs", type=int, default=40,
                   help="Số ảnh unmasked lấy mẫu (trải đều) / identity.")
    p.add_argument("--min-used", type=int, default=5,
                   help="Identity phải encode được >= ngần này ảnh; ít hơn → "
                        "đánh dấu 'insufficient', không tính clean.")
    p.add_argument("--outlier-tau", type=float, default=0.40,
                   help="Ảnh có intra (cos với centroid) < tau bị coi là outlier.")
    p.add_argument("--max-identities", type=int, default=None,
                   help="Debug: chỉ xử lý N identity đầu.")
    # Cutoff — chỉ ghi allowlist khi truyền --min-mean-intra.
    p.add_argument("--min-mean-intra", type=float, default=None,
                   help="Cutoff: identity clean cần mean_intra >= ngưỡng này. "
                        "Truyền giá trị này để BẬT ghi allowlist.")
    p.add_argument("--max-outlier-rate", type=float, default=1.0,
                   help="Cutoff: identity clean cần outlier_rate <= ngưỡng này.")
    p.add_argument("--output", default="experiments/rmfrd_gallery_consistency.json",
                   help="Report JSON per-identity.")
    p.add_argument("--allowlist-output", default="data/rmfrd/clean_identities.txt",
                   help="File allowlist (chỉ ghi khi có --min-mean-intra).")
    return p.parse_args()


def evenly_spaced(items: list, m: int) -> list:
    """Lấy m phần tử trải đều trên list đã sort (giữ thứ tự, bỏ trùng index)."""
    if len(items) <= m:
        return items
    idx = np.linspace(0, len(items) - 1, m).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    return [items[i] for i in idx]


def resolve_embedder_cfg(cfg: dict, name: str) -> dict:
    """Lấy block config (weights, providers) cho embedder `name` từ cfg.

    Hỗ trợ cả config single-model lẫn ensemble (chọn member trùng tên).
    Fallback weights path mặc định nếu config không khai báo member đó.
    """
    model = cfg["model"]
    providers = model.get("providers")
    if model.get("name") == "ensemble":
        for m in model["members"]:
            if m["name"] == name:
                # strip "name" — build_embedder nhận name qua arg riêng.
                return {"providers": providers, **{k: v for k, v in m.items() if k != "name"}}
    elif model.get("name") == name:
        return {k: v for k, v in model.items() if k != "name"}
    # Không tìm thấy → fallback path theo convention weights/.
    default_weights = {
        "arcface": "weights/arcface_r100.onnx",
        "lvface": "weights/lvface.onnx",
    }
    return {"weights": default_weights[name], "providers": providers}


def analyze_identity(
    img_paths: list[Path],
    detector: FaceDetector,
    embedder,
    outlier_tau: float,
) -> dict | None:
    """Encode các ảnh unmasked của 1 identity → consistency metrics.

    Returns None nếu không encode được ảnh nào (toàn detector skip).
    """
    embs: list[np.ndarray] = []
    n_fallback = 0
    for p in img_paths:
        img = read_image_bgr(p)
        face, source, _ = detector.detect_and_align_best(img)
        if face is None:
            continue
        if source == "fallback_resize":
            n_fallback += 1
        embs.append(embedder.embed(face))
    if not embs:
        return None
    embs_arr = np.stack(embs)                       # shape: (n_used, 512), L2-normalized
    centroid = l2_normalize(embs_arr.mean(axis=0))  # shape: (512,), L2-normalized
    intra = embs_arr @ centroid                     # shape: (n_used,) cosine vs centroid
    return {
        "n_sampled": len(img_paths),
        "n_used": int(embs_arr.shape[0]),
        "n_fallback": n_fallback,
        "mean_intra": float(intra.mean()),
        "median_intra": float(np.median(intra)),
        "min_intra": float(intra.min()),
        "outlier_rate": float((intra < outlier_tau).mean()),
    }


def print_distribution(values: list[float], label: str) -> None:
    """In percentile của một metric để chọn cutoff tại gap tự nhiên."""
    if not values:
        print(f"  {label}: (rỗng)")
        return
    arr = np.asarray(values)
    pcts = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    qs = np.percentile(arr, pcts)
    cells = "  ".join(f"p{p}={q:.3f}" for p, q in zip(pcts, qs))
    print(f"  {label} (n={len(arr)}): {cells}")


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
    emb_cfg = resolve_embedder_cfg(cfg, args.embedder)
    embedder = build_embedder(name=args.embedder, **emb_cfg)
    print(f"Consistency audit dùng embedder: {embedder.name} "
          f"(weights={emb_cfg.get('weights')})")

    # Duyệt thẳng AFDB_face_dataset (TẤT CẢ ảnh unmasked / identity) — KHÔNG
    # qua build_dataset vì split='gallery' chỉ trả K ảnh đầu, còn ta cần đánh
    # giá độ thuần của TOÀN folder.
    root = Path(cfg["dataset"]["root"])
    unmasked_root = root / RMFRDDataset.UNMASKED_DIR
    exts = _normalize_extensions(cfg["dataset"].get("allowed_extensions"))

    id_dirs = sorted(d for d in unmasked_root.iterdir() if d.is_dir())
    if args.max_identities is not None:
        id_dirs = id_dirs[: args.max_identities]

    records: list[dict] = []
    n_insufficient = 0
    for id_dir in tqdm(id_dirs, desc="Identities"):
        all_imgs = _iter_images(id_dir, exts)
        sample = evenly_spaced(all_imgs, args.max_imgs)
        stats = analyze_identity(sample, detector, embedder, args.outlier_tau)
        if stats is None or stats["n_used"] < args.min_used:
            n_insufficient += 1
            records.append({
                "identity": id_dir.name,
                "n_total_imgs": len(all_imgs),
                "status": "insufficient",
                **(stats or {"n_used": 0}),
            })
            continue
        records.append({
            "identity": id_dir.name,
            "n_total_imgs": len(all_imgs),
            "status": "ok",
            **stats,
        })

    # --- Phân phối để chọn cutoff ---
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\nIdentities: {len(records)} total | {len(ok)} ok | "
          f"{n_insufficient} insufficient")
    print("Phân phối (chọn cutoff tại gap tự nhiên):")
    print_distribution([r["mean_intra"] for r in ok], "mean_intra  ")
    print_distribution([r["outlier_rate"] for r in ok], "outlier_rate")

    # --- Áp cutoff (nếu có) → clean flag + allowlist ---
    write_allowlist = args.min_mean_intra is not None
    clean_ids: list[str] = []
    if write_allowlist:
        for r in ok:
            r["clean"] = (
                r["mean_intra"] >= args.min_mean_intra
                and r["outlier_rate"] <= args.max_outlier_rate
            )
            if r["clean"]:
                clean_ids.append(r["identity"])
        print(f"\nCutoff: mean_intra>={args.min_mean_intra} "
              f"AND outlier_rate<={args.max_outlier_rate}")
        print(f"Clean identities: {len(clean_ids)}/{len(ok)} "
              f"(loại {len(ok) - len(clean_ids)} folder nghi trộn người)")

    # --- Save report ---
    report = {
        "config_path": args.config,
        "embedder": embedder.name,
        "params": {
            "max_imgs": args.max_imgs,
            "min_used": args.min_used,
            "outlier_tau": args.outlier_tau,
            "min_mean_intra": args.min_mean_intra,
            "max_outlier_rate": args.max_outlier_rate,
        },
        "n_identities": len(records),
        "n_ok": len(ok),
        "n_insufficient": n_insufficient,
        "n_clean": len(clean_ids) if write_allowlist else None,
        "records": sorted(records, key=lambda r: r.get("mean_intra", -1.0)),
    }
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {out_path}")

    if write_allowlist:
        allow_path = Path(args.allowlist_output)
        ensure_dir(allow_path.parent)
        allow_path.write_text("\n".join(sorted(clean_ids)) + "\n", encoding="utf-8")
        print(f"Allowlist saved: {allow_path} ({len(clean_ids)} identities)")
    else:
        print("(Audit mode — chưa ghi allowlist. Chốt cutoff rồi chạy lại với "
              "--min-mean-intra <x> [--max-outlier-rate <y>].)")


if __name__ == "__main__":
    main()
