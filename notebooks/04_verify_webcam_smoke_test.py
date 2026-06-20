"""Sanity check cho demo/verify_webcam_app.py — enroll ref unmasked + verify probe masked.

Khác 02_pipeline_smoke_test.py (test embedder thô): script này gọi ĐÚNG code path
của app live — ReferenceStore.enroll() + _fuse_confidence() qua load_pipeline (cùng
cấu hình 5pt + TTA + fusion calibrated mà app thật dùng). Mục đích: bắt lỗi tích hợp
(không chỉ lỗi embedder) trước khi mở webcam.

Expected (per CLAUDE.md): cosine(unmasked, masked) cùng người > 0.4.
Default chạy trên cặp team_photos/ha (ha.png unmasked + ham1.png masked).

Usage:
    python notebooks/04_verify_webcam_smoke_test.py
    python notebooks/04_verify_webcam_smoke_test.py --override model.name=ensemble
    python notebooks/04_verify_webcam_smoke_test.py \
        --ref data/team_photos/hla/hla.jpg --probe data/team_photos/hla/hlam1.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

from src.utils import read_image_bgr
from verify_app import align_one, load_pipeline  # type: ignore
from verify_webcam_app import ReferenceStore, _fuse_confidence  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="data/team_photos/ha/ha.png",
                   help="Ảnh reference UNMASKED (chính chủ).")
    p.add_argument("--probe", default="data/team_photos/ha/ham1.png",
                   help="Ảnh probe MASKED của CÙNG người.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--cos-min", type=float, default=0.4,
                   help="Ngưỡng cosine tối thiểu (per CLAUDE.md rule).")
    return p.parse_args()


def main() -> None:
    # Console Windows mặc định cp1252 không in được ✓/✗ → ép UTF-8 cho an toàn.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    detector, embedders, threshold = load_pipeline(args.config, args.override)

    # Align cả 2 ảnh qua đúng pipeline app dùng.
    face_ref = align_one(detector, read_image_bgr(args.ref))
    face_probe = align_one(detector, read_image_bgr(args.probe))
    if face_ref is None:
        raise RuntimeError(f"Không detect được mặt trong ref: {args.ref}")
    if face_probe is None:
        raise RuntimeError(f"Không detect được mặt trong probe: {args.probe}")

    # Đi qua đúng code path của app: enroll ref → fuse confidence của probe.
    store = ReferenceStore(embedders)
    store.enroll(face_ref)
    confidence = _fuse_confidence(face_probe, embedders, store)

    # Cosine thô từng model để chẩn đoán (lấy max làm verdict — model mạnh nhất).
    cosines = {e.name: float(e.embed(face_probe) @ store.ref[e.name]) for e in embedders}
    best_cos = max(cosines.values())

    print(f"Models:      {', '.join(e.name for e in embedders)}")
    for name, c in cosines.items():
        print(f"  cosine[{name}]: {c:.4f}")
    print(f"Best cosine: {best_cos:.4f}  (cos-min = {args.cos_min})")
    print(f"Confidence:  {confidence * 100:.1f}/100  (app threshold ref = {threshold:.3f})")
    print(f"Result:      {'PASS ✓' if best_cos > args.cos_min else 'FAIL ✗'}")

    assert best_cos > args.cos_min, (
        f"Sanity check failed: best cosine {best_cos:.4f} <= {args.cos_min}. "
        "Kiểm tra: weights đúng model? preprocessing [-1,1]? alignment đúng template? "
        "ref/probe có đúng cùng người không?"
    )


if __name__ == "__main__":
    main()
