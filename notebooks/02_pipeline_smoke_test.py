"""Sanity check: 1 ảnh unmasked + 1 ảnh masked của cùng team member.

Expected: cosine similarity > 0.4 (per CLAUDE.md testing rule).
Chạy sau mỗi lần thay embedder hoặc đổi alignment.

Usage:
    python notebooks/02_pipeline_smoke_test.py \
        --unmasked data/team_photos/khai/unmasked.jpg \
        --masked   data/team_photos/khai/masked_01.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép chạy script trực tiếp từ project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.detector import FaceDetector
from src.embedder import build_embedder
from src.utils import read_image_bgr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--unmasked", required=True, help="Ảnh unmasked của 1 người")
    p.add_argument("--masked", required=True, help="Ảnh masked của CÙNG người")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--override", nargs="*", default=[],
                   help="Override field: key.subkey=value (vd: model.name=lvface).")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="Cosine min. Baseline ArcFace+mask ~0.2-0.3; "
                        "unmasked-unmasked ~0.5+; LVFace+mask ~0.4+.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    detector = FaceDetector(
        det_size=tuple(cfg["detector"]["det_size"]),
        ctx_id=cfg["detector"]["ctx_id"],
        det_thresh=cfg["detector"]["det_thresh"],
    )
    embedder = build_embedder(**cfg["model"])

    for label, path in [("unmasked", args.unmasked), ("masked", args.masked)]:
        img = read_image_bgr(path)
        faces = detector.detect_and_align(img)
        if not faces:
            raise RuntimeError(f"No face detected in {label}: {path}")
        if len(faces) > 1:
            print(f"[warn] {label}: {len(faces)} faces detected, dùng face đầu.")

    img_u = read_image_bgr(args.unmasked)
    img_m = read_image_bgr(args.masked)
    face_u = detector.detect_and_align(img_u)[0]
    face_m = detector.detect_and_align(img_m)[0]

    emb_u = embedder.embed(face_u)
    emb_m = embedder.embed(face_m)
    cosine = float(emb_u @ emb_m)

    print(f"Model:     {embedder.name}")
    print(f"Cosine:    {cosine:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Result:    {'PASS ✓' if cosine > args.threshold else 'FAIL ✗'}")

    assert cosine > args.threshold, (
        f"Sanity check failed: cosine {cosine:.4f} <= {args.threshold}. "
        "Kiểm tra: weights đúng model? preprocessing đúng [-1,1]? alignment đúng template?"
    )


if __name__ == "__main__":
    main()
