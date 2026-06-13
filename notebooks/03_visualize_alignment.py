"""Debug detector + alignment: save crops 112x112 + ảnh gốc có vẽ bbox/landmarks.

Dùng khi smoke test cho cosine lạ — visual check xem detector crop đúng mặt
không (đúng tâm, không skew, không miss landmark do mask).

Usage:
    python notebooks/03_visualize_alignment.py \
        --unmasked data/team_photos/hla/8b6fdf27-...jpg \
        --masked   data/team_photos/hla/d7ab5cf5-...jpg \
        --out-dir  outputs/alignment_check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.detector import FaceDetector
from src.embedder import build_embedder
from src.utils import ensure_dir, read_image_bgr


def annotate(img: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Draw bbox + 5 landmarks lên copy của img."""
    out = img.copy()
    for f in faces:
        x1, y1, x2, y2 = f["bbox"].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x, y) in f["kps"].astype(int):
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
        score_text = f"{f['score']:.2f}"
        cv2.putText(out, score_text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unmasked", required=True)
    p.add_argument("--masked", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out-dir", default="outputs/alignment_check")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)

    detector = FaceDetector(
        det_size=tuple(cfg["detector"]["det_size"]),
        ctx_id=cfg["detector"]["ctx_id"],
        det_thresh=cfg["detector"]["det_thresh"],
    )
    embedder = build_embedder(**cfg["model"])

    embeddings = {}
    for label, path in [("unmasked", args.unmasked), ("masked", args.masked)]:
        img = read_image_bgr(path)
        faces = detector.detect(img)
        print(f"\n=== {label}: {path}")
        print(f"  detected {len(faces)} face(s)")
        for i, f in enumerate(faces):
            print(f"  face {i}: bbox={f['bbox'].astype(int).tolist()} "
                  f"score={f['score']:.3f}")
            print(f"           landmarks (eye_L, eye_R, nose, mouth_L, mouth_R):")
            for (x, y) in f["kps"]:
                print(f"             ({x:.1f}, {y:.1f})")

        annotated = annotate(img, faces)
        cv2.imwrite(str(out_dir / f"annotated_{label}.png"), annotated)

        if faces:
            aligned = detector.align(img, faces[0]["kps"])
            cv2.imwrite(str(out_dir / f"aligned_{label}.png"), aligned)
            embeddings[label] = embedder.embed(aligned)
            print(f"  aligned 112x112 saved | embedding norm = "
                  f"{np.linalg.norm(embeddings[label]):.4f} (should be ~1.0)")

    if len(embeddings) == 2:
        cosine = float(embeddings["unmasked"] @ embeddings["masked"])
        print(f"\nCosine(unmasked, masked) = {cosine:.4f}")

    print(f"\nFiles saved to: {out_dir.resolve()}")
    print("→ Mở 4 file PNG, kiểm tra:")
    print("  - annotated_*.png: bbox có ôm sát mặt không? landmarks có đúng "
          "5 điểm (2 mắt, mũi, 2 khóe miệng)?")
    print("  - aligned_*.png: mặt có centered 112x112, mắt nằm ngang, "
          "không bị skew/crop lệch?")


if __name__ == "__main__":
    main()
