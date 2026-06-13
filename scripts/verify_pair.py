"""Sanity check 1:1 verification trên 2 ảnh thực.

Usage:
    python scripts/verify_pair.py data/team_photos/khai/ref.jpg data/team_photos/khai/probe.jpg
    python scripts/verify_pair.py ref.jpg probe.jpg --override model.name=lvface

Output: cosine similarity + PASS/FAIL theo matcher.threshold trong config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.detector import FaceDetector
from src.embedder import build_embedder
from src.utils import read_image_bgr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("ref", help="Ảnh reference (unmasked, mặt rõ)")
    p.add_argument("probe", help="Ảnh probe (có thể đeo mask)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--override", nargs="*", default=[])
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

    img_ref = read_image_bgr(args.ref)
    img_probe = read_image_bgr(args.probe)

    faces_ref = detector.detect_and_align(img_ref)
    faces_probe = detector.detect_and_align(img_probe)
    if not faces_ref:
        print(f"FAIL: không detect được face trong {args.ref}")
        sys.exit(1)
    if not faces_probe:
        print(f"FAIL: không detect được face trong {args.probe}")
        sys.exit(1)

    emb_ref = embedder.embed(faces_ref[0])
    emb_probe = embedder.embed(faces_probe[0])
    sim = float(emb_ref @ emb_probe)  # cosine vì cả 2 đã L2-normalized

    thresh = cfg["matcher"]["threshold"]
    verdict = "PASS (cùng người)" if sim >= thresh else "FAIL (khác người)"
    print(f"Model:       {embedder.name}")
    print(f"Ref faces:   {len(faces_ref)} (dùng face đầu)")
    print(f"Probe faces: {len(faces_probe)} (dùng face đầu)")
    print(f"Cosine sim:  {sim:.4f}")
    print(f"Threshold:   {thresh}")
    print(f"Verdict:     {verdict}")


if __name__ == "__main__":
    main()
