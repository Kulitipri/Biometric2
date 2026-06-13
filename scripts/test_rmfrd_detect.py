"""Debug RMFRD detection: thử các combo (det_size, det_thresh, padding) để
tìm config detect được face trên ảnh pre-cropped ~130x140.

Run: python scripts/test_rmfrd_detect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import FaceDetector


def test(det_size, thresh, pad_ratio, n_ids=20, per_id=2):
    det = FaceDetector(det_size=det_size, ctx_id=0, det_thresh=thresh, align_mode="detect")
    root = Path("data/rmfrd/self-built-masked-face-recognition-dataset/AFDB_face_dataset")
    ids = sorted([d for d in root.iterdir() if d.is_dir()])[:n_ids]
    ok, total = 0, 0
    for d in ids:
        for p in sorted(d.glob("*.jpg"))[:per_id]:
            img = cv2.imread(str(p))
            total += 1
            if pad_ratio > 0:
                h, w = img.shape[:2]
                pad = int(max(h, w) * pad_ratio)
                img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            if det.detect(img):
                ok += 1
    print(f"det_size={det_size[0]}, thresh={thresh}, pad={pad_ratio*100:.0f}%: {ok}/{total} ({100*ok/total:.0f}%)")


if __name__ == "__main__":
    configs = [
        ((640, 640), 0.5, 0.0),
        ((320, 320), 0.3, 0.0),
        ((640, 640), 0.3, 0.5),
        ((640, 640), 0.2, 1.0),
        ((320, 320), 0.2, 1.0),
    ]
    for det_size, thresh, pad_ratio in configs:
        test(det_size, thresh, pad_ratio)
