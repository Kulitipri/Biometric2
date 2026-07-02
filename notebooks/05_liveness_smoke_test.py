"""Sanity check cho liveness/PAD: ảnh mặt THẬT (live) vs ảnh chiếu-lại (spoof).

Mục đích kép:
  1. Xác nhận MiniFASNet ONNX load + chạy được trên pipeline (input/output đúng shape).
  2. Tự lộ 2 chỗ để ngỏ trong src/liveness.py:
       - live_index : in cả 3 lớp softmax cho ảnh THẬT → lớp nào cao nhất chính là "live".
       - normalize  : nếu ảnh thật mà cả 3 lớp ~0.33 (random) → preprocessing sai.

Cách lấy ảnh test:
  - real  : tự chụp mặt mình bằng webcam (mặt thật trước camera).
  - spoof : mở ảnh mặt mình trên điện thoại RỒI chụp lại màn hình đó bằng webcam
            (đúng kịch bản replay attack đang gặp).

Usage:
    python notebooks/05_liveness_smoke_test.py \
        --real  data/team_photos/khai/live.jpg \
        --spoof data/team_photos/khai/phone_replay.jpg \
        --weights weights/minifasnet.onnx

    # chưa biết live_index? chạy chỉ với --real, xem 3-class softmax rồi set:
    python notebooks/05_liveness_smoke_test.py --real data/team_photos/khai/live.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép chạy script trực tiếp từ project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.detector import FaceDetector
from src.liveness import LivenessDetector, _softmax, crop_with_scale
from src.utils import read_image_bgr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="Ảnh mặt THẬT trước camera (live).")
    p.add_argument("--spoof", default=None,
                   help="Ảnh mặt chiếu qua màn hình phone / in giấy (spoof). Optional.")
    p.add_argument("--weights", default="weights/minifasnet.onnx")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--scale", type=float, default=2.7)
    p.add_argument("--input-size", type=int, default=80)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--live-index", type=int, default=1,
                   help="Index lớp 'live' trong output 3-class. Xem 3-class softmax "
                        "in ra cho ảnh --real để chọn đúng.")
    return p.parse_args()


def detect_bbox(detector: FaceDetector, img: np.ndarray, tag: str) -> np.ndarray:
    """Lấy bbox của mặt rõ nhất; raise nếu không thấy mặt."""
    faces = detector.detect(img)
    if not faces:
        raise RuntimeError(f"No face detected in {tag}.")
    return faces[0]["bbox"]


def diagnose(liveness: LivenessDetector, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """In full 3-class softmax (giúp chốt live_index + kiểm tra normalize)."""
    crop = crop_with_scale(img, bbox, liveness.scale, liveness.input_size)
    x = liveness._preprocess(crop)
    logits = liveness.session.run(None, {liveness.input_name: x})[0]  # (1, 3)
    probs = _softmax(logits)[0]  # shape: (3,)
    print(f"    3-class softmax: {np.round(probs, 4)}  (argmax = lớp {int(probs.argmax())})")
    return probs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    detector = FaceDetector(
        det_size=tuple(cfg["detector"]["det_size"]),
        ctx_id=cfg["detector"]["ctx_id"],
        det_thresh=cfg["detector"]["det_thresh"],
    )
    liveness = LivenessDetector(
        weights=args.weights,
        scale=args.scale,
        input_size=args.input_size,
        threshold=args.threshold,
        live_index=args.live_index,
    )

    print(f"ONNX input : {liveness.session.get_inputs()[0].shape}")
    print(f"ONNX output: {liveness.session.get_outputs()[0].shape}")
    print(f"live_index = {args.live_index}, threshold = {args.threshold}\n")

    samples = [("REAL (kỳ vọng live)", args.real, True)]
    if args.spoof:
        samples.append(("SPOOF (kỳ vọng spoof)", args.spoof, False))

    ok = True
    for tag, path, expect_live in samples:
        img = read_image_bgr(path)
        bbox = detect_bbox(detector, img, tag)
        print(f"[{tag}] {path}")
        diagnose(liveness, img, bbox)
        res = liveness.predict(img, bbox)
        passed = res["is_live"] == expect_live
        ok = ok and passed
        print(f"    predict: {res}  -> {'PASS ✓' if passed else 'FAIL ✗'}\n")

    if not args.spoof:
        print("(!) Chưa có --spoof: chỉ kiểm tra ảnh thật. Xem 3-class softmax ở trên: "
              "lớp argmax chính là live_index cần set trong config.")
    print("RESULT:", "ALL PASS ✓" if ok else "có case FAIL ✗ — xem hướng dẫn chỉnh live_index/normalize.")


if __name__ == "__main__":
    main()
