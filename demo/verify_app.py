"""Gradio 1:1 verification: upload ảnh ref + ảnh masked → cosine + verdict + % quy đổi.

Khác webcam_app.py (identification 1:N realtime): đây là protocol phụ 1:1, 2 ảnh
tĩnh. Tái dùng FaceDetector + build_embedder + matcher.threshold từ config nên
verdict nhất quán với phần eval.

Lưu ý "tỷ lệ giống nhau": model trả về **cosine similarity ∈ [-1, 1]**, KHÔNG phải
xác suất. % hiển thị chỉ là quy đổi tuyến tính (sim+1)/2*100 cho dễ đọc — kết luận
cùng/khác người dựa trên so với threshold, không dựa vào %.

Usage:
    python demo/verify_app.py
    python demo/verify_app.py --override model.name=lvface
    python demo/verify_app.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fusion
from src.config import load_config
from src.detector import FaceDetector
from src.embedder import BaseEmbedder, build_embedder


def load_pipeline(
    config: str, overrides: list[str],
) -> tuple[FaceDetector, list[BaseEmbedder], float]:
    """Khởi tạo detector + (các) embedder + đọc threshold (1 lần lúc start app).

    Detector và ONNX session nặng (~vài trăm MB) nên KHÔNG load lại mỗi click.

    Demo bật sẵn cấu hình mạnh cho masked face (khác eval dùng cấu hình fair):
      - detector align_landmarks="5pt" → ablation RMFRD cho thấy 5pt > 3pt/auto
      - mỗi embedder: tta=True (occlusion_mask TẮT — ablation cho thấy xám hóa
        vùng mask hại Rank-1 vì model pretrained đã xử lý được mask thật)
      - hỗ trợ ensemble (model.name="ensemble") → fusion calibrated cho confidence
        hiển thị (trung tính về accuracy, nhưng cho thang điểm [0,1] dễ đọc)
    """
    cfg = load_config(config, overrides=overrides)
    detector = FaceDetector(
        det_size=tuple(cfg["detector"]["det_size"]),
        ctx_id=cfg["detector"]["ctx_id"],
        det_thresh=cfg["detector"]["det_thresh"],
        align_landmarks="5pt",
    )
    # Demo defaults — cấu hình đã được ablation xác nhận tốt nhất.
    demo_opts = {"tta": True, "occlusion_mask": "none"}
    if cfg["model"]["name"] == "ensemble":
        embedders = [build_embedder(**{**m, **demo_opts}) for m in cfg["model"]["members"]]
    else:
        embedders = [build_embedder(**{**cfg["model"], **demo_opts})]
    threshold = float(cfg["matcher"]["threshold"])
    return detector, embedders, threshold


def align_one(detector: FaceDetector, img_bgr: np.ndarray) -> np.ndarray | None:
    """1 ảnh BGR → crop align 112x112 (face score cao nhất), hoặc None nếu fail.

    detector.detect_and_align() đã sort theo detection score desc nên face[0] là
    face tự tin nhất, không phải face ngẫu nhiên.
    """
    faces = detector.detect_and_align(img_bgr)  # list of (112,112,3) BGR, best first
    if not faces:
        return None
    return faces[0]


def verify(
    img_ref_rgb: np.ndarray | None,
    img_probe_rgb: np.ndarray | None,
    detector: FaceDetector,
    embedders: list[BaseEmbedder],
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Core: nhận 2 ảnh RGB (Gradio) → (crop_ref_rgb, crop_probe_rgb, markdown).

    Gradio đưa ảnh dạng RGB nhưng cả pipeline làm việc trên BGR → convert ngay.
    Crop trả ra cũng convert BGR→RGB để Gradio hiển thị đúng màu.

    Mỗi model tính cosine riêng → fusion.weighted_fuse (calibrate + ưu tiên
    LVFace) → 1 confidence ∈ [0,1]. Verdict dựa trên confidence này, KHÔNG còn
    dùng thang (sim+1)/2 gây hiểu nhầm.
    """
    if img_ref_rgb is None or img_probe_rgb is None:
        return None, None, "⚠️ Cần upload **cả 2 ảnh** trước khi so sánh."

    # Gradio RGB → pipeline BGR.
    ref_bgr = cv2.cvtColor(img_ref_rgb, cv2.COLOR_RGB2BGR)
    probe_bgr = cv2.cvtColor(img_probe_rgb, cv2.COLOR_RGB2BGR)

    face_ref = align_one(detector, ref_bgr)
    face_probe = align_one(detector, probe_bgr)
    if face_ref is None:
        return None, None, "❌ Không detect được khuôn mặt trong **ảnh reference**."
    if face_probe is None:
        return None, None, "❌ Không detect được khuôn mặt trong **ảnh probe (mask)**."

    # Cosine mỗi model (embed() trả (512,) đã L2-normalized → cosine = dot product).
    sims = {}
    for e in embedders:
        sims[e.name] = float(e.embed(face_ref) @ e.embed(face_probe))

    confidence = float(fusion.weighted_fuse({k: np.array(v) for k, v in sims.items()}))
    band = fusion.verdict_band(confidence)

    # Bảng cosine từng model cho minh bạch.
    rows = "".join(f"| `{name}` | {s:.4f} |\n" for name, s in sims.items())
    md = (
        f"### {band}\n\n"
        f"**Độ tin cậy khớp:** **{confidence * 100:.0f}/100** "
        f"<sub>(calibrated, 50 ≈ ranh giới cùng/khác người)</sub>\n\n"
        f"| Model | Cosine thô |\n"
        f"|---|---|\n"
        f"{rows}\n"
        f"<sub>Confidence là điểm fusion đã chuẩn hóa giữa các model (ưu tiên "
        f"LVFace), không phải xác suất. ≥65 khớp mạnh · 45–65 biên · &lt;45 không "
        f"khớp. Cosine thô để tham khảo, mỗi model có thang riêng nên KHÔNG so "
        f"trực tiếp giữa các dòng.</sub>"
    )

    # Crop BGR→RGB cho Gradio hiển thị.
    crop_ref_rgb = cv2.cvtColor(face_ref, cv2.COLOR_BGR2RGB)
    crop_probe_rgb = cv2.cvtColor(face_probe, cv2.COLOR_BGR2RGB)
    return crop_ref_rgb, crop_probe_rgb, md


def build_app(
    detector: FaceDetector, embedders: list[BaseEmbedder], threshold: float,
) -> gr.Blocks:
    """UI: 2 ô upload ảnh, nút So sánh, output 2 crop align + bảng kết quả."""
    model_label = " + ".join(e.name for e in embedders)
    with gr.Blocks(title="Face Verify 1:1 (Masked)") as app:
        gr.Markdown(
            "# So sánh khuôn mặt 1:1 (có khẩu trang)\n"
            "Upload 1 ảnh **reference** (mặt rõ) + 1 ảnh **probe** (có thể đeo mask) "
            "→ model so sánh và đánh giá độ giống nhau.\n\n"
            f"Model: `{model_label}` · align=5pt · TTA=on · "
            f"fusion=calibrated"
        )
        # sources=["upload", "clipboard"] → cho phép kéo-thả file HOẶC paste ảnh
        # đã copy (Ctrl+V) trực tiếp vào ô.
        with gr.Row():
            in_ref = gr.Image(
                la