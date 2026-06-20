"""Gradio webcam demo — verification 1:1 LIVE (kịch bản phone-unlock thật).

Khác 2 demo kia:
  - verify_app.py : 1:1 nhưng 2 ảnh TĨNH (upload ref + upload probe).
  - webcam_app.py : 1:N realtime (gallery nhiều người → top-K + open-set reject).

App này là "Cách A" — enroll 1 lần + verify live: người dùng chụp 1 ảnh reference
(mặt rõ) → app cache embedding đó. Sau đó đứng trước webcam (có thể đeo mask) →
MỖI FRAME app so cosine(probe, ref) → fuse → 1 confidence → so với 1 ngưỡng →
verdict "chính chủ ✅ / không phải ❌" realtime.

Về bản chất đây là 1:N với N=1: thay vì lấy top-K trong gallery, ta chỉ giữ đúng
1 prototype và quyết định bằng threshold thay vì argmax. Vì vậy tái dùng được gần
hết pattern streaming của webcam_app.py + load_pipeline/align_one của verify_app.py.

Usage:
    python demo/verify_webcam_app.py
    python demo/verify_webcam_app.py --override model.name=lvface
    python demo/verify_webcam_app.py --config configs/default.yaml --share
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
from src.detector import FaceDetector
from src.embedder import BaseEmbedder

# Tái dùng (không copy-paste) helper khởi tạo pipeline + align 1 ảnh từ verify_app.
from verify_app import align_one, load_pipeline  # type: ignore


# ─────────────────────── Reference (state trong RAM) ────────────────────────
class ReferenceStore:
    """Giữ embedding của 1 ảnh reference (chính chủ), tính sẵn cho mỗi model.

    Lưu EMBEDDING chứ không lưu crop: ref chỉ enroll 1 lần nhưng được so ở MỌI
    frame webcam → embed ref sẵn 1 lần để vòng lặp verify khỏi embed lại (giảm
    latency mỗi frame còn đúng 1 lần embed probe).
    """

    def __init__(self, embedders: list[BaseEmbedder]) -> None:
        self.embedders = embedders
        # {model_name: ref_embedding (512,), L2-normalized}. Rỗng = chưa enroll.
        self._ref: dict[str, np.ndarray] = {}

    def enroll(self, face_bgr: np.ndarray) -> None:
        """Embed crop ref bằng mọi model → cache. Ghi đè ref cũ nếu enroll lại."""
        self._ref = {e.name: e.embed(face_bgr) for e in self.embedders}

    def clear(self) -> None:
        self._ref = {}

    def is_empty(self) -> bool:
        return not self._ref

    @property
    def ref(self) -> dict[str, np.ndarray]:
        return self._ref


# ───────────────────────────── Enroll handler ───────────────────────────────
def enroll_reference(
    img_rgb: np.ndarray | None,
    store: ReferenceStore,
    detector: FaceDetector,
) -> tuple[str, np.ndarray | None]:
    """1 ảnh RGB (webcam/upload) → detect+align+cache ref. Trả (status, crop)."""
    if img_rgb is None:
        return "⚠️ Chưa có ảnh — chụp webcam hoặc upload ảnh rõ mặt.", None

    # Gradio đưa ảnh RGB; cả pipeline làm việc trên BGR → convert ngay.
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    face_bgr = align_one(detector, img_bgr)
    if face_bgr is None:
        return "❌ Không detect được khuôn mặt — thử ảnh rõ mặt, đủ sáng, KHÔNG mask.", None

    store.enroll(face_bgr)
    # Crop BGR→RGB cho Gradio hiển thị đúng màu.
    crop_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return "✅ Đã lưu reference. Sang tab **Verify** để xác thực live.", crop_rgb


# ──────────────────────────── Verify (streaming) ────────────────────────────
def _fuse_confidence(
    face_bgr: np.ndarray,
    embedders: list[BaseEmbedder],
    store: ReferenceStore,
) -> float:
    """1 crop probe → cosine vs ref mỗi model → weighted_fuse → confidence [0,1].

    embed() trả (512,) đã L2-normalized nên cosine = dot product. weighted_fuse
    calibrate từng model về thang chung rồi average (ưu tiên LVFace) → 1 điểm
    dùng được cho cả single model lẫn ensemble.
    """
    sims = {
        e.name: np.array(float(e.embed(face_bgr) @ store.ref[e.name]))
        for e in embedders
    }
    return float(fusion.weighted_fuse(sims))


def verify_frame(
    frame_rgb: np.ndarray | None,
    threshold: float,
    store: ReferenceStore,
    detector: FaceDetector,
    embedders: list[BaseEmbedder],
) -> tuple[np.ndarray | None, str]:
    """Stream handler: 1 frame webcam → annotate khung + verdict markdown realtime.

    Verdict: confidence ≥ threshold → "Chính chủ" (mở khoá), ngược lại → từ chối.
    """
    if frame_rgb is None:
        return None, ""
    if store.is_empty():
        return frame_rgb, "⚠️ Chưa có reference — sang tab **Đăng ký** chụp ảnh chính chủ trước."

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    faces = detector.detect(frame_bgr)  # sorted theo score desc
    if not faces:
        return frame_rgb, "🔍 Không thấy khuôn mặt trong khung hình."

    # Chỉ xử lý face tự tin nhất (phone-unlock: 1 người trước camera).
    best = faces[0]
    face_crop = detector.align(frame_bgr, best["kps"])  # (112,112,3) BGR
    conf = _fuse_confidence(face_crop, embedders, store)

    accepted = conf >= threshold
    label = "Chinh chu" if accepted else "Khong khop"
    color = (0, 200, 0) if accepted else (0, 0, 255)  # BGR: xanh nếu nhận, đỏ nếu từ chối

    # Vẽ box + nhãn. Clamp toạ độ để không tràn ảnh.
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = best["bbox"].astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    # cv2.putText không render unicode → nhãn trên frame để ASCII; verdict đầy đủ
    # (có dấu) nằm ở markdown bên cạnh.
    cv2.putText(
        frame_bgr, f"{label} {conf * 100:.0f}", (x1, max(y1 - 8, 14)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )
    out_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    band = "✅ MỞ KHOÁ — đúng chính chủ" if accepted else "🚫 TỪ CHỐI — không phải chính chủ"
    md = (
        f"### {band}\n\n"
        f"**Confidence:** **{conf * 100:.1f}/100** · ngưỡng chấp nhận = "
        f"{threshold * 100:.0f}\n\n"
        f"<sub>Confidence là điểm fusion calibrated [0,100] (ưu tiên LVFace), "
        f"không phải xác suất. ≥ ngưỡng → coi là chính chủ.</sub>"
    )
    return out_rgb, md


# ───────────────────────────────── UI ───────────────────────────────────────
def build_app(
    detector: FaceDetector,
    embedders: list[BaseEmbedder],
    threshold: float,
) -> gr.Blocks:
    """2 tab: Đăng ký reference (1 ảnh) + Verify live (webcam realtime)."""
    # Demo local 1 người dùng → 1 ReferenceStore dùng chung qua closure. Detector/
    # embedder nặng nên giữ singleton (load 1 lần ở main).
    store = ReferenceStore(embedders)
    model_label = " + ".join(e.name for e in embedders)

    with gr.Blocks(title="Face Verify 1:1 Live (Masked)") as app:
        gr.Markdown(
            "# Xác thực khuôn mặt 1:1 LIVE (mở khoá khi đeo mask)\n"
            "**Tab Đăng ký:** chụp/upload 1 ảnh **chính chủ** (mặt rõ, không mask) "
            "→ app lưu lại đặc trưng.  \n"
            "**Tab Verify:** bật webcam (có thể đeo mask) → app xác thực realtime "
            "đây có phải chính chủ không.\n\n"
            f"Model: `{model_label}` · align=5pt · TTA=on · fusion=calibrated · "
            f"threshold(eval ref)={threshold:.3f}"
        )

        with gr.Tab("Đăng ký reference (Enroll)"):
            with gr.Row():
                with gr.Column():
                    ref_img = gr.Image(
                        label="Ảnh chính chủ — mặt rõ, không mask (webcam/upload/paste)",
                        type="numpy",
                        sources=["webcam", "upload", "clipboard"],
                    )
                    with gr.Row():
                        enroll_btn = gr.Button("Lưu reference", variant="primary")
                        clear_btn = gr.Button("Xoá reference", variant="stop")
                with gr.Column():
                    enroll_status = gr.Markdown()
                    crop_preview = gr.Image(label="Reference đã align (112×112)")

            enroll_btn.click(
                fn=lambda img: enroll_reference(img, store, detector),
                inputs=[ref_img],
                outputs=[enroll_status, crop_preview],
            )
            clear_btn.click(
                fn=lambda: (store.clear() or "🗑️ Đã xoá reference.", None),
                outputs=[enroll_status, crop_preview],
            )

        with gr.Tab("Verify live"):
            with gr.Row():
                live_img = gr.Image(
                    label="Webcam",
                    type="numpy",
                    sources=["webcam"],
                    streaming=True,
                )
                with gr.Column():
                    live_out = gr.Image(label="Kết quả (khung + verdict)")
                    thr_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Ngưỡng chấp nhận (confidence tối thiểu để mở khoá)",
                    )
                    live_md = gr.Markdown()

            # streaming=True + .stream → chạy verify_frame mỗi frame webcam.
            live_img.stream(
                fn=lambda frame, thr: verify_frame(frame, thr, store, detector, embedders),
                inputs=[live_img, thr_slider],
                outputs=[live_out, live_md],
                show_progress=False,
            )

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--share", action="store_true", help="Tạo public link Gradio.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector, embedders, threshold = load_pipeline(args.config, args.override)
    app = build_app(detector, embedders, threshold)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
