"""Gradio webcam demo — identification 1:N (mở khoá khi đeo mask).

Khác verify_app.py (protocol phụ 1:1, 2 ảnh tĩnh): đây là protocol CHÍNH 1:N
realtime. Người dùng enroll vài identity (ảnh unmasked) vào gallery, rồi đứng
trước webcam (có thể đeo mask) → hệ thống detect → align → embed → so với gallery
→ trả top-1 hoặc "Unknown" nếu điểm dưới ngưỡng (open-set, đúng tinh thần
phone-unlock: từ chối người lạ).

Tái dùng load_pipeline + align_one từ verify_app (không copy-paste) để detector,
embedder và fusion nhất quán với phần eval.

Enroll: multi-shot K (default 3) — mỗi identity nhiều ảnh unmasked → mean thành 1
prototype/model (utils.aggregate_by_identity), khớp design "K=3-5 cho improvement"
trong CLAUDE.md §Core Design Decision 4.

Usage:
    python demo/webcam_app.py
    python demo/webcam_app.py --override model.name=lvface
    python demo/webcam_app.py --config configs/default.yaml --share
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
from src.embedder import BaseEmbedder
from src.liveness import LivenessDetector, build_liveness
from src.matcher import Matcher
from src.utils import aggregate_by_identity

# Tái dùng (không copy-paste) helper khởi tạo pipeline + align 1 ảnh.
from verify_app import align_one, load_pipeline  # type: ignore


# ───────────────────────── Gallery (state trong RAM) ─────────────────────────
class GalleryStore:
    """Lưu các crop đã align theo identity (K-shot) + build prototype/model.

    Tách "storage" (crop 112x112 BGR) khỏi "embedding": gallery có thể đổi
    (thêm người / thêm ảnh) → chỉ rebuild matcher khi cần, không giữ embedding
    cũ lệch với cấu hình embedder.
    """

    def __init__(self, embedders: list[BaseEmbedder], max_shots: int) -> None:
        self.embedders = embedders
        self.max_shots = max_shots
        # id -> list các crop (112,112,3) BGR uint8. dict giữ thứ tự insert (Py3.7+)
        # nên thứ tự identity ổn định cho matcher output.
        self._faces: dict[str, list[np.ndarray]] = {}
        # Cache matchers + cờ dirty: chỉ rebuild (embed lại) khi gallery đổi.
        self._matchers: dict[str, Matcher] = {}
        self._dirty = True

    def add(self, name: str, face_bgr: np.ndarray) -> str:
        """Thêm 1 crop cho identity `name`. Trả message trạng thái."""
        name = name.strip()
        if not name:
            return "⚠️ Tên rỗng — nhập tên trước khi enroll."
        shots = self._faces.setdefault(name, [])
        shots.append(face_bgr)
        # Giữ tối đa K ảnh mới nhất: nếu vượt thì bỏ ảnh cũ nhất (FIFO) để user
        # có thể chụp lại mà không cần reset cả người.
        if len(shots) > self.max_shots:
            del shots[0]
        self._dirty = True
        return f"✅ Đã lưu ảnh cho **{name}** ({len(shots)}/{self.max_shots})."

    def reset(self) -> None:
        """Xoá toàn bộ gallery."""
        self._faces.clear()
        self._matchers = {}
        self._dirty = True

    def is_empty(self) -> bool:
        return not self._faces

    def summary(self) -> list[list]:
        """Bảng [identity, số ảnh] cho Gradio Dataframe."""
        return [[name, len(shots)] for name, shots in self._faces.items()]

    def build_matchers(self) -> dict[str, Matcher]:
        """Embed mọi crop → aggregate prototype/identity/model → {model: Matcher}.

        Cache lại; chỉ rebuild khi gallery đổi (dirty).
        """
        if not self._dirty:
            return self._matchers
        # Flatten crops theo thứ tự identity; ids song song với từng embedding.
        ids: list[str] = []
        faces: list[np.ndarray] = []
        for name, shots in self._faces.items():
            for face in shots:
                ids.append(name)
                faces.append(face)

        matchers: dict[str, Matcher] = {}
        for emb in self.embedders:
            # shape: (N, 512), mỗi hàng L2-normalized.
            embs = np.stack([emb.embed(f) for f in faces])
            # Mean K ảnh/identity → 1 prototype/người, re-L2-normalize.
            protos, unique_ids = aggregate_by_identity(embs, ids)
            matchers[emb.name] = Matcher(protos, unique_ids)

        self._matchers = matchers
        self._dirty = False
        return matchers


# ───────────────────────────── Enroll handler ───────────────────────────────
def enroll(
    img_rgb: np.ndarray | None,
    name: str,
    store: GalleryStore,
    detector: FaceDetector,
) -> tuple[str, list[list], np.ndarray | None]:
    """1 ảnh RGB (webcam/upload) + tên → detect+align+lưu. Trả (md, bảng, crop)."""
    if not name or not name.strip():
        return "⚠️ Nhập **tên** trước khi enroll.", store.summary(), None
    if img_rgb is None:
        return "⚠️ Chưa có ảnh — chụp webcam hoặc upload.", store.summary(), None

    # Gradio đưa ảnh RGB; cả pipeline làm việc trên BGR → convert ngay.
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    face_bgr = align_one(detector, img_bgr)
    if face_bgr is None:
        return "❌ Không detect được khuôn mặt — thử ảnh rõ mặt, đủ sáng.", store.summary(), None

    msg = store.add(name, face_bgr)
    # Crop BGR→RGB cho Gradio hiển thị đúng màu.
    crop_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return msg, store.summary(), crop_rgb


# ─────────────────────────── Identify (streaming) ───────────────────────────
def _fuse_topk(
    face_bgr: np.ndarray,
    embedders: list[BaseEmbedder],
    matchers: dict[str, Matcher],
    k: int = 3,
) -> list[tuple[str, float]]:
    """1 crop probe → top-K (identity, fused_confidence ∈ [0,1]) sorted desc.

    Mỗi model: cosine probe vs prototype (Matcher.score). Calibrate + weighted
    average qua các model (fusion.weighted_fuse) → 1 thang [0,1] dùng chung cho
    cả single model lẫn ensemble.
    """
    # Cosine probe vs prototype cho từng model. Mọi matcher build từ cùng gallery
    # nên các mảng (N,) thẳng hàng theo index identity.
    sims: dict[str, np.ndarray] = {}
    for emb in embedders:
        probe = emb.embed(face_bgr)  # shape: (512,), L2-normalized
        sims[emb.name] = matchers[emb.name].score(probe)  # shape: (N,)

    fused = fusion.weighted_fuse(sims)  # shape: (N,), ∈ [0, 1]
    ids = matchers[embedders[0].name].ids  # cùng thứ tự cho mọi model
    order = np.argsort(-fused)[:k]
    return [(ids[i], float(fused[i])) for i in order]


def identify(
    frame_rgb: np.ndarray | None,
    reject: float,
    store: GalleryStore,
    detector: FaceDetector,
    embedders: list[BaseEmbedder],
    liveness: LivenessDetector | None,
) -> tuple[np.ndarray | None, str]:
    """Stream handler: 1 frame webcam → annotate + verdict markdown.

    Open-set: nếu top-1 confidence < `reject` → "Unknown" (từ chối người lạ).
    PAD gate: nếu liveness phát hiện spoof (mặt chiếu qua phone/in giấy) → từ chối
    NGAY, không identify.
    """
    if frame_rgb is None:
        return None, ""
    if store.is_empty():
        return frame_rgb, "⚠️ Gallery rỗng — sang tab **Đăng ký** để enroll trước."

    matchers = store.build_matchers()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    faces = detector.detect(frame_bgr)  # sorted theo score desc
    if not faces:
        return frame_rgb, "🔍 Không thấy khuôn mặt trong khung hình."

    # Chỉ xử lý face tự tin nhất (use case phone-unlock: 1 người trước camera).
    best = faces[0]

    # ── PAD gate: chặn replay attack TRƯỚC khi identify ──────────────────────
    # MiniFASNet đọc frame GỐC + bbox (không phải crop align 112) để "nhìn thấy"
    # viền màn hình / moiré — dấu vết của mặt chiếu qua phone. Spoof → short-circuit.
    if liveness is not None:
        pad = liveness.predict(frame_bgr, best["bbox"])
        if not pad["is_live"]:
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = best["bbox"].astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)  # đỏ
            cv2.putText(
                frame_bgr, f"SPOOF {pad['score'] * 100:.0f}", (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )
            spoof_md = (
                "### 🚫 Phát hiện giả mạo (spoof)\n\n"
                f"Mặt có vẻ được **chiếu qua màn hình / ảnh in** — P(live) = "
                f"**{pad['score'] * 100:.1f}/100** dưới ngưỡng. Không nhận diện.\n\n"
                "<sub>Anti-spoofing PAD (MiniFASNet). Đưa mặt thật trước camera để mở khoá.</sub>"
            )
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), spoof_md

    face_crop = detector.align(frame_bgr, best["kps"])  # (112,112,3) BGR
    topk = _fuse_topk(face_crop, embedders, matchers, k=3)
    top_id, top_conf = topk[0]

    accepted = top_conf >= reject
    label = top_id if accepted else "Unknown"
    color = (0, 200, 0) if accepted else (0, 0, 255)  # BGR: xanh nếu nhận, đỏ nếu từ chối

    # Vẽ box + nhãn lên frame. Clamp toạ độ để không tràn ảnh.
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = best["bbox"].astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    # cv2.putText không render unicode → nhãn trên frame để ASCII; chi tiết đầy
    # đủ (kể cả tên có dấu) nằm ở bảng markdown bên cạnh.
    cap = f"{label} {top_conf * 100:.0f}"
    cv2.putText(
        frame_bgr, cap, (x1, max(y1 - 8, 14)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )
    out_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    band = "✅ Khớp" if accepted else "🚫 Unknown (dưới ngưỡng)"
    rows = "".join(
        f"| {i + 1} | `{name}` | {conf * 100:.1f} |\n"
        for i, (name, conf) in enumerate(topk)
    )
    md = (
        f"### {band}\n\n"
        f"**Top-1:** `{label}` · confidence **{top_conf * 100:.1f}/100** "
        f"(ngưỡng reject = {reject * 100:.0f})\n\n"
        f"| # | Identity | Confidence |\n|---|---|---|\n{rows}\n"
        f"<sub>Confidence là điểm fusion calibrated [0,100] (ưu tiên LVFace), "
        f"không phải xác suất. Dưới ngưỡng → coi là người lạ.</sub>"
    )
    return out_rgb, md


# ───────────────────────────────── UI ───────────────────────────────────────
def build_app(
    detector: FaceDetector,
    embedders: list[BaseEmbedder],
    threshold: float,
    max_shots: int,
    liveness: LivenessDetector | None,
) -> gr.Blocks:
    """2 tab: Enroll (đăng ký) + Identify (nhận diện realtime)."""
    # Demo chạy local 1 người dùng → 1 GalleryStore dùng chung qua closure (không
    # cần gr.State per-session). Detector/embedder nặng nên giữ singleton.
    store = GalleryStore(embedders, max_shots)
    model_label = " + ".join(e.name for e in embedders)

    with gr.Blocks(title="Face Identify 1:N (Masked)") as app:
        gr.Markdown(
            "# Nhận diện khuôn mặt 1:N (có khẩu trang)\n"
            "**Tab Đăng ký:** thêm người vào gallery bằng ảnh rõ mặt (webcam/upload), "
            f"tối đa {max_shots} ảnh/người.  \n"
            "**Tab Nhận diện:** bật webcam → hệ thống nhận diện realtime, từ chối "
            "người lạ nếu điểm dưới ngưỡng.\n\n"
            f"Model: `{model_label}` · align=5pt · TTA=on · fusion=calibrated · "
            f"threshold(eval ref)={threshold:.3f} · anti-spoof(PAD)="
            f"{'on 🛡️' if liveness is not None else 'off'}"
        )

        with gr.Tab("Đăng ký (Enroll)"):
            with gr.Row():
                with gr.Column():
                    enroll_img = gr.Image(
                        label="Ảnh rõ mặt — webcam, upload hoặc paste (Ctrl+V)",
                        type="numpy",
                        sources=["webcam", "upload", "clipboard"],
                    )
                    name_box = gr.Textbox(label="Tên identity", placeholder="vd: khai")
                    with gr.Row():
                        enroll_btn = gr.Button("Enroll", variant="primary")
                        reset_btn = gr.Button("Xoá gallery", variant="stop")
                with gr.Column():
                    enroll_status = gr.Markdown()
                    crop_preview = gr.Image(label="Crop đã align (112×112)")
                    gallery_df = gr.Dataframe(
                        headers=["Identity", "Số ảnh"],
                        value=store.summary(),
                        label="Gallery hiện tại",
                        interactive=False,
                    )

            enroll_btn.click(
                fn=lambda img, nm: enroll(img, nm, store, detector),
                inputs=[enroll_img, name_box],
                outputs=[enroll_status, gallery_df, crop_preview],
            )
            reset_btn.click(
                fn=lambda: ("🗑️ Đã xoá toàn bộ gallery.", store.reset() or store.summary(), None),
                outputs=[enroll_status, gallery_df, crop_preview],
            )

        with gr.Tab("Nhận diện (Identify)"):
            with gr.Row():
                id_img = gr.Image(
                    label="Webcam",
                    type="numpy",
                    sources=["webcam"],
                    streaming=True,
                )
                with gr.Column():
                    id_out = gr.Image(label="Kết quả (khung + nhãn)")
                    reject_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Ngưỡng reject (confidence tối thiểu để chấp nhận)",
                    )
                    id_md = gr.Markdown()

            # streaming=True + .stream → chạy identify mỗi frame webcam.
            id_img.stream(
                fn=lambda frame, rej: identify(frame, rej, store, detector, embedders, liveness),
                inputs=[id_img, reject_slider],
                outputs=[id_out, id_md],
                show_progress=False,
            )

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--max-shots", type=int, default=3, help="Số ảnh tối đa/identity (K).")
    p.add_argument("--share", action="store_true", help="Tạo public link Gradio.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector, embedders, threshold = load_pipeline(args.config, args.override)
    cfg = load_config(args.config, args.override)
    # build_liveness trả None nếu liveness.enabled=false hoặc thiếu block trong config.
    liveness = build_liveness(**cfg["liveness"]) if "liveness" in cfg else None
    app = build_app(detector, embedders, threshold, args.max_shots, liveness)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
