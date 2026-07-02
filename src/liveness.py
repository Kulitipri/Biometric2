"""Passive face anti-spoofing / liveness — Presentation Attack Detection (PAD).

Bịt lỗ hổng "replay attack": camera nhận diện cả khuôn mặt CHIẾU QUA màn hình
điện thoại / ảnh in. Theo ISO/IEC 30107, đây là tác vụ PAD (Presentation Attack
Detection). Module này wrap MiniFASNet (Silent-Face-Anti-Spoofing) ONNX để phân
biệt mặt THẬT (live) vs mặt GIẢ (print/replay) trước khi cho phép identify.

Quan hệ với pipeline: chạy như một GATE giữa detector và embedder
    detect (bbox) -> LivenessDetector.predict(frame, bbox)
        ├─ spoof -> REJECT (không embed, không identify)
        └─ live  -> align + embed + identify (luồng cũ)

QUAN TRỌNG — vì sao predict() nhận `frame + bbox` chứ KHÔNG nhận mặt đã align 112:
    MiniFASNet được train trên crop bbox NỚI RỘNG (~scale 2.7x), cố tình chứa
    viền màn hình, vân moiré, mép giấy, vùng phản chiếu — chính là dấu vết của
    spoof. Mặt align-sát 112x112 đã vứt bỏ đúng context đó, nên PAD phải tự crop
    lại từ frame gốc theo scale riêng (xem crop_with_scale).

Pretrained-only (đúng chủ trương project): chỉ load ONNX, không train/fine-tune.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import setup_onnx_runtime

setup_onnx_runtime()  # bootstrap CUDA DLL path + silence ORT warnings (giống embedder)


def crop_with_scale(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    scale: float = 2.7,
    out_size: int = 80,
) -> np.ndarray:
    """Crop a face region expanded by `scale` around the bbox center, resized to out_size.

    Port logic _get_new_box của Silent-Face: khi hộp nới ra tràn biên ảnh thì
    DỊCH hộp vào trong (không bóp méo) để giữ nguyên kích thước crop → tỉ lệ mặt
    ổn định, đúng phân phối model được train.

    Args:
        frame_bgr: ảnh gốc full-scene, shape (H, W, 3), BGR uint8.
        bbox: [x1, y1, x2, y2] từ detector.detect() (RetinaFace).
        scale: hệ số nới bbox quanh tâm (2.7 = mặc định Silent-Face MiniFASNetV2).
        out_size: cạnh ảnh đầu ra cho model (80 cho MiniFASNet).

    Returns:
        crop BGR uint8, shape (out_size, out_size, 3).
    """
    import cv2

    src_h, src_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    box_w, box_h = x2 - x1, y2 - y1

    # Cap scale để hộp nới không vượt quá kích thước ảnh (tránh crop rỗng/âm).
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale

    cx, cy = x1 + box_w / 2.0, y1 + box_h / 2.0
    left = cx - new_w / 2.0
    top = cy - new_h / 2.0
    right = cx + new_w / 2.0
    bottom = cy + new_h / 2.0

    # Tràn biên → dịch nguyên hộp vào trong thay vì clamp (giữ size crop).
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - (src_w - 1)
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - (src_h - 1)
        bottom = src_h - 1

    left, top = int(max(left, 0)), int(max(top, 0))
    right, bottom = int(right), int(bottom)
    crop = frame_bgr[top:bottom, left:right]
    return cv2.resize(crop, (out_size, out_size))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis. shape giữ nguyên."""
    z = logits - logits.max(axis=-1, keepdims=True)  # trừ max → tránh exp overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LivenessDetector:
    """Wrap MiniFASNet ONNX → trả quyết định live/spoof cho một khuôn mặt.

    Input ONNX: (1, 3, 80, 80) — crop nới-scale, BGR, thang [0,255] (xem _preprocess).
    Output ONNX: (1, 3) logits 3 lớp. live_index = 1 (convention Silent-Face gốc) —
    verify với preprocessing đúng: 4 ảnh mặt thật cho class 1 ≈ 0.88-0.99.
    """

    name = "minifasnet"
    DOWNLOAD_HINT = (
        "Download MiniFASNet anti-spoofing ONNX và đặt vào weights/minifasnet.onnx:\n"
        "  https://github.com/yakhyo/face-anti-spoofing  (bản PyTorch+ONNX tối giản)\n"
        "  hoặc https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx\n"
        "Model gốc: minivision-ai/Silent-Face-Anti-Spoofing (MiniFASNetV2 2.7_80x80)."
    )

    def __init__(
        self,
        weights: str,
        scale: float = 2.7,
        input_size: int = 80,
        threshold: float = 0.5,
        live_index: int = 1,
        providers: list[str] | None = None,
        **_,
    ) -> None:
        """Load ONNX session.

        Args:
            weights: đường dẫn file .onnx.
            scale: scale crop quanh bbox (phải khớp scale model được train: 2.7).
            input_size: cạnh input model (80).
            threshold: ngưỡng P(live) để coi là người thật.
            live_index: index lớp "live" trong output 3-class (=1, convention
                Silent-Face gốc; verify: ảnh thật → class 1).
            providers: ORT execution providers (mặc định CUDA → CPU fallback).
        """
        import onnxruntime as ort

        p = Path(weights)
        if not p.exists():
            raise FileNotFoundError(
                f"{self.name} ONNX weights not found: {p}\n{self.DOWNLOAD_HINT}"
            )
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(p), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.scale = scale
        self.input_size = input_size
        self.threshold = threshold
        self.live_index = live_index

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """(out,out,3) BGR uint8 → (1,3,out,out) float32 đúng chuẩn MiniFASNet.

        MiniFASNet (Silent-Face) dùng custom ToTensor: CHỈ đổi sang float + transpose,
        GIỮ thang [0,255], **KHÔNG chia 255** (khác torchvision ToTensor) và GIỮ BGR.
        Verify: graph ONNX không có node normalize nào; input [0,1] làm model "chết"
        (output ~hằng số), chỉ input [0,255] mới phản ứng theo nội dung.
        """
        x = crop_bgr.astype(np.float32)  # [0,255], vẫn BGR — KHÔNG /255
        x = x.transpose(2, 0, 1)         # HWC → CHW
        x = x[np.newaxis]                # → (1,3,out,out)
        return np.ascontiguousarray(x)

    def predict(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> dict:
        """Quyết định live/spoof cho 1 mặt.

        Args:
            frame_bgr: ảnh gốc full-scene (H,W,3) BGR uint8.
            bbox: [x1,y1,x2,y2] của mặt cần kiểm (từ detector.detect()).

        Returns:
            dict {
                "is_live": bool,        # score >= threshold
                "score":   float,       # P(live) sau softmax, ∈ [0,1]
                "label":   str,         # "live" | "spoof"
            }
        """
        crop = crop_with_scale(frame_bgr, bbox, self.scale, self.input_size)
        # Fail-closed: crop rỗng (bbox lỗi) → coi như spoof, không cho qua.
        if crop.size == 0:
            return {"is_live": False, "score": 0.0, "label": "spoof",
                    "probs": [0.0, 0.0, 0.0]}

        x = self._preprocess(crop)
        logits = self.session.run(None, {self.input_name: x})[0]  # shape: (1, 3)
        prob = _softmax(logits)[0]                                # shape: (3,)
        score = float(prob[self.live_index])
        is_live = score >= self.threshold
        return {
            "is_live": is_live,
            "score": score,
            "label": "live" if is_live else "spoof",
            "probs": [round(float(p), 4) for p in prob],  # full 3-class cho debug
        }


def build_liveness(
    weights: str,
    enabled: bool = True,
    **kwargs,
) -> LivenessDetector | None:
    """Factory giống build_embedder: trả None nếu liveness.enabled=False trong config.

    Cho phép bật/tắt gate qua config mà không phải sửa code demo.
    """
    if not enabled:
        return None
    return LivenessDetector(weights, **kwargs)
