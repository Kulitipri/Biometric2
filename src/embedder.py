"""Embedder wrappers — all output 512-d L2-normalized vectors.

Three backends share the same interface so matcher.py stays model-agnostic:
- FaceNet  (VGGFace2, baseline yếu)
- ArcFace R100 (Glint360K, baseline mạnh)
- LVFace (ICCV 2025 MFR challenge #1, specialist cho masked face)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .utils import l2_normalize, setup_onnx_runtime

setup_onnx_runtime()  # bootstrap CUDA DLL path + silence ORT warnings


def apply_occlusion_mask(face_bgr: np.ndarray, mode: str = "none") -> np.ndarray:
    """Suppress the mask region of an aligned 112x112 face before embedding.

    Occlusion-aware preprocessing: vùng mask (nửa dưới mặt) không mang identity
    mà còn nhiễu (texture/màu mask) → ép về xám trung tính 127 (≈ 0 sau khi
    normalize [-1,1]) để model "bớt chú ý" vào đó.

    mode:
      - "none"       : không đổi (mặc định, hành vi cũ)
      - "lower_half" : xám hóa từ dưới mũi trở xuống (~hàng 62/112) — vùng mask
      - "periocular" : chỉ giữ dải quanh mắt, xám hóa trán + nửa dưới

    Mốc hàng dựa trên template ArcFace: mắt ~y=51, mũi ~y=71, miệng ~y=92.
    """
    if mode == "none":
        return face_bgr
    out = face_bgr.copy()
    if mode == "lower_half":
        out[62:, :, :] = 127
    elif mode == "periocular":
        out[:28, :, :] = 127   # trán
        out[70:, :, :] = 127   # từ mũi trở xuống
    else:
        raise ValueError(
            f"occlusion_mask={mode!r} — chọn: none | lower_half | periocular"
        )
    return out


class BaseEmbedder(ABC):
    """All embedders must return shape (512,), L2-normalized, float32.

    TTA (test-time augmentation): nếu self.tta=True, embed() forward cả ảnh
    gốc + horizontal flip, mean rồi L2-renormalize → 1 embedding/face. Doubles
    inference cost but tăng robustness (mặt người gần đối xứng — flip giữ
    identity, mask occlusion vẫn cùng vị trí sau flip).
    """

    name: str
    dim: int = 512
    tta: bool = False  # subclass __init__ sets từ config
    occlusion_mask: str = "none"  # subclass __init__ sets từ config

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Input: aligned (112,112,3) BGR uint8. Output: (512,) L2-normalized."""
        # Occlusion-aware: xám hóa vùng mask TRƯỚC khi (tùy chọn) TTA + embed.
        face = apply_occlusion_mask(aligned_face, self.occlusion_mask)
        if self.tta:
            # shape: (2, 112, 112, 3) — original + horizontal flip (axis=1 = width).
            batch = np.stack([face, face[:, ::-1, :]])
            embs = self.embed_batch(batch)  # (2, 512), each L2-normalized
            return l2_normalize(embs.mean(axis=0))
        return self.embed_batch(face[np.newaxis])[0]

    @abstractmethod
    def embed_batch(self, aligned_faces: np.ndarray) -> np.ndarray:
        """Input: (N,112,112,3). Output: (N,512), L2-normalized."""
        ...


class ArcFaceEmbedder(BaseEmbedder):
    name = "arcface"
    DOWNLOAD_HINT = (
        "Download glintr100.onnx (Glint360K R100) from InsightFace model zoo:\n"
        "  https://github.com/deepinsight/insightface/tree/master/model_zoo\n"
        "Hoặc dùng pack 'antelopev2' và copy glintr100.onnx vào weights/."
    )

    def __init__(
        self,
        weights: str,
        providers: list[str] | None = None,
        tta: bool = False,
        occlusion_mask: str = "none",
        **_,
    ) -> None:
        import onnxruntime as ort
        p = Path(weights)
        if not p.exists():
            raise FileNotFoundError(
                f"{self.name} ONNX weights not found: {p}\n{self.DOWNLOAD_HINT}"
            )
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(p), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.tta = tta
        self.occlusion_mask = occlusion_mask

    def _preprocess(self, faces: np.ndarray) -> np.ndarray:
        # (N,112,112,3) BGR uint8 → (N,3,112,112) float32 in [-1, 1] — chuẩn ArcFace.
        x = faces.astype(np.float32)
        x = (x - 127.5) / 127.5
        x = x.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(x)

    def embed_batch(self, aligned_faces: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_faces)
        emb = self.session.run(None, {self.input_name: x})[0]
        return l2_normalize(emb).astype(np.float32)


class FaceNetEmbedder(BaseEmbedder):
    name = "facenet"

    def __init__(self, weights: str, device: str = "cuda", **_) -> None:
        # TODO: load facenet-pytorch InceptionResnetV1(pretrained='vggface2')
        # FaceNet preprocessing KHÁC ArcFace: 160x160 RGB, normalize (x - 127.5) / 128.0.
        # Cần resize 112→160 + BGR→RGB trước khi forward.
        raise NotImplementedError("FaceNet wrapper chưa làm — sẽ thêm sau ArcFace OK.")

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, aligned_faces: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LVFaceEmbedder(ArcFaceEmbedder):
    """LVFace specialist cho masked face.

    Per repo bytedance/LVFace (inference_onnx.py::_preprocess_image):
    input là RGB 112x112, NCHW, normalize ((x/255)-0.5)/0.5 = (x-127.5)/127.5.
    Kh�