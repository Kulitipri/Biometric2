"""Face detection + alignment (RetinaFace, 5-landmark, 112x112 ArcFace standard).

InsightFace tự download model pack lần đầu chạy (~/.insightface/models/buffalo_l/).
Nếu offline hoặc muốn dùng pack khác, set INSIGHTFACE_HOME env var.

align_mode:
  - 'detect'  (default): RetinaFace detect → 5-landmark affine → 112x112. Dùng cho
              ảnh full-scene (LFW, team_photos, webcam).
  - 'resize'  : Bỏ qua detector, resize trực tiếp ảnh → 112x112. Dùng cho dataset
              đã pre-cropped face (RMFRD AFDB — ảnh chỉ ~150x140, RetinaFace fail
              vì upscale ratio cao + thiếu head context). Trade-off: mất affine
              alignment, nhưng dataset đã align-ish sẵn nên drift nhỏ. Cả gallery
              + probe cùng treatment → so sánh fair trong dataset đó.

fallback_align_mode:
  - None      (default): nếu detect fail thì sample bị skip (count vào n_skipped).
  - 'resize'  : nếu detect fail, fallback resize ảnh thẳng về 112x112 và đánh dấu
              fallback. Giảm probe loss cho RMFRD nhưng các sample fallback có
              chất lượng thấp hơn — số lượng được log riêng để biết tỉ lệ.

pad_ratio (Lever #4a):
  - 0.0 (default): không pad.
  - >0  : trước khi detect, pad ảnh BORDER_REFLECT theo ratio * min(H, W) mỗi
          cạnh. Mục đích: RMFRD tight-crop không có "tai" / phần trên trán →
          RetinaFace mất tham chiếu khi locate landmark. Pad cho RetinaFace
          một context giả thường tăng landmark quality. Trên RMFRD pad_ratio
          0.15-0.25 + det_size=416 thường là sweet spot.

last_landmark_residual (Lever #5 — measurement only):
  - Sau mỗi call detect_and_align_best có RetinaFace chạy, attribute này được
    set = mean per-point residual (px @ 112x112) khi fit similarity transform
    5-landmark vs template ArcFace. None nếu detect không chạy hoặc fail.
  - Gating là việc của caller (eval/run_experiment.py:encode_dataset) — detector
    chỉ ĐO, không filter, vì gallery (unmasked, K-shot enrollment) và probe
    (masked, strict) thường cần ngưỡng KHÁC NHAU. Đặt gate ở caller cho phép
    probe-only filtering mà không phá gallery prototype.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .utils import setup_onnx_runtime

setup_onnx_runtime()  # phải gọi trước khi import insightface (kéo theo onnxruntime)

from insightface.app import FaceAnalysis  # noqa: E402
from insightface.utils import face_align  # noqa: E402


# ArcFace canonical 5-landmark template @ 112x112 (per insightface.utils.face_align.arcface_dst).
# Trật tự: mắt trái, mắt phải, mũi, khoé miệng trái, khoé miệng phải.
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# 3-point subset (mắt trái, mắt phải, mũi) — bỏ 2 khóe miệng. Dùng cho mặt đeo
# mask: khóe miệng bị che → RetinaFace đoán → warp 5-point méo. 3 landmark trên
# luôn nhìn thấy nên alignment ổn định hơn nhiều với occlusion phần dưới.
ARCFACE_DST_3PT = ARCFACE_DST_112[:3]  # shape: (3, 2)


def landmark_alignment_residual(kps: np.ndarray, image_size: int = 112) -> float:
    """Mean per-landmark distance (pixel, ở không gian sau warp) khi fit similarity
    transform từ 5-landmark detect → template ArcFace.

    Đo "mặt detect được khớp tới đâu với hình học mặt chính diện chuẩn":
      - Frontal sharp face: ~0-2 px
      - Pose hoặc partial occlusion: ~5-10 px
      - Detection nhầm / kps lộn xộn / mask che 2 landmark dưới: >10 px

    Dùng làm landmark-quality gate cho RMFRD (probe masked tight-crop): detector
    nhiều khi return score>0.5 nhưng kps lệch nặng vì che 2 mouth points → align
    ra mặt méo → embedding sai. Gate này ép skip những ca như vậy.

    Returns float('inf') nếu fit thất bại (degenerate kps).
    """
    # Local import: skimage lazy-load (vài trăm ms) — chỉ trả giá khi thực sự gate.
    from skimage import transform as sktransform

    dst = ARCFACE_DST_112 if image_size == 112 else ARCFACE_DST_112 * (image_size / 112.0)
    tform = sktransform.SimilarityTransform()
    src = np.asarray(kps, dtype=np.float32)
    if not tform.estimate(src, dst):
        return float("inf")
    proj = tform(src)
    return float(np.linalg.norm(proj - dst, axis=1).mean())


class FaceDetector:
    """Wrap InsightFace RetinaFace + 5-landmark affine alignment to 112x112."""

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        det_thresh: float = 0.5,
        align_mode: str = "detect",
        fallback_align_mode: str | None = None,
        pad_ratio: float = 0.0,
        align_landmarks: str = "5pt",
        auto_residual_thresh: float = 3.0,
    ) -> None:
        assert align_mode in {"detect", "resize"}, f"align_mode={align_mode!r}"
        assert fallback_align_mode in {None, "resize"}, (
            f"fallback_align_mode={fallback_align_mode!r}"
        )
        assert 0.0 <= pad_ratio < 1.0, f"pad_ratio={pad_ratio!r} ∉ [0, 1)"
        assert align_landmarks in {"5pt", "3pt", "auto"}, (
            f"align_landmarks={align_landmarks!r}"
        )
        self.align_mode = align_mode
        self.fallback_align_mode = fallback_align_mode
        self.pad_ratio = pad_ratio
        # Lever mới: chọn template landmark cho affine.
        #   5pt  = mặc định, hành vi cũ (không phá eval đã chạy)
        #   3pt  = 2 mắt + mũi, robust với mask che phần dưới mặt
        #   auto = đo residual 5pt; nếu lệch hơn auto_residual_thresh px thì
        #          chuyển sang 3pt (suy ra khóe miệng nhiều khả năng bị che)
        self.align_landmarks = align_landmarks
        self.auto_residual_thresh = auto_residual_thresh
        # Đo (không gate) residual mỗi call có detect chạy → caller filter.
        self.last_landmark_residual: float | None = None
        # InsightFace appends "models/<name>/" to root, so pass the parent dir.
        insight_home = os.path.expanduser(
            os.environ.get("INSIGHTFACE_HOME", "~/.insightface")
        )

        # allowed_modules chỉ có ở một số version insightface.
        # Fallback để tương thích version hiện tại trong venv.
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                root=insight_home,
                allowed_modules=["detection"],
            )
        except TypeError:
            self.app = FaceAnalysis(name="buffalo_l", root=insight_home)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        self.det_thresh = det_thresh

    def detect(self, img: np.ndarray) -> list[dict]:
        """Return list of {bbox, kps, score} sorted by detection score desc."""
        faces = self.app.get(img)
        out = [
            {"bbox": f.bbox, "kps": f.kps, "score": float(f.det_score)}
            for f in faces
        ]
        out.sort(key=lambda f: f["score"], reverse=True)
        return out

    def align(
        self, img: np.ndarray, kps: np.ndarray, mode: str | None = None
    ) -> np.ndarray:
        """Affine-warp face to 112x112 using landmark template.

        mode None → dùng self.align_landmarks. "auto" resolve thành 5pt/3pt
        bằng residual của bộ 5 landmark.
        """
        mode = mode or self.align_landmarks
        if mode == "auto":
            res = landmark_alignment_residual(kps)
            mode = "3pt" if res > self.auto_residual_thresh else "5pt"
        if mode == "3pt":
            return self._warp_by_landmarks(img, kps[:3], ARCFACE_DST_3PT)
        # 5pt: dùng norm_crop của insightface (giữ nguyên hành vi cũ).
        # shape: (112, 112, 3), BGR uint8
        return face_align.norm_crop(img, landmark=kps, image_size=112)

    def _warp_by_landmarks(
        self, img: np.ndarray, kps: np.ndarray, dst: np.ndarray
    ) -> np.ndarray:
        """Fit SimilarityTransform (src landmarks → dst template) rồi warpAffine.

        Tự dựng affine thay vì norm_crop vì norm_crop cứng 5 điểm. Degenerate
        (vd 3 điểm thẳng hàng) → fallback resize để không vỡ pipeline.
        """
        from skimage import transform as sktransform

        tform = sktransform.SimilarityTransform()
        src = np.asarray(kps, dtype=np.float32)
        if not tform.estimate(src, dst):
            return self._resize_face(img)
        M = tform.params[0:2, :]  # 2x3 affine matrix
        return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    def _resize_face(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

    def _maybe_pad(self, img: np.ndarray) -> np.ndarray:
        """Lever #4a: pad BORDER_REFLECT trước khi detect.

        Pad cùng số pixel mọi cạnh → bbox/landmark vẫn đúng coordinate space
        so với ảnh đã pad; downstream alignment dùng cùng ảnh đã pad nên
        không cần shift ngược về ảnh gốc.
        """
        if self.pad_ratio <= 0.0:
            return img
        h, w = img.shape[:2]
        pad = int(round(min(h, w) * self.pad_ratio))
        if pad == 0:
            return img
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    def detect_and_align(self, img: np.ndarray) -> list[np.ndarray]:
        """Convenience: detect → align all faces. Returns list of (112,112,3),
        best detection first.

        Khi align_mode='resize': skip detector, resize ảnh thẳng về 112x112 và
        return [single_face].
        """
        if self.align_mode == "resize":
            return [self._resize_face(img)]
        padded = self._maybe_pad(img)
        return [self.align(padded, f["kps"]) for f in self.detect(padded)]

    def detect_and_align_best(
        self, img: np.ndarray
    ) -> tuple[np.ndarray | None, str, float]:
        """Return (best_face, source_tag, det_score) for a single image.

        source_tag values:
            'resize'           -- align_mode='resize' (bypass detector by config)
            'detect'           -- RetinaFace found at least one face; pick highest
            'fallback_resize'  -- detector failed; fallback_align_mode='resize'
                                  resizes whole image as a last resort
            'skip'             -- detector failed and no fallback configured

        Side-effect: self.last_landmark_residual set sau mỗi detect thành công
        (None nếu detect không chạy / fail). Caller dùng để gate (Lever #5,
        áp probe-only — đặt ở encode_dataset, không phải ở đây).
        """
        self.last_landmark_residual = None
        if self.align_mode == "resize":
            return self._resize_face(img), "resize", 0.0
        padded = self._maybe_pad(img)
        faces = self.detect(padded)
        if faces:
            best = faces[0]
            # Đo cho diagnostic / probe-gate downstream (vài trăm µs — rẻ).
            self.last_landmark_residual = landmark_alignment_residual(best["kps"])
            return self.align(padded, best["kps"]), "detect", best["score"]
        if self.fallback_align_mode == "resize":
            return self._resize_face(img), "fallback_resize", 0.0
        return None, "skip", 0.0
