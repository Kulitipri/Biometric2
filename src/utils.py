"""Common helpers shared across modules."""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np

_ONNX_SETUP_DONE = False


def setup_onnx_runtime() -> None:
    """Bootstrap ONNX Runtime: expose PyTorch CUDA DLLs + silence warnings.

    Why: env không có CUDA Toolkit, nhưng PyTorch (build cu121) ship sẵn
    cublasLt64_12.dll + cudnn64_9.dll trong site-packages/torch/lib. Thêm dir
    đó vào DLL search path để onnxruntime-gpu load CUDAExecutionProvider được.

    Idempotent — gọi nhiều lần không sao.
    Phải gọi TRƯỚC khi import/dùng onnxruntime hoặc insightface.
    """
    global _ONNX_SETUP_DONE
    if _ONNX_SETUP_DONE:
        return
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        # 0=verbose 1=info 2=warning 3=error 4=fatal
        ort.set_default_logger_severity(3)
    except ImportError:
        pass
    _ONNX_SETUP_DONE = True


def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Row-wise L2 normalize. Works for shape (D,) or (N, D)."""
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + eps)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read image as BGR uint8 (OpenCV convention). Raises FileNotFoundError if missing.

    Uses ``np.fromfile + cv2.imdecode`` instead of ``cv2.imread`` so paths
    containing non-ASCII characters (CJK names in RMFRD AFDB) on Windows are
    decoded reliably — ``cv2.imread`` silently returns None for such paths.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty image file: {p}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image (corrupt or unsupported format): {p}")
    return img


def set_seed(seed: int = 42) -> None:
    """Seed numpy + torch + python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def aggregate_by_identity(
    embs: np.ndarray,
    ids: list[str],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Mean K embeddings per identity → 1 prototype, re-L2-normalized.

    Multi-shot enrollment helper. Preserves first-seen order of identities so
    downstream matcher output stays deterministic given a deterministic dataset.

    Args:
        embs:    (N, D) L2-normalized embeddings.
        ids:     list of identity strings, len == N.
        weights: optional (N,) array of non-negative weights. None = uniform.
                 Lever #2: tăng K thì có ảnh kém kéo prototype lệch — weighting
                 theo det_score × sharpness giảm tác động của ảnh tệ.
                 Trong nhóm cùng identity, weights được renormalize → sum=1
                 trước khi weighted-mean.

    Returns:
        (proto_embs, unique_ids) where proto_embs shape == (n_unique, D),
        L2-normalized; unique_ids in first-seen order.
    """
    # shape: (N, D); ids len N. Group rows by id, mean, renormalize.
    unique_ids: list[str] = []
    id_to_rows: dict[str, list[int]] = {}
    for i, idn in enumerate(ids):
        if idn not in id_to_rows:
            id_to_rows[idn] = []
            unique_ids.append(idn)
        id_to_rows[idn].append(i)

    if weights is None:
        proto_rows = [embs[id_to_rows[idn]].mean(axis=0) for idn in unique_ids]
    else:
        weights = np.asarray(weights, dtype=np.float64)
        assert weights.shape == (embs.shape[0],), (
            f"weights shape {weights.shape} != (N,)={embs.shape[0]}"
        )
        proto_rows = []
        for idn in unique_ids:
            rows = id_to_rows[idn]
            w = weights[rows]
            wsum = w.sum()
            if wsum <= 0:
                # Toàn bộ K ảnh weight 0 — fallback về uniform mean để khỏi NaN.
                w = np.ones_like(w) / len(w)
            else:
                w = w / wsum
            proto_rows.append((embs[rows] * w[:, None]).sum(axis=0))
    proto = np.stack(proto_rows)
    return l2_normalize(proto), unique_ids


def laplacian_sharpness(img_bgr: np.ndarray) -> float:
    """Variance of Laplacian = focus / sharpness proxy.

    Cao = sharp (nhiều high-frequency content); thấp = blur. Đơn vị tùy ảnh,
    chỉ dùng để xếp hạng *trong cùng identity*, không có ngưỡng tuyệt đối.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_quality_weights(
    det_scores: np.ndarray | list[float],
    sharpness: np.ndarray | list[float],
    det_score_floor: float = 0.1,
) -> np.ndarray:
    """Combined quality weight = det_score * sharpness_normalized.

    det_score: confidence của detector, ∈ [0, 1] (fallback samples → floor).
    sharpness: Laplacian variance, đơn vị tự nhiên — chuẩn hoá max-norm để
    đưa về [0, 1] (mỗi identity quan tâm tỉ lệ tương đối, aggregate_by_identity
    renorm theo group nên scale toàn cục không ảnh hưởng).
    """
    d = np.asarray(det_scores, dtype=np.float64)
    s = np.asarray(sharpness, dtype=np.float64)
    # Floor cho fallback (det_score=0) để không bị loại hẳn — vẫn đóng góp
    # nhưng thấp hơn ảnh detect thành công.
    d = np.maximum(d, det_score_floor)
    s_max = s.max() if s.size and s.max() > 0 else 1.0
    s_norm = s / s_max
    return d * s_norm
