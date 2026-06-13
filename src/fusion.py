"""Score-level fusion cho ensemble — calibrate per-model rồi weighted-combine.

Vì sao cần file này: mean cosine thô giữa ArcFace và LVFace là sai, vì hai model
có thang điểm khác nhau (cosine của cặp genuine/impostor nằm ở dải khác nhau —
EER threshold ArcFace ≈ 0.135, LVFace ≈ 0.080 trên LFW masked). Cộng thẳng làm
model có thang cao "lấn át" model kia. Giải pháp: map mỗi cosine về thang chung
[0, 1] (calibrate) rồi mới weighted-average, ưu tiên LVFace (specialist masked).

Calibration ở đây là TĨNH (tham số hard-code tune offline từ EER threshold), đủ
nhẹ cho demo 1:1 — không cần cohort impostor như s-norm đầy đủ.
"""

from __future__ import annotations

import numpy as np

# (center, scale) cho sigmoid calibration mỗi model.
#   center = cosine tại ranh giới genuine/impostor (~EER threshold đo trên LFW masked)
#   scale  = độ "dốc"; nhỏ → chuyển 0→1 gắt quanh center
# calibrated = sigmoid((cosine - center) / scale) ∈ (0, 1); = 0.5 đúng tại center.
CALIB: dict[str, tuple[float, float]] = {
    "arcface": (0.135, 0.10),
    "lvface": (0.080, 0.10),
    "ensemble": (0.094, 0.10),
}
_CALIB_FALLBACK = (0.10, 0.15)  # model lạ → calibration trung tính

# Trọng số fusion mặc định: ưu tiên LVFace vì là specialist cho masked face.
DEFAULT_WEIGHTS: dict[str, float] = {"arcface": 0.4, "lvface": 0.6}


def calibrate_cosine(sim: np.ndarray | float, model_name: str) -> np.ndarray:
    """Map cosine thô → confidence [0,1] theo calibration tĩnh của model.

    Args:
        sim:        cosine similarity (scalar hoặc array bất kỳ shape).
        model_name: key trong CALIB ("arcface" | "lvface" | "ensemble").

    Returns:
        Mảng cùng shape, giá trị ∈ (0, 1). = 0.5 tại center của model.
    """
    center, scale = CALIB.get(model_name, _CALIB_FALLBACK)
    x = (np.asarray(sim, dtype=np.float64) - center) / scale
    return 1.0 / (1.0 + np.exp(-x))


def weighted_fuse(
    sims: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Calibrate từng model rồi weighted-average thành 1 điểm fused ∈ [0,1].

    Args:
        sims:    {model_name: cosine_array}. Mọi array phải cùng shape.
        weights: {model_name: weight}. None → DEFAULT_WEIGHTS, lọc theo các
                 model thực có trong `sims` rồi renormalize về tổng = 1.

    Returns:
        Mảng fused cùng shape với các array đầu vào, giá trị ∈ (0, 1).
    """
    if not sims:
        raise ValueError("sims rỗng — cần ít nhất 1 model.")
    weights = weights or DEFAULT_WEIGHTS
    # Chỉ giữ weight của model thực có; model thiếu weight → mặc định 1.0.
    w = {name: weights.get(name, 1.0) for name in sims}
    wsum = sum(w.values())
    if wsum <= 0:
        raise ValueError(f"Tổng weight không dương: {w}")
    fused = None
    for name, sim in sims.items():
        cal = calibrate_cosine(sim, name) * (w[name] / wsum)
        fused = cal if fused is None else fused + cal
    return fused


def verdict_band(confidence: float, lo: float = 0.45, hi: float = 0.65) -> str:
    """Phân loại confidence fused thành 3 mức cho demo (thay % gây hiểu nhầm).

    confidence là output calibrated ∈ [0,1] (0.5 ≈ ranh giới genuine/impostor).
    """
    if confidence >= hi:
        return "✅ Khớp mạnh"
    if confidence >= lo:
        return "⚠️ Biên — cần xác minh thêm"
    return "❌ Không khớp"
