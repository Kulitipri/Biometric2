"""Identification + Verification metrics.

Identification (1:N): Rank-K accuracy, CMC curve.
Verification (1:1):   ROC, TAR@FAR, EER.
"""

from __future__ import annotations

import numpy as np


# --- Identification ---------------------------------------------------------

def rank_k_accuracy(
    sim_matrix: np.ndarray,        # shape: (n_probe, n_gallery)
    probe_ids: list[str],
    gallery_ids: list[str],
    k: int = 1,
) -> float:
    """Fraction of probes whose correct identity falls in top-K of gallery."""
    assert sim_matrix.shape == (len(probe_ids), len(gallery_ids))
    # argsort desc → top-k indices per probe.
    top_k_idx = np.argsort(-sim_matrix, axis=1)[:, :k]  # (n_probe, k)
    gallery_arr = np.asarray(gallery_ids)
    probe_arr = np.asarray(probe_ids)
    top_k_ids = gallery_arr[top_k_idx]                  # (n_probe, k)
    hits = (top_k_ids == probe_arr[:, None]).any(axis=1)
    return float(hits.mean())


def cmc_curve(
    sim_matrix: np.ndarray,
    probe_ids: list[str],
    gallery_ids: list[str],
    max_k: int | None = None,
) -> np.ndarray:
    """Return rank-k accuracy for k=1..max_k. shape: (max_k,)."""
    max_k = max_k or len(gallery_ids)
    max_k = min(max_k, len(gallery_ids))
    sorted_idx = np.argsort(-sim_matrix, axis=1)        # (n_probe, n_gallery)
    gallery_arr = np.asarray(gallery_ids)
    probe_arr = np.asarray(probe_ids)
    sorted_ids = gallery_arr[sorted_idx[:, :max_k]]     # (n_probe, max_k)
    correct = sorted_ids == probe_arr[:, None]          # (n_probe, max_k)
    # CMC[k] = fraction of probes correct at any rank <= k.
    cum = np.cumsum(correct, axis=1) > 0
    return cum.mean(axis=0).astype(np.float64)


# --- Verification -----------------------------------------------------------

def roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fars, tars, thresholds) — ascending threshold."""
    from sklearn.metrics import roc_curve as sk_roc
    fpr, tpr, thresh = sk_roc(labels, scores)
    return fpr, tpr, thresh


def tar_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> float:
    """True-accept rate at a given false-accept rate."""
    fars, tars, _ = roc_curve(scores, labels)
    # fars ascending; tìm điểm fars <= target_far cao nhất.
    valid = fars <= target_far
    if not valid.any():
        return 0.0
    return float(tars[valid].max())


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Returns (eer, threshold_at_eer)."""
    fars, tars, thresh = roc_curve(scores, labels)
    frrs = 1 - tars
    # EER là điểm fars ≈ frrs.
    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2)
    return eer, float(thresh[idx])
