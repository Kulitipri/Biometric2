"""Cosine similarity matching against a gallery — model-agnostic."""

from __future__ import annotations

import numpy as np


class Matcher:
    """Cosine-sim matcher: 1 embedding per identity in gallery.

    Multi-shot enrollment (K>1) must aggregate (mean + L2-normalize) upstream
    via utils.aggregate_by_identity so each identity has a single prototype row.
    """

    def __init__(self, gallery_emb: np.ndarray, gallery_ids: list[str]) -> None:
        # gallery_emb shape: (N, 512), L2-normalized
        # gallery_ids: identity label for each row
        assert gallery_emb.ndim == 2 and gallery_emb.shape[1] == 512
        assert len(gallery_ids) == gallery_emb.shape[0]
        self.gallery = gallery_emb.astype(np.float32)
        self.ids = list(gallery_ids)
        self._id_to_idx = {name: i for i, name in enumerate(self.ids)}

    def score(self, probe_emb: np.ndarray) -> np.ndarray:
        """probe_emb shape: (512,) → (N,); hoặc (M, 512) → (M, N).
        Cosine == dot product vì cả 2 phía đã L2-normalized.
        """
        return probe_emb @ self.gallery.T

    def top_k(self, probe_emb: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return top-K (identity, similarity) sorted desc."""
        assert probe_emb.ndim == 1, "top_k expects single probe shape (512,)"
        sims = self.score(probe_emb)
        idx = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def verify(self, probe_emb: np.ndarray, claimed_id: str, threshold: float) -> bool:
        """1:1 protocol — compare probe vs the gallery entry of claimed_id."""
        if claimed_id not in self._id_to_idx:
            raise KeyError(f"Identity not in gallery: {claimed_id!r}")
        idx = self._id_to_idx[claimed_id]
        sim = float(probe_emb @ self.gallery[idx])
        return sim >= threshold
