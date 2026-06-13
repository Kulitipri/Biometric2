"""PyTorch Datasets cho LFW (raw + masked), RMFRD và MAFA.

Gallery 1-shot rule: gallery và probe KHÔNG được share ảnh — chỉ share identity.
Mỗi dataset đóng 1 vai trò riêng (per proposal §3) — không trộn lẫn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .utils import read_image_bgr


DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def _iter_images(directory: Path, allowed_extensions: tuple[str, ...]) -> list[Path]:
    """List image files in `directory` whose extension matches allowed set (case-insensitive)."""
    allowed = {e.lower() for e in allowed_extensions}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )


def _normalize_extensions(value) -> tuple[str, ...]:
    """Accept None / list / tuple → tuple of lowercased extensions with leading dot."""
    if value is None:
        return DEFAULT_IMAGE_EXTENSIONS
    out: list[str] = []
    for ext in value:
        ext = str(ext).lower()
        if not ext.startswith("."):
            ext = "." + ext
        out.append(ext)
    return tuple(out)


def _load_identity_allowlist(value) -> set[str] | None:
    """Accept None / list / path-to-textfile → set of identity names (or None).

    File format: 1 identity / dòng (dòng trống và whitespace bị bỏ qua). Dùng
    bởi scripts/build_clean_rmfrd_subset.py để giới hạn evaluation về subset
    "clean" (folder unmasked thuần một người) — per khuyến nghị label-noise.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip()}
    p = Path(value)
    if not p.is_file():
        raise FileNotFoundError(
            f"identity_allowlist trỏ tới file không tồn tại: {p}. "
            f"Chạy: python scripts/build_clean_rmfrd_subset.py ... --min-mean-intra <x>"
        )
    return {
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


class LFWMaskedDataset(Dataset):
    """LFW with synthetic masks (MaskTheFace).

    split='gallery' -> unmasked, 1 image / identity (lấy từ raw_root)
    split='probe'   -> masked, các ảnh CÒN LẠI của identity đó (lấy từ masked_root)

    Selection deterministic: identity sort theo tên, mỗi identity chọn stem
    đầu tiên (sorted) làm gallery, còn lại làm probe → cùng args = cùng split.
    """

    def __init__(
        self,
        raw_root: str | Path,
        masked_root: str | Path,
        split: str,
        gallery_size: int | None = 500,
        mask_suffix: str = "_surgical",
        probe_masked: bool = True,
        gallery_shots: int = 1,
        allowed_extensions=None,
        seed: int = 42,
        **_,
    ) -> None:
        assert split in {"gallery", "probe"}
        assert gallery_shots >= 1
        self.split = split
        self.gallery_shots = gallery_shots
        raw_root = Path(raw_root)
        masked_root = Path(masked_root)
        exts = _normalize_extensions(allowed_extensions)

        # 1. Discover identities. Khi probe_masked=True, phải lọc theo masked
        # availability vì MaskTheFace có thể skip ảnh nó không detect được.
        # Khi probe_masked=False (baseline unmasked), chỉ cần raw → không lọc.
        # Multi-shot: cần >=K+1 ảnh (K cho gallery, >=1 cho probe).
        # LFW gốc dùng .jpg; allowed_extensions cho phép mở rộng nếu data layout đổi.
        # raw_paths_by_stem giữ path thật để dùng extension đúng khi build sample.
        identity_to_data: dict[
            str, tuple[list[str], list[str], dict[str, Path], dict[str, Path]]
        ] = {}
        for id_dir in sorted(raw_root.iterdir()):
            if not id_dir.is_dir():
                continue
            raw_paths = {p.stem: p for p in _iter_images(id_dir, exts)}
            stems = sorted(raw_paths.keys())
            if len(stems) < gallery_shots + 1:
                continue
            gallery_stems = stems[:gallery_shots]
            masked_id_dir = masked_root / id_dir.name
            masked_paths_by_stem: dict[str, Path] = {}
            if probe_masked and masked_id_dir.is_dir():
                masked_paths_by_stem = {
                    p.stem: p for p in _iter_images(masked_id_dir, exts)
                }
            if probe_masked:
                probe_stems = [
                    s for s in stems[gallery_shots:]
                    if f"{s}{mask_suffix}" in masked_paths_by_stem
                ]
            else:
                probe_stems = stems[gallery_shots:]
            if probe_stems:
                identity_to_data[id_dir.name] = (
                    gallery_stems, probe_stems, raw_paths, masked_paths_by_stem
                )

        # 2. Limit theo gallery_size (sorted by name → deterministic).
        identities = sorted(identity_to_data.keys())
        if gallery_size is not None:
            identities = identities[:gallery_size]

        # 3. Build samples. Gallery: K ảnh/identity (cùng label); aggregate sẽ
        # được làm ở downstream (eval/run_experiment.py) sau khi embed.
        self.samples: list[tuple[Path, str]] = []
        for identity in identities:
            gallery_stems, probe_stems, raw_paths, masked_paths = identity_to_data[identity]
            if split == "gallery":
                for stem in gallery_stems:
                    self.samples.append((raw_paths[stem], identity))
            else:  # probe
                for stem in probe_stems:
                    if probe_masked:
                        path = masked_paths[f"{stem}{mask_suffix}"]
                    else:
                        path = raw_paths[stem]
                    self.samples.append((path, identity))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        path, identity = self.samples[idx]
        return read_image_bgr(path), identity


class RMFRDDataset(Dataset):
    """Real-World Masked Face Recognition Dataset.

    Structure (sau khi unzip):
        <root>/AFDB_face_dataset/<identity>/*.jpg          # unmasked
        <root>/AFDB_masked_face_dataset/<identity>/*.jpg   # masked

    Same gallery/probe contract như LFWMaskedDataset:
        gallery: 1 unmasked / identity (stem sorted đầu tiên)
        probe:   tất cả masked của identity đó

    Lọc identity: chỉ giữ ID có mặt ở CẢ 2 folder và có ≥1 ảnh ở mỗi bên.
    """

    UNMASKED_DIR = "AFDB_face_dataset"
    MASKED_DIR = "AFDB_masked_face_dataset"
    SYNTH_MASKED_GALLERY_DIR = "AFDB_face_masked_gallery"

    def __init__(
        self,
        root: str | Path,
        split: str,
        gallery_size: int | None = None,
        gallery_shots: int = 1,
        allowed_extensions=None,
        mask_gallery: bool = False,
        mask_gallery_suffix: str = "_surgical",
        identity_allowlist=None,
        seed: int = 42,
        **_,
    ) -> None:
        assert split in {"gallery", "probe"}
        assert gallery_shots >= 1
        self.split = split
        self.gallery_shots = gallery_shots
        self.mask_gallery = mask_gallery
        root = Path(root)
        unmasked_root = root / self.UNMASKED_DIR
        masked_root = root / self.MASKED_DIR
        synth_gallery_root = root / self.SYNTH_MASKED_GALLERY_DIR
        exts = _normalize_extensions(allowed_extensions)

        # Lever #1: nếu mask_gallery=True, gallery dùng ảnh đã synthetic-mask
        # (scripts/make_masked_rmfrd_gallery.py sinh sẵn). Probe vẫn là ảnh
        # real-world masked. Mục tiêu: kéo embedding gallery + probe về cùng
        # phân phối "có khẩu trang", thu hẹp domain gap.
        if mask_gallery and not synth_gallery_root.is_dir():
            raise FileNotFoundError(
                f"mask_gallery=True nhưng không thấy {synth_gallery_root}. "
                f"Chạy: python scripts/make_masked_rmfrd_gallery.py"
            )

        # 1. Discover identities có cả unmasked (≥K) + masked (≥1).
        # Trước đây chỉ glob "*.jpg" → bỏ sót .png/.jpeg/.JPG. Allowlist case-
        # insensitive bắt được khoảng 403 identity / 1945 masked probe thay vì 376 / 1871.
        # Khi mask_gallery=True: chỉ giữ identity có ≥K ảnh trong masked-gallery folder
        # (MaskTheFace có thể fail landmark trên ảnh tight-crop → nhiều ảnh không sinh được).
        identity_to_paths: dict[str, tuple[list[Path], list[Path]]] = {}
        for id_dir in sorted(unmasked_root.iterdir()):
            if not id_dir.is_dir():
                continue
            masked_id_dir = masked_root / id_dir.name
            if not masked_id_dir.is_dir():
                continue
            if mask_gallery:
                gallery_dir = synth_gallery_root / id_dir.name
                if not gallery_dir.is_dir():
                    continue
                gallery_imgs = [
                    p for p in _iter_images(gallery_dir, exts)
                    if mask_gallery_suffix in p.stem
                ] or _iter_images(gallery_dir, exts)
            else:
                gallery_imgs = _iter_images(id_dir, exts)
            masked_imgs = _iter_images(masked_id_dir, exts)
            if len(gallery_imgs) >= gallery_shots and masked_imgs:
                identity_to_paths[id_dir.name] = (gallery_imgs[:gallery_shots], masked_imgs)

        # 2. Limit (RMFRD nhỏ — default None = dùng tất cả ~442 ID).
        # identity_allowlist (label-noise clean subset): giữ giao identity hợp
        # lệ ∩ allowlist TRƯỚC khi cắt gallery_size, để gallery_size vẫn đếm
        # trên tập đã clean. Áp cho cả gallery + probe → so sánh fair.
        identities = sorted(identity_to_paths.keys())
        allow = _load_identity_allowlist(identity_allowlist)
        if allow is not None:
            identities = [i for i in identities if i in allow]
            if not identities:
                raise ValueError(
                    "identity_allowlist không giao với identity nào trong dataset "
                    "— kiểm tra tên trong allowlist khớp folder name."
                )
        if gallery_size is not None:
            identities = identities[:gallery_size]

        # 3. Build samples.
        self.samples: list[tuple[Path, str]] = []
        for identity in identities:
            gallery_imgs, masked_imgs = identity_to_paths[identity]
            if split == "gallery":
                for p in gallery_imgs:
                    self.samples.append((p, identity))
            else:  # probe
                for p in masked_imgs:
                    self.samples.append((p, identity))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        path, identity = self.samples[idx]
        return read_image_bgr(path), identity


class MAFADataset(Dataset):
    """MAFA subset cho Exp 1 error stratification.

    Mỗi sample trả về metadata mask_type / mask_color / pose để metrics có thể
    breakdown theo từng chiều. KHÔNG dùng cho gallery 1-shot — chỉ dùng để
    đánh giá robustness của model trên các biến thể mask trong context masked.
    """

    def __init__(self, root: str | Path, **_) -> None:
        # TODO: parse MAFA annotation file (.mat) để lấy metadata.
        raise NotImplementedError("MAFA dataset chưa làm — chỉ dùng cho Exp 1 stratified.")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        raise NotImplementedError


def build_dataset(name: str, split: str, **kwargs) -> Dataset:
    """Factory: 'lfw_masked' | 'rmfrd' | 'mafa'.

    Nhận `dataset` config block; mỗi class swallow kwargs không liên quan qua **_.
    """
    if name == "lfw_masked":
        return LFWMaskedDataset(split=split, **kwargs)
    if name == "lfw_raw":
        # Unmasked baseline: gallery + probe đều từ raw_root (cùng identity, khác stem).
        # Force probe_masked=False để override config nếu có.
        kwargs = {**kwargs, "probe_masked": False}
        return LFWMaskedDataset(split=split, **kwargs)
    if name == "rmfrd":
        return RMFRDDataset(split=split, **kwargs)  # cần kwargs['root']
    if name == "mafa":
        return MAFADataset(**kwargs)
    raise ValueError(f"Unknown dataset: {name!r}. Choose: lfw_masked | rmfrd | mafa")
