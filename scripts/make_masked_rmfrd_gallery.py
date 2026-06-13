"""Sinh masked-gallery cho RMFRD bằng MaskTheFace — Lever #1 (mask gallery).

Lý do: RMFRD probe là ảnh masked thật, gallery là unmasked. Embedding gallery
và probe nằm ở 2 phân phối khác nhau → ArcFace/LVFace phải nội suy qua
~40% diện tích bị che. Pre-mask gallery (synthetic surgical mask) đẩy 2 phân
phối về cùng không gian "có khẩu trang", thường +5-10pp Rank-1 trên RMFRD
(LVFace paper §4.3, MTArcFace).

Output:
    data/rmfrd/self-built-masked-face-recognition-dataset/AFDB_face_masked_gallery/
        <identity>/<image>_surgical.jpg

Sample size nhỏ: ~150x140 px. MaskTheFace có thể fail landmark trên ảnh quá
tight-crop → script log danh sách ảnh skip + tỉ lệ thành công per-identity.
Identity có 0 ảnh masked được sinh sẽ rớt khỏi gallery khi load với
`mask_gallery=true`.

Usage:
    python scripts/make_masked_rmfrd_gallery.py
    python scripts/make_masked_rmfrd_gallery.py --mask-type surgical --limit 10
        # limit=10 → smoke test 10 identity đầu trước khi chạy full

Pre-requisites (giống make_masked_lfw.py):
    pip install dlib-bin
    pip install face_recognition --no-deps
    pip install face_recognition_models dotmap imutils
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RMFRD_ROOT = PROJECT_ROOT / "data" / "rmfrd" / "self-built-masked-face-recognition-dataset"
UNMASKED_DIR = RMFRD_ROOT / "AFDB_face_dataset"
MASKED_GALLERY_DIR = RMFRD_ROOT / "AFDB_face_masked_gallery"
MASKTHEFACE = PROJECT_ROOT / "third_party" / "MaskTheFace"

REQUIRED_DEPS = ["dlib", "face_recognition", "dotmap", "imutils"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def fail(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def check_input() -> int:
    if not UNMASKED_DIR.exists():
        fail(f"RMFRD unmasked không có: {UNMASKED_DIR}")
    n = sum(1 for p in UNMASKED_DIR.iterdir() if p.is_dir())
    print(f"[ok] Input: {n} identities tại {UNMASKED_DIR}")
    if n < 100:
        fail(f"Chỉ thấy {n} identities — RMFRD có thể chưa extract đầy đủ.")
    return n


def check_masktheface() -> None:
    if not (MASKTHEFACE / "mask_the_face.py").exists():
        fail(f"MaskTheFace chưa clone vào {MASKTHEFACE}")
    print(f"[ok] MaskTheFace: {MASKTHEFACE}")


def check_deps() -> None:
    missing = [m for m in REQUIRED_DEPS if not _can_import(m)]
    if missing:
        fail(f"Thiếu deps: {missing}\nXem header của file để biết cách cài.")
    print(f"[ok] Deps: {REQUIRED_DEPS}")


def _can_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def run_masktheface_on_identity(
    id_dir: Path, mask_type: str, tmp_root: Path
) -> tuple[int, int]:
    """Run MaskTheFace on a single identity dir → return (n_input, n_output).

    MaskTheFace tạo sibling folder với suffix '_masked' cạnh input. Ta dùng
    tmp staging dir để mỗi identity được copy in/out rõ ràng, tránh collide
    khi chạy parallel/resume.
    """
    n_input = sum(
        1 for p in id_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if n_input == 0:
        return 0, 0

    # Stage: copy identity images vào tmp/<id>/<files>; MaskTheFace sẽ tạo
    # tmp/<id>_masked/<files>_<mask_type>.<ext>.
    staged = tmp_root / id_dir.name
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True, exist_ok=True)
    for src in id_dir.iterdir():
        if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(src, staged / src.name)

    cmd = [
        sys.executable, "mask_the_face.py",
        "--path", str(staged),
        "--mask_type", mask_type,
        "--verbose",
    ]
    # MaskTheFace prints landmark failures; we capture stdout to keep terminal clean.
    proc = subprocess.run(cmd, cwd=MASKTHEFACE, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[warn] {id_dir.name}: MaskTheFace exit={proc.returncode}")
        print(proc.stderr[-500:] if proc.stderr else "(no stderr)")

    masked_tmp = tmp_root / f"{id_dir.name}_masked"
    n_output = 0
    if masked_tmp.exists():
        dest = MASKED_GALLERY_DIR / id_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for src in masked_tmp.iterdir():
            if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
                shutil.copy2(src, dest / src.name)
                n_output += 1
    return n_input, n_output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-type", default="surgical",
                   choices=["surgical", "cloth", "N95", "KN95"])
    p.add_argument("--limit", type=int, default=0,
                   help="Chỉ chạy N identity đầu (smoke test). 0 = tất cả.")
    p.add_argument("--force", action="store_true",
                   help="Xoá masked-gallery cũ và sinh lại từ đầu.")
    args = p.parse_args()

    check_input()
    check_masktheface()
    check_deps()

    if MASKED_GALLERY_DIR.exists():
        if args.force:
            print(f"[clean] Xoá {MASKED_GALLERY_DIR} (force=true)")
            shutil.rmtree(MASKED_GALLERY_DIR)
        else:
            existing = sum(1 for p in MASKED_GALLERY_DIR.iterdir() if p.is_dir())
            print(f"[info] {MASKED_GALLERY_DIR} đã có {existing} identity → "
                  f"resume mode (chỉ xử lý identity chưa có folder).")
    MASKED_GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    identities = sorted(p for p in UNMASKED_DIR.iterdir() if p.is_dir())
    if args.limit > 0:
        identities = identities[:args.limit]
    print(f"[run] {len(identities)} identities ({'limit smoke' if args.limit else 'full'})")

    total_in = total_out = n_zero = 0
    zero_identities: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mtf_rmfrd_") as tmp:
        tmp_root = Path(tmp)
        for i, id_dir in enumerate(identities, 1):
            existing_out = MASKED_GALLERY_DIR / id_dir.name
            if not args.force and existing_out.exists() and any(existing_out.iterdir()):
                continue
            n_in, n_out = run_masktheface_on_identity(id_dir, args.mask_type, tmp_root)
            total_in += n_in
            total_out += n_out
            if n_out == 0:
                n_zero += 1
                zero_identities.append(id_dir.name)
            if i % 20 == 0 or i == len(identities):
                pct = total_out / total_in if total_in else 0
                print(f"  [{i:4d}/{len(identities)}] in={total_in} out={total_out} "
                      f"({pct:.1%}) zero_id={n_zero}")

    print(f"\n[done] Total input={total_in}, masked output={total_out} "
          f"({total_out/total_in:.1%} success)" if total_in else "[done] 0 input")
    print(f"       Output dir: {MASKED_GALLERY_DIR}")
    print(f"       Identities with 0 masked: {n_zero}")
    if zero_identities and len(zero_identities) <= 30:
        print(f"       Failed list: {zero_identities}")

    print("\nNext: bật cờ dataset trong config:")
    print("  dataset.mask_gallery: true")
    print("  dataset.mask_gallery_suffix: '_{}'".format(args.mask_type))


if __name__ == "__main__":
    main()
