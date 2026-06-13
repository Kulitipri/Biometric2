"""Sinh masked LFW bằng MaskTheFace.

Tiền điều kiện (1 lần):
    # 1. Clone MaskTheFace vào third_party/
    git clone https://github.com/aqeelanwar/MaskTheFace.git third_party/MaskTheFace

    # 2. Cài deps cho MaskTheFace (trong env face_rec đang active):
    #    Windows + Python 3.10: dùng dlib-bin (prebuilt) thay vì dlib (cần MSVC).
    #    face_recognition có hard-dep trên 'dlib' nên phải --no-deps để skip:
    pip install dlib-bin
    pip install face_recognition --no-deps
    pip install face_recognition_models dotmap imutils

Usage:
    python scripts/make_masked_lfw.py
    python scripts/make_masked_lfw.py --mask-type cloth   # surgical | cloth | N95 | KN95

Output:
    data/lfw/masked/<identity>/<image>_<mask_type>.jpg
    (đúng path mà LFWMaskedDataset đang đọc)

Mất ~30-60 phút cho 13K ảnh LFW. Anh chạy 1 lần là xong cho cả project.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFW_RAW = PROJECT_ROOT / "data" / "lfw" / "lfw-deepfunneled"
LFW_MASKED = PROJECT_ROOT / "data" / "lfw" / "masked"
MASKTHEFACE = PROJECT_ROOT / "third_party" / "MaskTheFace"
# MaskTheFace tạo sibling folder với suffix "_masked" cạnh input.
MTF_OUT = LFW_RAW.with_name(LFW_RAW.name + "_masked")

REQUIRED_DEPS = ["dlib", "face_recognition", "dotmap", "imutils"]


def fail(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def check_input() -> None:
    if not LFW_RAW.exists():
        fail(f"LFW input không có: {LFW_RAW}\n"
             "Download LFW deepfunneled từ https://vis-www.cs.umass.edu/lfw/")
    n = sum(1 for p in LFW_RAW.iterdir() if p.is_dir())
    print(f"[ok] Input: {n} identities trong {LFW_RAW}")
    if n < 100:
        fail(f"Chỉ thấy {n} identities — LFW có thể chưa extract đầy đủ.")


def check_masktheface() -> None:
    if not (MASKTHEFACE / "mask_the_face.py").exists():
        fail(f"MaskTheFace chưa clone vào {MASKTHEFACE}\n"
             "Clone trước:\n"
             "  git clone https://github.com/aqeelanwar/MaskTheFace.git "
             f"{MASKTHEFACE.relative_to(PROJECT_ROOT)}")
    print(f"[ok] MaskTheFace tìm thấy tại {MASKTHEFACE}")


def check_deps() -> None:
    missing = []
    for mod in REQUIRED_DEPS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"Thiếu Python packages: {missing}\n"
             "Cài bằng (Windows + Python 3.10):\n"
             "  pip install dlib-bin\n"
             "  pip install face_recognition --no-deps\n"
             "  pip install face_recognition_models dotmap imutils")
    print(f"[ok] All deps installed: {REQUIRED_DEPS}")


def run_masktheface(mask_type: str) -> None:
    print(f"\n[run] MaskTheFace (mask_type={mask_type})...")
    print(f"      Input:  {LFW_RAW}")
    print(f"      TmpOut: {MTF_OUT}")
    cmd = [
        sys.executable, "mask_the_face.py",
        "--path", str(LFW_RAW),
        "--mask_type", mask_type,
    ]
    rc = subprocess.run(cmd, cwd=MASKTHEFACE).returncode
    if rc != 0:
        fail(f"MaskTheFace exited with code {rc}")


def finalize(mask_type: str) -> None:
    if not MTF_OUT.exists():
        fail(f"Không thấy output MaskTheFace: {MTF_OUT}")
    if LFW_MASKED.exists():
        print(f"[warn] {LFW_MASKED} đã có — xoá trước khi rename.")
        shutil.rmtree(LFW_MASKED)
    print(f"[mv]  {MTF_OUT.name} → {LFW_MASKED.name}")
    MTF_OUT.rename(LFW_MASKED)
    n = sum(1 for _ in LFW_MASKED.rglob(f"*_{mask_type}.jpg"))
    print(f"\n[done] {n} masked images tại {LFW_MASKED}")
    print(f"       Verify config: dataset.mask_suffix='_{mask_type}'")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-type", default="surgical",
                   choices=["surgical", "cloth", "N95", "KN95"],
                   help="Loại mask (mặc định: surgical).")
    args = p.parse_args()

    check_input()
    check_masktheface()
    check_deps()
    run_masktheface(args.mask_type)
    finalize(args.mask_type)


if __name__ == "__main__":
    main()
