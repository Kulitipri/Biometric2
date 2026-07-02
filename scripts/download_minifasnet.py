"""Tải weights MiniFASNet anti-spoofing (ONNX) về weights/minifasnet.onnx.

Chạy 1 lần để lấy model cho src/liveness.py:
    conda activate face_rec
    pip install huggingface_hub        # nếu chưa có
    python scripts/download_minifasnet.py

Nguồn: garciafido/minifasnet-v2-anti-spoofing-onnx (MiniFASNetV2 2.7_80x80).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = "garciafido/minifasnet-v2-anti-spoofing-onnx"
DST = Path("weights/minifasnet.onnx")


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        sys.exit("Thiếu thư viện. Chạy:  pip install huggingface_hub")

    files = list_repo_files(REPO)
    print("Files trong repo:", files)

    onnx = [f for f in files if f.endswith(".onnx")]
    if not onnx:
        sys.exit(f"Không thấy file .onnx nào trong {REPO}.")

    DST.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(REPO, onnx[0])
    shutil.copy(cached, DST)
    print(f"saved -> {DST}  ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
