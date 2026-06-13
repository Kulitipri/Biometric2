#!/usr/bin/env bash
# Stage 1 entry point: hướng dẫn tải LFW + RMFRD + MAFA, sinh masked LFW.
# Run từ project root: bash scripts/download_data.sh
#
# Trên Windows: script này chỉ in hướng dẫn. Bước sinh masked LFW có Python
# wrapper riêng: `python scripts/make_masked_lfw.py`.
set -euo pipefail

cat <<'EOF'
===========================================================================
Stage 1 — Data acquisition (manual, vì các dataset cần đăng ký/download
từ nguồn riêng, không có direct link tự động được).

(1) LFW deepfunneled  → data/lfw/lfw-deepfunneled/
    https://vis-www.cs.umass.edu/lfw/
    Tải lfw-deepfunneled.tgz, extract vào data/lfw/.

(2) Masked LFW (sinh từ LFW bằng MaskTheFace) → data/lfw/masked/
    a. git clone https://github.com/aqeelanwar/MaskTheFace.git third_party/MaskTheFace
    b. Windows + Python 3.10:
         pip install dlib-bin
         pip install face_recognition --no-deps
         pip install face_recognition_models dotmap
       Linux/macOS:
         pip install dlib face_recognition dotmap
    c. python scripts/make_masked_lfw.py              # wrapper Python tự động

(3) RMFRD (Real-World Masked Face Dataset) → data/rmfrd/
    https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
    Tải RMFD subset (~1.4GB), extract vào data/rmfrd/.

(4) MAFA subset (stress test mask type/color/pose) → data/mafa/
    https://drive.google.com/.../MAFA  (proposal §3 references)
    Tải, extract vào data/mafa/.

(5) Team photos (demo) → data/team_photos/<name>/{unmasked,masked_NN}.jpg
    Tự chụp 4 thành viên team.
===========================================================================
EOF
