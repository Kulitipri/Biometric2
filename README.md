# Face Recognition with Partial Facial Visibility

Group 12 — HUST · Môn Xác thực sinh trắc học.

Mục tiêu: hệ thống face recognition vẫn nhận diện chính xác khi người dùng đeo
khẩu trang (use case: mở điện thoại khi đeo mask). Chỉ dùng **pretrained
models**, không train từ đầu, không fine-tune.

- Identification (1:N) là protocol chính, Verification (1:1) là phụ
- 3 model so sánh: FaceNet → ArcFace R100 → LVFace
- Datasets: LFW (+ MaskTheFace synthetic), RMFRD, ảnh team

Xem [CLAUDE.md](CLAUDE.md) để biết design decisions, workflow rules và pipeline
stages. Proposal đầy đủ ở `docs/Group_12_Proposal_v2.docx`.

## Quick start

```bash
conda activate face_rec
pip install -r requirements.txt

# Sanity check
python notebooks/02_pipeline_smoke_test.py

# Chạy 1 experiment (config-driven; đổi model/dataset qua --override)
python eval/run_experiment.py --config configs/default.yaml