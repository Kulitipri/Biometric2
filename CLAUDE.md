# Face Recognition with Partial Facial Visibility

## Project Overview (WHAT)

Đây là dự án môn **Xác thực sinh trắc học** của Group 12 (HUST). Mục tiêu: xây dựng và đánh giá hệ thống face recognition vẫn nhận diện chính xác khi người dùng đeo khẩu trang — use case chính là **mở điện thoại khi đeo mask**.

Bài toán cụ thể:
- **Identification (1:N)** là protocol chính, **Verification (1:1)** là protocol phụ
- Chỉ tập trung vào **masked face occlusion** (không xét kính, mũ, pose như occlusion độc lập — head pose chỉ được phân tích trong context masked qua MAFA)
- Chỉ dùng **pretrained models**, KHÔNG train từ đầu, KHÔNG fine-tune

Chi tiết đầy đủ trong file proposal: @docs/Group_12_Proposal_v2.docx

## Tech Stack (WHAT)

- **Python 3.10** + Conda env tên `face_rec`
- **PyTorch** (CUDA 12.1) cho FaceNet
- **ONNX Runtime GPU** cho ArcFace và LVFace
- **InsightFace** library cho RetinaFace detector và alignment 112×112
- **Gradio** cho webcam demo (không dùng Tkinter)
- **scikit-learn** + **matplotlib** cho metrics và biểu đồ

## Project Structure (WHAT)

```
face-recognition-masked/
├── CLAUDE.md                 # File này
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml          # Tham số experiment (model, dataset, matcher...)
│   └── <exp>.yaml            # Các config experiment/ablation (rmfrd_*, ensemble_*)
├── data/
│   ├── lfw/{raw,masked}/     # LFW gốc + LFW sinh mask bằng MaskTheFace
│   ├── rmfrd/                # Real-World Masked Face Dataset
│   ├── mafa/                 # MAFA subset — stress test mask type/color/pose
│   └── team_photos/          # 4 thành viên cho demo
├── src/
│   ├── config.py             # YAML loader + CLI override
│   ├── detector.py           # Face detection + alignment (RetinaFace)
│   ├── embedder.py           # Model wrappers: FaceNet/ArcFace/LVFace
│   ├── matcher.py            # Cosine similarity, top-K, threshold
│   ├── fusion.py             # Score-level fusion cho ensemble (calibrate + weight)
│   ├── restorer.py           # CodeFormer face restoration (Lever A, optional)
│   ├── dataset.py            # PyTorch Dataset cho LFW/RMFRD/MAFA
│   └── utils.py              # Common helpers
├── eval/
│   ├── metrics.py            # Rank-K, CMC, ROC, TAR@FAR, EER
│   └── run_experiment.py     # CLI để chạy 1 (model, dataset) combo
├── scripts/                  # Data prep + diagnostics (mask gen, label-noise audit...)
├── experiments/              # JSON kết quả + biểu đồ (gitignored)
├── notebooks/                # Sanity checks, visualization
├── demo/
│   ├── verify_app.py         # Gradio 1:1 verification
│   └── webcam_app.py         # Gradio 1:N webcam realtime (enroll K-shot + identify + open-set reject)
└── weights/                  # Pretrained ONNX/pth files (gitignored)
```

## Core Design Decisions (WHY)

Đây là các quyết định đã chốt, **KHÔNG được tự ý đổi** mà không hỏi:

1. **3 model so sánh:** FaceNet (baseline yếu, VGGFace2) → ArcFace R100 (baseline mạnh, Glint360K) → LVFace (specialist, ICCV 2025 MFR challenge #1). Bộ 3 này kể được câu chuyện degradation rõ ràng.
2. **Tất cả output embedding phải là 512-d L2-normalized.** Điều này cho phép code `matcher.py` dùng chung cho mọi model.
3. **Preprocessing chuẩn ArcFace cho mọi model:** RetinaFace detect → affine 5-landmark → **112×112 BGR**, normalize về **[-1, 1]**. Đây là input format ArcFace yêu cầu và áp cho cả 3 model để fair comparison.
4. **Gallery K-shot (default K=1, baseline):** mỗi identity có K ảnh unmasked trong gallery; probe là ảnh masked KHÁC của cùng người. Không reuse ảnh giữa gallery và probe. **K=1** dùng cho headline baseline (so sánh fair với literature). **K=3-5** được phép cho improvement experiments (multi-shot enrollment — sát use case phone unlock thực tế); khi K>1, embeddings của K ảnh được mean + L2-renormalize thành 1 prototype/identity (matcher code không đổi). Bật qua `dataset.gallery_shots` trong config.
5. **Headline experiment dùng 500-identity subset của LFW** (per proposal §8 risk mitigation). Không scale lên full LFW trừ khi đã chạy xong toàn bộ experiment chính.
6. **4 dataset, mỗi cái 1 vai trò riêng** (per proposal §3): LFW+synthetic = main controlled test, RMFRD = real-world validation, MAFA = mask-style diversity, team_photos = demo. KHÔNG trộn lẫn để tránh leak.

## Workflow Rules (HOW)

### Code style
- Dùng **type hints** cho mọi function public (`def embed(self, img: np.ndarray) -> np.ndarray:`)
- Docstring kiểu Google style, ngắn gọn
- Mọi numpy array phải comment rõ shape: `# shape: (N, 512), L2-normalized`
- File code không quá 300 dòng — nếu dài hơn, tách module

### Git workflow
- Commit message tiếng Anh, format: `[module] action: detail` (ví dụ: `[embedder] add LVFace ONNX wrapper`)
- KHÔNG commit file weights, dataset, hoặc `experiments/*.json` chứa kết quả lớn
- KHÔNG push thẳng vào `main`, dùng PR

### Testing & verification
- Sau mỗi module mới, **YOU MUST** chạy sanity check: load 1 ảnh team member unmasked + masked → cosine similarity phải > 0.4
- Trước khi report kết quả Rank-1, **YOU MUST** verify gallery và probe không leak (không chứa cùng ảnh)
- Khi đo latency: warm-up 10 lần trước, sau đó mới đo trên 1000 ảnh

### Khi viết code mới
- **IMPORTANT:** Trước khi tạo file mới, đọc file tương tự đã có để giữ pattern nhất quán
- Không tự ý thêm dependency mới vào `requirements.txt` — hỏi trước
- Hàm helper dùng chung phải vào `src/utils.py`, không copy-paste

## Pipeline Stages (HOW)

Project chia thành 5 stage, mỗi stage có entry point rõ:

| Stage | Tuần | Entry point | Output |
|---|---|---|---|
| 1. Setup & data | 1 | `scripts/download_data.sh` | `data/` populated |
| 2. Core pipeline | 2 | `src/{detector,embedder,matcher}.py` | Importable modules |
| 3. Evaluation | 3 | `eval/run_experiment.py` | Metrics framework |
| 4. Experiments | 4 | `eval/run_experiment.py --config configs/<exp>.yaml` | `experiments/*.json` + plots |
| 5. Demo + report | 5–6 | `demo/verify_app.py` | Gradio app + báo cáo |

## Communication Preferences (HOW)

Khi tôi (Khải) làm việc với Claude trên project này:
- Trả lời **bằng tiếng Việt** mặc định; code và technical terms giữ tiếng Anh
- Khi viết code mới: scaffold trước (skeleton + comments), confirm rồi mới fill in từng phần. KHÔNG dump cả file dài
- Khi tôi hỏi khái niệm mới: giải thích từ gốc, dùng analogy. KHÔNG chỉ copy đáp án
- Khi tôi báo bug: hỏi 1-2 câu làm rõ trước khi sửa, không đoán
- **IMPORTANT:** Ưu tiên giúp tôi hiểu bản chất hơn là đẩy code ra nhanh — tôi đang học, không phải đang ship product

## Common Commands

```bash
# Activate env
conda activate face_rec

# Run a single experiment (config-driven)
python eval/run_experiment.py --config configs/default.yaml
python eval/run_experiment.py --override model.name=lvface dataset.name=rmfrd

# Sanity check pipeline với 1 ảnh
python notebooks/02_pipeline_smoke_test.py

# Demo 1:1 verification (Gradio)
python demo/verify_app.py

# Generate synthetic masks
cd third_party/MaskTheFace && python mask_the_face.py --path ../../data/lfw/raw --mask_type surgical
```

## Reference Documents

- **Project proposal (v2):** @docs/Group_12_Proposal_v2.docx — đọc khi cần context về methodology, dataset strategy, expected outcomes
- **MaskTheFace tool:** https://github.com/aqeelanwar/MaskTheFace
- **InsightFace docs:** https://github.com/deepinsight/insightface
- **LVFace (SOTA masked):** https://github.com/bytedance/LVFace

## Out of Scope (tránh làm)

- Train model từ đầu hoặc fine-tune (đã quyết định pretrained-only)
- Xử lý kính, mũ, pose occlusion (chỉ mask)
- GAN-based unmasking approaches (quá phức tạp cho thời lượng môn)
- Mobile deployment (chỉ demo trên laptop)
