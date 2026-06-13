# Báo cáo tiến độ — Face Recognition with Partial Facial Visibility

**Group 12 · HUST · Môn Xác thực sinh trắc học**
**Ngày báo cáo:** 2026-05-23

---

## 1. Tổng quan

Project xây dựng hệ thống face recognition cho người đeo khẩu trang (use case: mở
điện thoại khi đeo mask), chỉ dùng pretrained models, không train/fine-tune.
Tính đến hôm nay, **các stage 1–4 trong pipeline đã hoàn thành ở mức chạy được
end-to-end**, có kết quả định lượng trên cả LFW (controlled) và RMFRD
(real-world). Stage 5 (demo + báo cáo cuối) còn thiếu.

| Stage | Mô tả | Trạng thái |
|---|---|---|
| 1 | Setup môi trường + chuẩn bị dữ liệu | ✅ Done |
| 2 | Core pipeline (detector / embedder / matcher) | ✅ Done (trừ FaceNet) |
| 3 | Evaluation framework | ✅ Done |
| 4 | Experiments + ablations | ✅ Done (LFW + RMFRD) |
| 5 | Demo Gradio + báo cáo cuối | ⏳ Chưa bắt đầu |

---

## 2. Phương pháp tổng thể

### 2.1 Bài toán

- **Input:** ảnh khuôn mặt có thể đeo khẩu trang (occlusion che ~50% diện tích mặt: mũi + miệng + cằm).
- **Output 1 — Identification (1:N, protocol chính):** với 1 probe masked, tìm identity khớp nhất trong gallery có N identities đã enroll (unmasked). Metric chính: **Rank-1, Rank-5, CMC**.
- **Output 2 — Verification (1:1, protocol phụ):** cho 1 cặp (probe, claim_id), trả lời "cùng người không?". Metric: **EER, TAR @ FAR ∈ {0.001, 0.01, 0.1}**.
- **Ràng buộc thiết kế:** chỉ dùng pretrained models — không train từ đầu, không fine-tune.

### 2.2 Pipeline tổng thể

```text
                                  ┌─── Gallery (unmasked) ─┐
   Ảnh đầu vào                    │   K ảnh / identity     │
        │                         └──────────┬─────────────┘
        ▼                                    ▼
  ┌────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐
  │ RetinaFace │───▶│   Affine     │───▶│ Embedder │───▶│   Matcher    │
  │  detector  │    │ alignment    │    │ (3 models)│    │  cosine sim  │
  │ + 5 landmk │    │  112×112     │    │ → 512-d   │    │  top-K + thr │
  └────────────┘    └──────────────┘    │ L2-normed │    └──────┬───────┘
                                         └─────┬─────┘           │
                                               │                 ▼
                                               │           ┌──────────┐
                                               └──────────▶│ Decision │
                                                           │ ID + sim │
                                                           └──────────┘
```

4 giai đoạn xử lý 1 ảnh probe:

1. **Detection** — RetinaFace (InsightFace) phát hiện mặt + 5 landmark (2 mắt, mũi, 2 khóe miệng). Có pad ảnh `BORDER_REFLECT` trước detect (Lever #4a) cho ảnh tight-crop như RMFRD.
2. **Alignment** — Affine transform từ 5 landmark về kích thước chuẩn **112×112 BGR**. Nếu detect fail, có thể `fallback_align_mode='resize'` để vẫn embed được (trade quality lấy coverage).
3. **Embedding** — Đưa qua 1 trong 3 backbone (hoặc ensemble), output vector **512-d, L2-normalized**. Tuỳ chọn TTA: forward thêm horizontal-flip rồi mean + renorm.
4. **Matching** — Tính cosine similarity giữa probe embedding với toàn bộ gallery, lấy top-K. Verification dùng threshold cố định (tuned ở EER point).

### 2.3 Lựa chọn 3 model — câu chuyện degradation

| Model | Training data | Vai trò | Lý do chọn |
| --- | --- | --- | --- |
| **FaceNet** (Inception-Resnet V1) | VGGFace2 | Baseline yếu | Model phổ biến trước ArcMargin, chưa tối ưu cho occlusion |
| **ArcFace R100** | Glint360K | Baseline mạnh | SOTA general-purpose, ArcMargin loss, được dùng làm chuẩn so sánh |
| **LVFace** | (per ByteDance) | Specialist masked | Top-1 ICCV 2025 MFR challenge — kỳ vọng nhỉnh hơn trên masked |

Bộ 3 này kể được câu chuyện rõ: từ baseline cũ → baseline mạnh general → specialist cho masked. Cộng thêm **Ensemble (ArcFace + LVFace, score-level mean fusion)** làm phương án nâng cao.

### 2.4 Preprocessing chuẩn ArcFace cho mọi model

Tất cả model nhận input cùng định dạng để đảm bảo so sánh **fair**:

- Crop 112×112 sau affine theo 5 landmark
- Normalize về **[-1, 1]**: `(x - 127.5) / 127.5`
- **ArcFace:** input là BGR (giữ thứ tự kênh OpenCV)
- **LVFace:** input là RGB (đặc thù của bytedance/LVFace, đã được verify từ code gốc) — đây là điểm đã từng gây bug

→ Cùng 1 ảnh aligned, 3 model chỉ khác nhau ở backbone, không ở chất lượng input.

### 2.5 Chiến lược dữ liệu — 4 dataset, 4 vai trò

| Dataset | Vai trò | Phân vai trong câu chuyện |
|---|---|---|
| **LFW raw** | Unmasked baseline | Trần lý thuyết (ceiling) khi không có occlusion |
| **LFW + MaskTheFace** | Main controlled test | Cùng identity, cùng pose, **chỉ khác mask** → cô lập tác động của occlusion |
| **RMFRD** | Real-world validation | Mask thật, ánh sáng/pose thật → kiểm chứng generalization từ synthetic |
| **MAFA** | Stress test mask diversity | Mask đa dạng kiểu/màu + pose lớn |
| **team_photos** | Demo qualitative | Webcam demo cuối kỳ |

**Nguyên tắc tách bạch:** mỗi dataset 1 mục đích, không trộn lẫn để tránh data leak.

### 2.6 Protocol enrollment & evaluation

- **K-shot gallery:** mỗi identity có K ảnh unmasked trong gallery; probe là ảnh masked **khác** của cùng người (không reuse).
  - **K = 1:** baseline so sánh với literature.
  - **K = 3 / 5:** sát use case phone unlock thực tế (enroll nhiều ảnh khi setup).
  - Khi K > 1: K embedding được **mean + L2-renormalize** thành **1 prototype/identity** → matcher giữ nguyên code.
- **Headline subset:** 500-identity LFW (per proposal §8) — đủ thống kê, scan ablation nhanh.
- **Hai flavor Rank-K** cho RMFRD: `*_encoded` (chỉ tính trên probe encode được, fair với literature) và `*_all_probes` (skip = miss, headline metric — không giấu lỗi pipeline).

### 2.7 Levers cải thiện cho RMFRD (real-world)

Vì RMFRD là ảnh thật, tight-crop và có nhãn nhiễu, project có **6 lever** có thể bật/tắt độc lập qua YAML config để đo đóng góp từng cái:

| # | Lever | Mục đích |
|---|---|---|
| 1 | `mask_gallery` | Gallery cũng là masked (sinh bằng MaskTheFace) → gallery + probe cùng phân phối |
| 2 | `quality_weighted` | K-shot aggregate dùng weight = `det_score × Laplacian sharpness` thay cho uniform mean |
| 3 | `gallery_shots` (K=3/5) | Multi-shot enrollment |
| 4a | `pad_ratio` | Pad `BORDER_REFLECT` ảnh trước detect → RetinaFace có context để định vị landmark |
| 4b | `fallback_align_mode='resize'` | Detect fail vẫn embed bằng resize → tăng coverage |
| 5 | `tta` | Test-time augmentation (original + horizontal flip) |
| 6 | `ensemble` | Score-level mean fusion ArcFace + LVFace |

Ablation được chạy bằng các config `configs/rmfrd_v2_only_*.yaml` để cô lập đóng góp của từng lever.

### 2.8 Sanity checks bắt buộc (per CLAUDE.md)

- Sau mỗi module mới: load 1 ảnh team unmasked + masked → cosine similarity phải **> 0.4**.
- Trước khi report Rank-1: verify gallery và probe không reuse cùng ảnh.
- Đo latency: warm-up 10 forward pass trước, sau đó mới đo trên 1000 ảnh.

---

## 3. Stage 1 — Setup & Data

### Môi trường
- Conda env `face_rec` (Python 3.10) — ✅
- `requirements.txt` đầy đủ — ✅
- Weights pretrained (đã tải về [weights/](../weights/)): — ✅
  - [arcface_r100.onnx](../weights/arcface_r100.onnx)
  - [glintr100.onnx](../weights/glintr100.onnx)
  - [lvface.onnx](../weights/lvface.onnx)

### Datasets
| Dataset | Vai trò (theo proposal) | Trạng thái |
|---|---|---|
| **LFW raw** (`lfw-deepfunneled`) | Gallery unmasked + baseline đối chứng | ✅ 5749 identities |
| **LFW masked** (synthetic, MaskTheFace) | Main controlled test | ✅ 5749 identities, suffix `_surgical` |
| **RMFRD** (`AFDB_*`) | Real-world validation | ✅ gallery (unmasked) + masked probe + masked gallery (sinh thêm) |
| **MAFA** | Stress test mask type/pose | ❌ Chưa tải |
| **team_photos** | Demo webcam | ⚠️ Chưa có |

### Scripts hỗ trợ data
- [scripts/download_data.sh](../scripts/download_data.sh) — tải LFW
- [scripts/make_masked_lfw.py](../scripts/make_masked_lfw.py) — sinh masked LFW bằng MaskTheFace
- [scripts/make_masked_rmfrd_gallery.py](../scripts/make_masked_rmfrd_gallery.py) — sinh synthetic-masked gallery cho RMFRD (phục vụ Lever #1)
- [scripts/audit_rmfrd_label_noise.py](../scripts/audit_rmfrd_label_noise.py) — phát hiện nhãn nhiễu trong RMFRD
- [scripts/measure_alignment_quality.py](../scripts/measure_alignment_quality.py) — đo chất lượng alignment trước/sau khi pad

---

## 3. Stage 2 — Core Pipeline

### Module đã hoàn thành
| File | Mục đích | Trạng thái |
|---|---|---|
| [src/config.py](../src/config.py) | YAML loader + CLI override | ✅ |
| [src/detector.py](../src/detector.py) | RetinaFace + alignment 112×112, có pad/fallback | ✅ |
| [src/embedder.py](../src/embedder.py) | ArcFace + LVFace ONNX wrappers + TTA | ✅ |
| [src/matcher.py](../src/matcher.py) | Cosine similarity, top-K | ✅ |
| [src/dataset.py](../src/dataset.py) | LFW / RMFRD dataset loader + K-shot gallery | ✅ |
| [src/utils.py](../src/utils.py) | aggregate_by_identity, quality weights, Laplacian sharpness | ✅ |

### Decisions đã hiện thực hoá đúng spec
- Mọi embedding output **512-d, L2-normalized** → matcher dùng chung cho mọi model
- Preprocessing chuẩn ArcFace cho ArcFace; **fix riêng cho LVFace dùng RGB** (bug history đã ghi trong `src/embedder.py:104-115`)
- K-shot enrollment (K=1 baseline, K=3/5 cho improvement) — mean + L2-renorm thành 1 prototype/identity
- 500-identity subset cho headline LFW experiment

### Còn thiếu
- ❌ **FaceNet wrapper** chỉ là skeleton (`NotImplementedError`). Hiện 3-model story đang là **ArcFace vs LVFace vs Ensemble**, không có baseline yếu FaceNet. Đây là gap so với proposal — cần quyết định: bổ sung hay đổi narrative.

---

## 4. Stage 3 — Evaluation Framework

[eval/metrics.py](../eval/metrics.py) + [eval/run_experiment.py](../eval/run_experiment.py) đã chạy được:

- **Identification:** Rank-1 / 5 / 10, CMC tới rank 20
- **Verification:** ROC, EER, TAR @ FAR ∈ {0.001, 0.01, 0.1}
- **Latency:** warm-up 10 + đo 1000 ảnh (per CLAUDE.md)
- **Hai flavor Rank-K** cho RMFRD: `*_encoded` (so với literature) và `*_all_probes` (skip = miss, dùng làm headline)
- **Ensemble** (`model.name=ensemble`): detector chạy 1 lần, mỗi member embed → score-level mean fusion
- **Ablation levers** (cho RMFRD): `pad_ratio`, `fallback_align_mode`, `mask_gallery`, `quality_weighted`, `gallery_shots`

---

## 5. Stage 4 — Experiments & Results

### 5.1 LFW (controlled — main story)

| Setup | Model | Rank-1 | Rank-5 | EER | TAR@FAR=0.001 |
|---|---|---:|---:|---:|---:|
| Unmasked baseline | ArcFace | **0.9803** | 0.9803 | 0.0171 | 0.9817 |
| Masked (K=3, TTA) | ArcFace | **0.9708** | 0.9725 | 0.0265 | 0.9686 |
| Masked (K=1) | LVFace | **0.9609** | 0.9629 | 0.0330 | 0.9598 |
| Masked (K=3, ensemble) | ArcFace + LVFace | **0.9716** | 0.9735 | 0.0260 | 0.9711 |

**Insight chính:** Trên LFW masked, ArcFace + K=3 đã rất sát unmasked baseline
(chênh ~1%). Ensemble với LVFace cho gain nhỏ (+0.08% Rank-1, -0.0005 EER) —
không đáng kể trên LFW synthetic.

### 5.2 RMFRD (real-world — gap thật)

| Setup | Rank-1 (all probes) | Rank-5 | Skip | Ghi chú |
|---|---:|---:|---:|---|
| ArcFace baseline (K=1, detect 320, thresh 0.3) | 0.3152 | 0.4915 | 4.1% | — |
| Ensemble v2 (K=5 + mask_gallery + quality_w + pad 0.2 + fallback + TTA) | **0.5759** | 0.6813 | 0% | Best đến nay |
| Ensemble v2 (no mask_gallery) — ablation | 0.5620 | 0.6766 | 0% | +1.4% từ mask_gallery |

**Insight chính:**
- Gap LFW → RMFRD **rất lớn** (~97% → 58%) — chứng minh synthetic mask không đại
  diện được real-world.
- Cộng dồn các lever (mask_gallery + quality_weighted + K=5 + pad + fallback +
  ensemble + TTA) tăng Rank-1 **từ 31% → 58%** (gần gấp đôi).
- Label noise audit ([rmfrd_label_noise_audit.json](../experiments/rmfrd_label_noise_audit.json))
  phát hiện **1393 / 1865 probe (≈74%)** đáng nghi → một phần "lỗi" có thể là do
  nhãn dataset, không phải model. Đây là finding quan trọng cần đưa vào báo cáo.

### 5.3 Ablations & configs đã chạy
- 13 file YAML trong [configs/](../configs/) — bao gồm 5 variant `rmfrd_v2_*`
  (only_pad / only_qw / only_mask_gallery / no_mask_gallery) để cô lập đóng góp
  của từng lever.
- 23 file kết quả JSON trong [experiments/](../experiments/).

---

## 6. Sanity checks & quality

- ✅ Smoke test pipeline: [notebooks/02_pipeline_smoke_test.py](../notebooks/02_pipeline_smoke_test.py)
- ✅ Alignment visualization: [notebooks/03_visualize_alignment.py](../notebooks/03_visualize_alignment.py) + ảnh trong [outputs/alignment_check/](../outputs/alignment_check/)
- ✅ Verify gallery/probe không leak — đã đảm bảo qua [src/dataset.py](../src/dataset.py)
- ✅ Pair verifier: [scripts/verify_pair.py](../scripts/verify_pair.py)
- ✅ RMFRD detect test: [scripts/test_rmfrd_detect.py](../scripts/test_rmfrd_detect.py)

---

## 7. Việc còn lại

### Bắt buộc (để đóng project)
1. **Demo Gradio** ([demo/webcam_app.py](../demo/webcam_app.py)) — hiện chỉ là
   skeleton `NotImplementedError`. Cần wire enroll tab + identify tab + webcam stream.
2. **Bổ sung team_photos** — folder `khai/` đang rỗng; mỗi thành viên cần ≥1 ảnh
   unmasked + 1 ảnh masked.
3. **Báo cáo cuối + slide** — chưa bắt đầu.

### Nên có
4. **Quyết định về FaceNet:** implement hay bỏ và đổi narrative thành
   "ArcFace (general) vs LVFace (specialist) vs Ensemble".
5. **MAFA dataset** — để stress test mask type/color/pose theo proposal §3.
6. **Latency benchmark** — code đã sẵn (`measure_latency: true`), nhưng chưa
   thấy số liệu trong các JSON đã save → cần xem lại đường save.
7. **Mở rộng label-noise analysis** — tận dụng [rmfrd_suspect_probes.csv](../experiments/rmfrd_suspect_probes.csv)
   để báo cáo Rank-1 sau khi lọc noise (estimate "true ceiling" trên RMFRD).

---

## 8. Risk & blocker hiện tại

| Risk | Mức | Mitigation |
|---|---|---|
| FaceNet chưa làm → câu chuyện 3-model thiếu baseline yếu | Trung bình | Hoặc dành 1 buổi implement, hoặc đổi narrative |
| Demo Gradio chưa có → mất điểm trình bày | Cao | Ưu tiên tuần này |
| RMFRD label noise có thể làm sai bias kết luận | Trung bình | Audit đã có, cần report "filtered" alongside "raw" |
| team_photos thiếu → demo không chạy được với cả nhóm | Cao | Chụp ảnh trong tuần này |

---

## 9. Tóm tắt 1 dòng

> Pipeline + evaluation + experiments LFW/RMFRD đã chạy ổn với kết quả định
> lượng đầy đủ (ArcFace LFW masked 97.1%, ensemble RMFRD 57.6% Rank-1).
> Còn lại: demo Gradio, ảnh team đầy đủ, và báo cáo cuối — ước tính 1-2 tuần
> để khoá sổ.
