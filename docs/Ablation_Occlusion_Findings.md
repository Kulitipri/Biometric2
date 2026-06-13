# Ablation: Occlusion-aware preprocessing & alignment trên RMFRD

**Group 12 · HUST · Xác thực sinh trắc học**
**Ngày:** 2026-06-13

## Câu hỏi nghiên cứu

Liệu ba kỹ thuật tiền xử lý "occlusion-aware" có cải thiện nhận diện mặt đeo
khẩu trang **dưới ràng buộc pretrained-only** (không train/fine-tune) trên dữ
liệu thật RMFRD hay không:

1. **Align 3 điểm / auto** — bỏ 2 khóe miệng (bị mask che) khỏi affine alignment,
   chỉ dùng 2 mắt + mũi; chế độ `auto` tự chuyển 3pt khi residual 5pt > 3.0px.
2. **Occlusion mask (`lower_half`)** — xám hóa vùng mask (nửa dưới khuôn mặt đã
   align) về 127 trước khi embed.
3. **Calibrated fusion** — chuẩn hóa cosine mỗi model về thang chung rồi gộp có
   trọng số ưu tiên LVFace, thay cho mean cosine thô.

## Thiết lập

Ensemble ArcFace R100 + LVFace, K=5 multi-shot enrollment, RMFRD clean subset
(397 identity, 1919 probe, 0 skip). Mỗi lever bật độc lập trên cùng baseline để
cô lập đóng góp.

## Kết quả

| Cấu hình | Rank-1 | Rank-5 | Rank-10 | EER | TAR@FAR=0.001 |
|---|---:|---:|---:|---:|---:|
| **Baseline** (5pt · mean · no occlusion) | **0.6123** | 0.7150 | 0.7436 | 0.1755 | 0.5508 |
| A — chỉ align auto/3pt | 0.5946 | 0.6972 | 0.7358 | 0.1772 | 0.5373 |
| B — chỉ occlusion `lower_half` | 0.5263 | 0.6722 | 0.7134 | 0.1767 | 0.3538 |
| C — chỉ calibrated fusion | 0.6123 | **0.7181** | **0.7441** | **0.1750** | **0.5524** |
| A+B+C (gộp cả ba) | 0.5044 | 0.6524 | 0.7061 | 0.1870 | 0.3299 |

## Phân tích

**Không lever nào vượt baseline về Rank-1.** Tác động cô lập:

- **Occlusion (B) hại nặng nhất** (Rank-1 −8.6, TAR@FAR=0.001 −19.7 điểm). Nguyên
  nhân: ArcFace và đặc biệt LVFace được huấn luyện trên khuôn mặt đầy đủ và LVFace
  vốn là specialist đã học cách xử lý khẩu trang thật. Việc xám hóa nửa dưới tạo
  ảnh **out-of-distribution** (vùng phẳng + cạnh tương phản nhân tạo) làm embedding
  xấu đi, thay vì để model tự xử lý mask thật. Occlusion-aware chỉ hữu ích cho
  model **chưa** được tối ưu cho occlusion.

- **Align 3pt/auto (A) hại nhẹ** (Rank-1 −1.8). Affine 5 điểm — kể cả khi 2 khóe
  miệng bị mask che và RetinaFace phải ước lượng — vẫn nhiều ràng buộc hơn và ổn
  định hơn 3 điểm trên ảnh có biến thiên pose. Sai số ước lượng khóe miệng nhỏ
  hơn lợi ích mất đi khi bỏ chúng.

- **Calibrated fusion (C) trung tính**, nhỉnh nhẹ ở Rank-5 (+0.3), EER (−0.0005)
  và TAR (+0.16) nhưng trong khoảng nhiễu; Rank-1 không đổi.

- Tổn thất cộng dồn A+B (−10.4) khớp với cấu hình gộp (−10.8), xác nhận các lever
  tác động gần như cộng tính.

## Kết luận

Trần Rank-1 ~61% trên RMFRD **không thể phá bằng tiền xử lý** dưới ràng buộc
pretrained-only. Giới hạn nằm ở **dữ liệu** (mặt phân giải thấp — median cạnh
ngắn 107px; nhiễu nhãn còn sót) chứ không ở pipeline alignment/embedding. Đây là
một **negative result hợp lệ**: với SOTA pretrained masked-FR, occlusion-aware
preprocessing không cộng thêm giá trị vì độ robust đã nằm sẵn trong trọng số model.

Cấu hình khuyến nghị cho headline RMFRD: **baseline `rmfrd_clean`** (5pt · mean ·
no occlusion). Có thể bật `fusion=calibrated` cho phần verification (gain biên).

## Tái lập

```bash
# Baseline
python eval/run_experiment.py --config configs/rmfrd_clean.yaml
# A / B / C (cô lập từng lever)
python eval/run_experiment.py --config configs/rmfrd_occlusion.yaml --override model.occlusion_mask=none model.fusion=mean
python eval/run_experiment.py --config configs/rmfrd_occlusion.yaml --override detector.align_landmarks=5pt model.fusion=mean
python eval/run_experiment.py --config configs/rmfrd_occlusion.yaml --override detector.align_landmarks=5pt model.occlusion_mask=none
```
