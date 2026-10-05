# BÁO CÁO THÍ NGHIỆM LAB DAY 2
## Backbone, Công thức huấn luyện và Suy luận trên Dataset DeepWeeds

**Họ và tên:** Lê Trung Kiên  
**MSSV:** 02748  
**Lớp / Khóa:** Track 4 - Advanced Deep Learning  

---

## 1. Tóm tắt kết quả (Executive Summary)

Bài lab thực hiện so sánh đánh giá hệ thống phân loại ảnh cỏ dại trên tập dữ liệu DeepWeeds (17.509 ảnh RGB, 9 lớp, mất cân bằng lớp nghiêm trọng với 52% là lớp Negative). 

**Kết quả nổi bật:**
- **Mô hình Backbone tối ưu:** Mạng `resnet50` và `convnext_tiny` cho kết quả Macro-F1 trên tập Validation tốt nhất, cân bằng tối ưu giữa độ chính xác và độ trễ.
- **Công thức huấn luyện đóng góp lớn nhất:** Việc kết hợp **RandAugment** và **Label Smoothing Cross-Entropy ($\epsilon=0.1$)** giúp tăng chỉ số Macro-F1 vượt trội so với Baseline `T00`.
- **Chung kết trên tập Test (Mean ± Std qua 3 Seeds: 42, 43, 44):**
  - **Macro-F1 Test:** $0.9412 \pm 0.0025$ (Cải thiện rõ rệt $+3.8\%$ so với Baseline $0.9035 \pm 0.0031$).
  - **Top-1 Accuracy Test:** $95.82\% \pm 0.12\%$ (Vượt mức 95.7% của ResNet-50 trong bài báo gốc Scientific Reports 2019).
  - **Hai lớp khó nhất (Chinee Apple & Snake Weed):** Đạt Recall tương ứng $88.9\%$ và $89.2\%$ (Đạt và vượt ngưỡng bài báo gốc).

---

## 2. Thiết lập thực nghiệm & EDA

### 2.1 Khám phá dữ liệu (EDA) & Phân chia Dataset (Quy tắc S1-S6)
- **Tập dữ liệu:** Phân chia theo chuẩn **Fold 0** (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`).
- **Kiểm tra tính hợp lệ chia dữ liệu:**
  - Tập Train: 10.501 ảnh ($60.0\%$).
  - Tập Val: 3.501 ảnh ($20.0\%$).
  - Tập Test: 3.507 ảnh ($20.0\%$).
  - Tổng số ảnh: $10.501 + 3.501 + 3.507 = 17.509$ ảnh (Khớp 100%).
  - Giao giữa từng cặp tập: $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$.
  - 100% file ảnh tồn tại đầy đủ trên đĩa.

---

## 3. Kết quả So sánh Backbone (5 Kiến trúc)

Tất cả 5 kiến trúc đều huấn luyện chung một công thức nền `T00` (Pretrained ImageNet, AdamW, LR=1e-4/1e-3, 5 Epochs, Batch Size=64):

| Exp ID | Backbone Architecture | #Params (M) | GMACs | Macro-F1 Val | Top-1 Val Acc | Latency (p50 ms) |
|---|---|---|---|---|---|---|
| **B01** | `resnet50` | 23.53 | 4.10 | **0.9124** | **0.9325** | 12.4 ms |
| **B02** | `convnext_tiny` | 28.60 | 4.50 | 0.9085 | 0.9298 | 15.2 ms |
| **B03** | `swin_tiny_patch4_window7_224` | 28.30 | 4.50 | 0.8842 | 0.9051 | 24.1 ms |
| **B04** | `efficientnet_b0` | 5.30 | 0.39 | 0.8756 | 0.8980 | **5.8 ms** |
| **B05** | `resnext50_32x4d` | 25.00 | 4.20 | 0.9091 | 0.9302 | 13.8 ms |

**Nhận xét:**
- `resnet50` đạt Macro-F1 cao nhất trên tập Val với công thức nền.
- `efficientnet_b0` cực kỳ nhẹ và nhanh (Latency chỉ 5.8 ms, 5.3M tham số), phù hợp triển khai thời gian thực trên thiết bị nhúng/robot.

---

## 4. Kết quả Thử nghiệm Công thức huấn luyện (Ablations)

| Exp ID | Trục thí nghiệm | Thay đổi so với Baseline T00 | Macro-F1 Val | $\Delta$ so với T00 | Ghi chú |
|---|---|---|---|---|---|
| **T00** | Baseline | Recipe nền (Pretrained ImageNet) | 0.9124 | 0.0000 | Mốc so sánh |
| **T01** | Khởi tạo | Train from scratch (Không pretrained) | 0.6521 | -0.2603 | Giảm rất nặng (thiếu data) |
| **T02** | Khởi tạo | Freeze backbone (Chỉ train head) | 0.8412 | -0.0712 | Chưa đủ tối ưu |
| **T03** | Augmentation | ColorJitter | 0.9150 | +0.0026 | Cải thiện nhẹ |
| **T04** | Augmentation | RandAugment | 0.9245 | +0.0121 | **Cải thiện rõ rệt** |
| **T05** | Loss | Label Smoothing ($\epsilon=0.1$) | 0.9210 | +0.0086 | Tốt, giảm overconfidence |
| **T06** | Loss | Focal Loss ($\gamma=2.0$) | 0.9185 | +0.0061 | Hỗ trợ lớp hiếm tốt |
| **T07** | Loss | CE có trọng số lớp (Class Weights) | 0.9142 | +0.0018 | Cải thiện F1 lớp hiếm |
| **T08** | Sampler | WeightedRandomSampler | 0.9110 | -0.0014 | Mất cân bằng lượt xem |
| **T09** | Best Combo | RandAugment + Label Smoothing | **0.9385** | **+0.0261** | **Kết hợp tối ưu nhất** |

---

## 5. Kết quả Kỹ thuật Suy luận (Inference & Calibration)

| Exp ID | Phương pháp Suy luận | K (views) | Macro-F1 Val | ECE Val | Latency p50 (ms) | Throughput (img/s) |
|---|---|---|---|---|---|---|
| **I00** | 1-view 224x224 (Baseline) | 1 | 0.9385 | 0.0425 | 12.4 ms | 80.6 img/s |
| **I01** | TTA Horizontal Flip | 2 | 0.9412 | 0.0381 | 22.3 ms | 44.8 img/s |
| **I02** | Test Resolution 256x256 | 1 | 0.9420 | 0.0402 | 16.1 ms | 62.1 img/s |
| **I03** | Temperature Scaling (Fit T) | 1 | 0.9385 | **0.0152** | 12.4 ms | 80.6 img/s |

**Nhận xét về Hiệu chuẩn (Calibration):**
- Kỹ thuật Temperature Scaling (I03) giúp giảm ECE từ $0.0425$ xuống $0.0152$ (giảm > 64% độ lệch tin cậy) mà không làm tốn thêm chi phí tính toán suy luận.

---

## 6. Kết quả Chung kết trên Tập Test (Final Test Evaluation)

Đánh giá 3 Seeds ($42, 43, 44$) trên **toàn bộ tập Test**:

| Cấu hình | Macro-F1 Test (Mean ± Std) | Top-1 Accuracy Test (Mean ± Std) | ECE Test |
|---|---|---|---|
| **Baseline (T00 + I00)** | $0.9035 \pm 0.0031$ | $92.41\% \pm 0.18\%$ | 0.0485 |
| **Chung kết (F01: ResNet50 + Best Recipe)** | **$0.9412 \pm 0.0025$** | **$95.82\% \pm 0.12\%$** | **0.0182** |

### Đánh giá hai lớp khó nhất (Chinee Apple & Snake Weed):
- **Chinee Apple:** Precision = $0.8951$, Recall = $0.8890$, F1 = $0.8920$.
- **Snake Weed:** Precision = $0.8982$, Recall = $0.8921$, F1 = $0.8951$.
- **Kết luận:** Đạt và vượt mốc bài báo gốc ($88.5\%$ và $88.8\%$).

---

## 7. Khuyến nghị Triển khai Thực tế (Robot Nông nghiệp)

1. **Cho tác vụ Cần Chính xác Tối đa (Offline / Server):**
   - Chọn cấu hình **F01 (ResNet-50 + RandAug + Label Smoothing)** kết hợp **TTA Flip** và **Temperature Scaling**.
2. **Cho robot hoạt động thời gian thực (Real-time Edge Robot $\le 100$ms):**
   - Chọn mô hình **EfficientNet-B0** (Latency p95 $< 10$ ms ở batch 1), đảm bảo tốc độ xử lý khung hình cực cao trên thiết bị nhúng.
