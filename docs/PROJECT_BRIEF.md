# AEGIS — Project Brief (Bối cảnh cốt lõi của đồ án)

> Tài liệu này là bối cảnh nghiên cứu bắt buộc cho mọi agent và người đóng góp. Mọi quyết định thiết kế và mọi dòng code phải phục vụ mục tiêu thực nghiệm ở Mục 3.

## 1. Bản chất & bối cảnh đồ án

- **AEGIS** (Adaptive Multi-Expert AI Image Forensics) là đồ án nghiên cứu khoa học cấp sinh viên, chạy trên máy tính cá nhân (**local-first**), giải quyết bài toán phát hiện ảnh do AI tạo ra (**Real vs. AI-Generated**).
- **Chiến lược:** "Lõi Chuyên Sâu + Vành Đai Đạt Chuẩn" (Adaptive Multi-Expert Routing). Không tự huấn luyện mô hình khổng lồ, không chạy vòng lặp Continual Learning nặng (dễ gây Catastrophic Forgetting và quá tải phần cứng).
- **Vấn đề thực tế cần giải quyết:**
  - Detector đơn lẻ từ các bài báo thường sụp đổ khi ảnh bị suy giảm chất lượng (nén JPEG qua mạng xã hội, làm mờ, resize).
  - Detector đơn lẻ thất bại khi gặp công nghệ tạo sinh mới chưa từng thấy (**OOD**).
  - Dữ liệu mới thu thập được thường rất ít mẫu (**few-shot**).

## 2. Cơ chế hoạt động cốt lõi

### 2.1. Ba chuyên gia đóng băng trọng số (Frozen Backbones)
Ba góc nhìn khác nhau để triệt tiêu điểm mù của từng mô hình đơn lẻ:

| Expert | Mô hình | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| 1 — Ngữ nghĩa | CLIP (kiểu UnivFD) | Logic tổng thể của ảnh | Yếu khi mất ngữ cảnh cục bộ |
| 2 — Cấu trúc vi mô | DINOv2 | Bề mặt vật liệu, ánh sáng, chi tiết cục bộ | — |
| 3 — Miền tần số | FFT / High-Pass CNN | Bắt dấu vết lưới tần số cao | Rất nhạy, dễ hỏng khi ảnh bị nén JPEG hoặc resize |

### 2.2. Bộ điều phối thông minh (Adaptive Gating Controller)
- Trước khi kết luận, hệ thống phân tích:
  - **Hồ sơ chất lượng ảnh:** mức nén JPEG, độ mờ, độ phân giải.
  - **Độ bất định nội tại của từng chuyên gia:** Predictive Entropy, Energy / Cosine OOD score.
- Từ đó chia trọng số động `(w1, w2, w3)`.
- **Ví dụ:** ảnh bị nén nặng → giảm trọng số chuyên gia Tần số, dồn trọng số cho CLIP / DINOv2.

### 2.3. Thích nghi theo lượng dữ liệu (Data-Aware Few-Shot Adaptation)

| Số mẫu mới N | Hành vi |
|---|---|
| `N < 10` | **Conservative Routing:** giữ nguyên trọng số chuẩn, không cập nhật để tránh nhiễu, chỉ ghi log cảnh báo khan hiếm dữ liệu. |
| `10 ≤ N ≤ 50` | Hiệu chỉnh nhẹ trọng số/prototype trong không gian đặc trưng theo công thức neo: `w_final = (1 - α) * w_base + α * w_new`. **Không fine-tune backbone.** |
| `N > 50` | Tối ưu lại toàn bộ bộ điều phối. |

### 2.4. An toàn & trải nghiệm người dùng
- Luôn trả về **kết quả dự đoán rõ ràng kèm độ tin cậy**.
- Gắn **cờ cảnh báo vàng** khi cả 3 chuyên gia mâu thuẫn gay gắt hoặc ảnh quá biến dạng.
- Mọi cập nhật trọng số phải qua kiểm thử **Champion-Challenger** và **tự động Rollback** nếu làm giảm hiệu năng trên tập chuẩn.

## 3. Mục tiêu thực nghiệm bắt buộc (Ablation Study)

Toàn bộ code phải phục vụ được **bảng thực nghiệm đối chứng cuối cùng** trên 4 kịch bản:

| # | Kịch bản |
|---|---|
| S1 | Ảnh gốc sạch |
| S2 | Ảnh nén JPEG |
| S3 | Ảnh mờ / resize |
| S4 | Ảnh từ generator mới (OOD / few-shot) |

**Các phương pháp so sánh:** CLIP riêng · DINOv2 riêng · FFT riêng · Naive Average Ensemble · Static Learned Weights · **AEGIS Adaptive Gating**.

**Phải chứng minh được:**
- AEGIS Adaptive Gating **vượt trội và ổn định hơn** từng chuyên gia đơn lẻ và hơn phép lấy trung bình cộng ngây thơ (Naive Average Ensemble).
- **Tuyệt đối không rò rỉ dữ liệu** giữa 3 tập:
  - `train_experts` — huấn luyện head của chuyên gia.
  - `val_gating` — calibration xác suất + huấn luyện bộ điều phối.
  - `test_benchmark` — chỉ dùng cho đánh giá cuối cùng.

## 4. Quyết định thiết kế đã chốt

1. **Champion-Challenger không dùng `test_benchmark`.** `val_gating` được chia nội bộ thành `val_gating_fit` (calibration + huấn luyện bộ điều phối) và `val_gating_verify` (quyết định thăng hạng/rollback). `test_benchmark` chỉ được dùng **một lần** cho bảng ablation cuối. *Lý do:* nếu mỗi lần cập nhật đều dựa vào `test_benchmark` để quyết định, tập test gián tiếp tham gia chọn mô hình → rò rỉ.
2. **Bộ điều phối phải được học trên ảnh suy giảm.** `val_gating` chứa cả các bản nén JPEG / làm mờ / resize. Split theo nhóm: mọi phiên bản suy giảm của cùng một ảnh gốc nằm chung một tập. *Lý do:* nếu chỉ thấy ảnh sạch, bộ điều phối không thể học được quy tắc "nén nặng → giảm trọng số FFT"; nếu các bản của cùng một ảnh nằm ở nhiều tập → rò rỉ qua ảnh gần trùng.
3. **Thêm baseline Static Learned Weights** (logistic regression stacking trên xác suất đã calibrate, fit trên `val_gating_fit`). *Lý do:* để chứng minh lợi ích nằm ở việc trọng số **thay đổi theo từng ảnh**, AEGIS phải thắng cả bộ trọng số cố định đã học, không chỉ thắng trung bình cộng.
4. **Giao thức kịch bản S4 (OOD / few-shot):** giữ lại hoàn toàn ít nhất một generator khỏi `train_experts` và `val_gating`. Mẫu few-shot và ảnh đánh giá của generator đó không trùng nhau. Chạy với nhiều mức N (ví dụ 5, 20, 100) để thể hiện cả 3 chế độ thích nghi.

## 5. Thông số kỹ thuật đã chốt (v2.0)

> Mục này thay thế mọi mô tả chung chung ở các mục trên nếu có khác biệt.

### 5.1. Dataset & chống đường tắt (BẮT BUỘC)
- **Nguồn:** tập con GenImage, ~12.000–16.000 ảnh, 50% Real (ImageNet) / 50% AI.
- **Generator đã thấy (Seen):** ADM, Stable Diffusion v1.4 → dùng cho `train_experts`, `val_gating`, S1/S2/S3.
- **Generator chưa thấy (Unseen, chỉ cho S4):** Midjourney, GLIDE — khóa kín hoàn toàn khỏi huấn luyện head và bộ điều phối.
- **Loại bỏ:** BigGAN (128×128, không chuẩn hóa được về 224). CIFAKE (32×32) chỉ làm dữ liệu giả để test pipeline, không bao giờ vào benchmark.
- **Lý do:** trong GenImage, ảnh Real là JPEG nhiều kích thước, ảnh AI là PNG kích thước cố định → định dạng/kích thước tự "tiết lộ" nhãn.
- **Chuẩn hóa trước khi chia tập:** mọi ảnh Center-Crop **224×224**, lưu **PNG không nén**. Bước hòa đồng lịch sử JPEG có thể bật/tắt trong config (`data.normalize.jpeg_harmonize`), vì ảnh Real vẫn mang dấu vết nén JPEG trong điểm ảnh dù đã lưu lại thành PNG.
- **Bài test bắt buộc `test_no_dataset_shortcut`:** Logistic Regression chỉ dùng 3 đặc trưng chất lượng ảnh trên S1 sạch phải có Accuracy và AUROC trong khoảng **0.50 ± 0.05**.

### 5.2. Backbone & chuyên gia (phần cứng: RTX 4060 Laptop 8 GB)
| Expert | Mô hình | Huấn luyện |
|---|---|---|
| 1 — Ngữ nghĩa | CLIP `ViT-L/14` (dự phòng `ViT-B/16`) | Đóng băng, chỉ train head trên ảnh sạch |
| 2 — Cấu trúc | DINOv2 `ViT-B/14` | Đóng băng, chỉ train head trên ảnh sạch |
| 3 — Tần số | 4-block Frequency CNN (~1,2M tham số) trên phổ log-magnitude FFT / SRM | **Ngoại lệ duy nhất:** train từ đầu trên `train_experts` sạch |

Embedding cache dạng `.pt` float16 trong `data/cache/`.

### 5.3. Bộ điều phối (Gating)
- **Input 12 chiều:** 3 đặc trưng chất lượng (`jpeg_quality_est` — ước lượng mù blockiness, cấm đọc quantization table; `log_laplacian_var`; `high_freq_energy_ratio`) + 3 entropy + 3 cosine distance tới tâm `train_experts` + 3 xác suất đã calibrate.
- **Mô hình:** `Linear(12,32) → ReLU → Dropout(0.2) → Linear(32,3)` + entropy regularization.
- **Trọng số:** `w = 0.05 + 0.85 · softmax(z)` ⇒ tổng bằng 1, mỗi `w_i ∈ [0.05, 0.90]`. Kết quả `P_final = Σ w_i · P_i`.

### 5.4. Few-shot adaptation (S4)
- Support set = **N/2 ảnh AI mới + N/2 ảnh Real** từ kho Real riêng. Chạy N ∈ {5, 20, 100}.
- `N < 10`: `α = 0`, giữ nguyên trọng số, ghi log cảnh báo khan hiếm.
- `10 ≤ N ≤ 50`: `w_adapted = (1 − α)·w_gating + α·w_few_shot`, với `α = min(0.6, N/50)`; `w_few_shot` tìm bằng regularized logistic regression / grid search trên support set; cập nhật prototype từng expert cho khoảng cách OOD. Không sửa trọng số gốc của `val_gating`.
- `N > 50`: refit toàn bộ bộ điều phối trên `val_gating_fit` + support set.

### 5.5. Thông số suy giảm
| Tập | JPEG | Resize | Blur |
|---|---|---|---|
| `val_gating` | Ngẫu nhiên liên tục `Q ∈ [30, 95]` | — | Ngẫu nhiên `σ ∈ [0.5, 3.0]` |
| `test_benchmark` S2 | `Q ∈ {85, 65, 40}` | — | — |
| `test_benchmark` S3 | — | 0.5× bilinear thu nhỏ rồi phóng lại | `k=11, σ=1.5` và `k=17, σ=2.5` |

Head của các expert chỉ được train trên ảnh sạch.

### 5.6. Cờ vàng & chỉ số đánh giá
- `warning_flag = True` nếu thỏa ít nhất một điều kiện:
  - **Mâu thuẫn gay gắt:** có `P_i > 0.80` và `P_j < 0.20`, **và** `Std(P1, P2, P3) > 0.30`.
  - **Lưỡng lự:** `P_final ∈ [0.40, 0.60]` **hoặc** điểm OOD trung bình > phân vị 95 của `val_gating`.
- Chỉ số bắt buộc: Accuracy, Precision, Recall, F1, AUROC, ECE, **Selective Accuracy** (độ chính xác trên mẫu không bị gắn cờ) và **Flagged Rate** (tỷ lệ mẫu bị gắn cờ), cho từng kịch bản S1–S4.
