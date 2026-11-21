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
  - **Độ bất định nội tại của từng chuyên gia:** Predictive Entropy, Cosine OOD distance (Energy score không dùng — xem §5.3 và `docs/research/METHOD_SPECS.md` câu d).
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
- **Nguồn tải thực tế:** file `genimage_test.zip` (split *test* của GenImage, 6.000 AI + 6.000 Real mỗi generator) trên mirror Hugging Face `jzousz/GenImage` (bản mirror cộng đồng; bản chính thức phát hành qua Baidu Yunpan, repo GitHub `GenImage-Dataset/GenImage`). Lấy ngẫu nhiên có seed 2.000 AI + 2.000 Real mỗi generator (ADM, SD v1.4, Midjourney, GLIDE) = 16.000 ảnh, rồi AEGIS **tự chia lại** thành `train_experts` / `val_gating` / `test_benchmark` / S4. Phải ghi rõ điều này trong báo cáo (không dùng split train/val gốc của GenImage). Giấy phép: CC BY-NC-SA 4.0, chỉ dùng phi thương mại. Danh mục file xác nhận độ lệch định dạng: 100% ảnh AI là PNG, 100% ảnh Real là JPEG.
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
- `10 ≤ N ≤ 50`: `w_adapted = (1 − α)·w_gating + α·w_few_shot`, với `α = min(0.6, N/50)`; `w_few_shot` tìm bằng regularized logistic regression / grid search trên support set. Không sửa trọng số gốc của `val_gating`. *(Quyết định 2026-11-18: không cập nhật prototype trong few-shot — chỉ hiệu chỉnh trọng số.)*
- `N > 50`: refit toàn bộ bộ điều phối trên `val_gating_fit` + support set.
- **So sánh công bằng ở S4:** ngoài các baseline không thích nghi, bảng S4 có thêm baseline **Static Learned Weights + support** (refit logistic regression stacking trên `val_gating_fit` + cùng support set). AEGIS phải thắng baseline này để chứng minh lợi ích đến từ *cơ chế* thích nghi, không chỉ từ việc có thêm N mẫu có nhãn.

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

## 6. Quyết định sau rà soát tài liệu (2026-11-18)

Căn cứ: `docs/research/METHOD_SPECS.md` (research-analyst). Người dùng chọn cả ba phương án dựa trên bằng chứng:

1. **Chuẩn hóa kích thước — quyết định theo audit.** Center-crop 224 trực tiếp làm lệch "độ zoom" giữa generator (Midjourney 1024² chỉ lấy ~5% diện tích, ADM/GLIDE 256² lấy ~77%). Chuẩn hóa dữ liệu thật theo **cả hai** chế độ: `crop` (crop 224 trực tiếp) và `resize_crop` (resize cạnh ngắn về 256 có antialias rồi crop 224, theo hướng *Fake or JPEG?*). Chạy shortcut audit trên cả hai; chọn chế độ vượt audit (nếu cả hai cùng vượt, ưu tiên `resize_crop` vì cân bằng tỉ lệ đối tượng). Quy tắc "không resize trước expert tần số" vẫn áp dụng **sau** bước chuẩn hóa này (resize ở bước chuẩn hóa áp dụng đồng đều cho cả Real và AI).
2. **Head expert — báo cáo cả hai biến thể.** `clean`: train chỉ trên ảnh sạch. `aug`: train có augmentation theo UnivFD/CNNSpot (mỗi ảnh nguồn của `train_experts` có thêm biến thể suy giảm: Gaussian blur với xác suất 0.5, σ ~ U[0, 3]; JPEG với xác suất 0.5, Q ~ U[30, 100]). Bảng ablation xuất cho cả hai biến thể head; AEGIS phải được so với baseline mạnh nhất.
3. **Expert 3 — chọn theo validation.** Train cả `log_magnitude_fft` và `srm`; chọn biến thể có AUROC cao hơn trên `val_gating_fit`; báo cáo cả hai.
4. **Báo cáo S4 tách riêng Midjourney và GLIDE**, vì GLIDE có thể không đủ OOD (ADM — thuộc nhóm Seen — chuyển sang GLIDE đạt 97,1% trong GenImage Bảng 3).
5. **Không dùng Energy score** (không có căn cứ cho head nhị phân một logit — METHOD_SPECS câu d).

## 7. Kết quả audit trên dữ liệu thật & quyết định (2026-11-18)

Audit (LR chỉ trên 3 đặc trưng chất lượng, ảnh sạch, 16.000 ảnh GenImage):

| AUROC | `crop` | `resize_crop` |
|---|---|---|
| Tổng | 0.706 | 0.631 |
| Real vs ADM | 0.761 | 0.751 |
| Real vs SD v1.4 | 0.663 | 0.638 |

- `jpeg_harmonize` có hiệu quả: AUROC riêng của `jpeg_quality_est` chỉ 0.54–0.62.
- Đường tắt còn lại là **lịch sử thu nhỏ ảnh**: ảnh Real (~500 px) và SD (512 px) cùng bị thu nhỏ ~2× trong `resize_crop` nên SD gần sạch (từng đặc trưng 0.53–0.56); ADM (256 px gốc) không bị thu nhỏ nên mềm hơn ảnh Real (độ sắc nét 0.72, tần số cao 0.70). Một phần chênh lệch có thể là khác biệt thật giữa ảnh chụp và ảnh AI.

**Quyết định của người dùng:**
1. Dùng **`resize_crop`**.
2. Trên dữ liệu thật, audit **được báo cáo** theo từng generator (cả seen lẫn unseen), không còn là cổng chặn cứng. Thêm baseline **`quality_only`** (cùng LR trên 3 đặc trưng chất lượng, fit trên `val_gating_fit`) vào mọi bảng ablation làm **mức sàn**: mọi phương pháp phải vượt xa mức này. Ghi rõ hạn chế của ADM trong báo cáo.
3. **Cân bằng 50/50 Real/AI trong mọi split** (ảnh Real được phân bổ theo số ảnh AI của từng split; trước đó Real nhiều gấp ~1,8 lần do ảnh Real từ thư mục Midjourney/GLIDE dồn vào các tập seen). 137 ảnh Real nhỏ hơn 224 px bị loại.

## 8. Kết quả thật đầu tiên & chẩn đoán (2026-11-18)

Nguồn: `reports/v1` (ablation), `reports/v2` (bootstrap CI theo cụm ảnh gốc, B=2000; phân tích trọng số). v2 khớp v1 từng byte.

**Đã có ý nghĩa thống kê (CI 95% không chứa 0):**
- S3 (head clean): AEGIS > mọi baseline (+0.023 AUROC vs Static LR, +0.022 vs DINOv2).
- S4 Midjourney, N ≥ 20: AEGIS > Static LR + support (cùng ngân sách nhãn) ở cả hai head (+0.011 → +0.016).
- S1/S2: AEGIS ≈ Static LR (không khác biệt có ý nghĩa). Không được viết "vượt trội" cho S1/S2.
- CLIP đơn lẻ rất mạnh; ở S4 Midjourney head clean N=100, AEGIS thua CLIP nhẹ có ý nghĩa (−0.004).

**Trọng số (val_gating_verify):** khi blur tăng, trọng số chuyển CLIP → DINOv2 (đúng với AUROC từng expert); head aug: w_freq giảm đơn điệu khi JPEG Q giảm (0.35 → 0.14), khớp AUROC freq giảm.

**Chẩn đoán (chỉ để hiểu, không dùng chọn mô hình):**
1. Giả thuyết "đặc trưng chất lượng nhầm JPEG với blur" **bị bác bỏ**: phân biệt JPEG vs blur AUROC 0.94–0.97; nhầm JPEG→blur chỉ 0,5%.
2. Trộn CLIP + DINOv2 dưới JPEG nặng **có lợi** (Q40 head clean: AEGIS 0.869 > CLIP 0.840) — không phải định tuyến sai.
3. **Mức trần oracle** (trọng số cố định tốt nhất cho từng mức suy giảm, dò trên chính lát test — cận trên lạc quan): AEGIS cách oracle chỉ 0.001–0.011 AUROC ở mọi mức S1–S3. ⇒ Bộ điều phối đã gần tối ưu trong họ "trung bình có trọng số"; giới hạn còn lại nằm ở **chất lượng expert**, không ở định tuyến. Cải tiến bộ đo JPEG bị loại bỏ vì dư địa < 0.01.

**Lưu ý toàn vẹn:** `test_benchmark` đã được xem ở v1/v2. Mọi thay đổi hệ thống từ đây phải được chọn **chỉ trên `val_gating_verify`** và phải ghi rõ trong báo cáo là đã có một vòng xem kết quả test trước khi thay đổi.

## 9. Vòng cải tiến thứ nhất (quyết định của người dùng, 2026-11-18)

1. **Thay GLIDE bằng VQDM** làm generator unseen thứ hai (cùng nguồn `genimage_test.zip`, 2.000 AI + 2.000 Real). Lý do: GLIDE tách được 0.979 AUROC chỉ bằng 3 đặc trưng chất lượng ⇒ không phải bài kiểm tra OOD có ý nghĩa. Unseen = Midjourney + VQDM. VQDM cũng phải được audit; nếu cũng tách quá dễ thì báo cáo trung thực.
2. **Thử trộn theo logit** (`gating.fusion: logit`, $P = \sigma(\sum w_i \ell_i)$) như một họ trộn khác ngoài trung bình xác suất. Lựa chọn prob/logit được quyết định **chỉ trên `val_gating_verify`** (Δ AUROC > 0 với CI 95% không chứa 0 và ECE không xấu đi quá `max_ece_increase`).
3. Vì đổi dữ liệu, toàn bộ pipeline chạy lại (ingest → embedding → head → bundle → chọn fusion → ablation v3). Split mới khác split v1/v2. **Báo cáo phải ghi rõ** các quyết định thiết kế ở §6–§9 đã được đưa ra sau khi xem kết quả test v1/v2.

## 10. Kết quả chính v3 (dữ liệu có VQDM, split mới) — 2026-11-21

Nguồn: `reports/v3` (ablation chạy **một lần** trên `test_benchmark` của split v2; bootstrap theo cụm ảnh gốc, B=2000). Lựa chọn trước test (chỉ dùng val): Freq = SRM (val AUROC 0.977/0.968 vs FFT 0.912/0.877); fusion = `prob` (logit không cải thiện AEGIS trên `val_gating_verify`: Δ −0.0001 / −0.0012, CI chứa 0). Audit VQDM: 0.727 (tương đương Midjourney 0.693, không dễ như GLIDE 0.979).

**ΔAUROC (AEGIS − baseline), `*` = CI 95% không chứa 0:**

| Ô | Head | vs Static LR | vs Naive avg | vs CLIP | vs Static LR + support |
|---|---|---|---|---|---|
| S1 | clean | −0.002* | −0.005* | −0.004* | – |
| S1 | aug | +0.000 | +0.001 | +0.009* | – |
| S2 (JPEG) | clean | **−0.009*** | **−0.013*** | +0.023* | – |
| S2 (JPEG) | aug | −0.003 | +0.001 | +0.030* | – |
| S3 (blur/resize) | clean | **+0.013*** | **+0.015*** | +0.139* | – |
| S3 (blur/resize) | aug | −0.001 | −0.003 | +0.045* | – |
| S4 Midjourney N=20 | clean | +0.014* | +0.008* | +0.002* | **+0.014*** |
| S4 Midjourney N=20 | aug | +0.015* | +0.018* | +0.007* | **+0.014*** |
| S4 Midjourney N=100 | clean | +0.008* | +0.002 | −0.004* | **+0.009*** |
| S4 Midjourney N=100 | aug | +0.013* | +0.015* | +0.004 | **+0.010*** |
| S4 VQDM (mọi N) | cả hai | ≈0 (|Δ| ≤ 0.001) | ≈0 | ≈0 | ≈0 |

**Kết luận v3:**
1. **Lặp lại được (v1 → v3):** lợi thế ở S3 với head clean (+0.023 → +0.013) và lợi thế few-shot trên Midjourney ở N=20 so với baseline cùng ngân sách nhãn (+0.013/+0.016 → +0.014/+0.014). Cơ chế neo (N=20) cải thiện có ý nghĩa so với không thích nghi (+0.006 clean, +0.012 aug).
2. **Mới / bất lợi:** ở S2 với head clean, AEGIS **thua có ý nghĩa** Static LR (−0.009) và Naive avg (−0.013); ở v1 là hòa. Với head aug, AEGIS hòa Static LR ở S1–S3.
3. **VQDM bão hòa:** mọi phương pháp ≈ 0.998 AUROC (CLIP một mình 0.998) dù đặc trưng chất lượng không tách được (0.596) ⇒ VQDM dễ với các expert ngữ nghĩa, không phân biệt được các phương pháp. Bằng chứng S4 vẫn dựa vào Midjourney.
4. **Trọng số** lặp lại xu hướng v1/v2: head clean dưới blur chuyển sang DINOv2 (w 0.15 → 0.67); head aug dưới JPEG giảm w_freq (0.39 → 0.28 → 0.20); dưới blur w_freq về sàn (~0.07).
5. Lưu ý toàn vẹn: các quyết định §6–§9 được đưa ra sau khi xem kết quả test v1/v2 (split cũ). v3 là split mới, nhưng phải ghi rõ trong báo cáo.
