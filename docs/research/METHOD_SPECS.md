# METHOD_SPECS — Đặc tả phương pháp cho các engineer agent (AEGIS)

> Căn cứ: 8 paper trong `docs/raw_papers/` (phân tích chi tiết ở `docs/research/<slug>.md`) và các thông số đã chốt v2.0 ở `docs/PROJECT_BRIEF.md` §5. Quy ước: **[Paper]** = nội dung có trong tài liệu (kèm trích dẫn file + mục/bảng); **[Đề xuất]** = ý kiến của Research Analyst, không phải nội dung paper. Khi §5 và paper khác nhau, §5 thắng; điểm khác biệt được liệt kê ở mục 5 để người dùng quyết.
> Lưu ý về độ tin cậy số liệu: các file Markdown chuyển từ PDF; bảng bị vỡ thành cột. Số nào chưa chắc đã ghi rõ trong file phân tích từng paper.
> Phạm vi: nhóm 8 paper chỉ trực tiếp hỗ trợ Dataset, Expert 1/2/3, Calibration, OOD (energy). **Gating MLP, Static Learned Weights, Few-shot (α = min(0,6, N/50)), Champion-Challenger, cờ vàng không có paper nguồn trong 8 bài** — đó là quyết định thiết kế của dự án.

---

## 1. Trả lời năm câu hỏi (a–e)

### a. fake_or_jpeg_2024 và pipeline chuẩn hóa của AEGIS

**Paper mô tả thiên lệch (fake_or_jpeg_2024):**
- Bảng 1: GenImage — ảnh thật JPEG, kích thước đa dạng (ImageNet); ảnh AI PNG, kích thước 128/256/512/1024 (§3 Bảng 1). Cụ thể: Midjourney 1024², SD1.4/SD1.5/Wukong 512², GLIDE/ADM/VQDM 256², BigGAN 128² (§3.2). ImageNet đa số QF 96 (§3.1, Phụ lục A Hình 6 — trích từ nguồn blog ngoài).
- Bằng chứng JPEG: nén ảnh AI làm recall giảm mạnh dù QF 95 (Hình 2, Phụ lục B); ảnh thật chưa nén (FFHQ PNG) được xếp "thật" tăng từ 80,45% (PNG) lên 94,84 (JPEG95), 98,93 (JPEG90), 99,95 (JPEG80), 99,99 (JPEG70), 100 (JPEG60) (§3.1 Bảng 2; detector huấn luyện trên Midjourney).
- Bằng chứng kích thước: độ chính xác trên ảnh thật giảm ở khoảng kích thước trùng kích thước ảnh AI trong huấn luyện (§3.2 Hình 3).

**Paper đề xuất khắc phục:**
1. *JPEG constraint:* ảnh thật chỉ lấy QF 96; ảnh AI nén QF 96 (§4.1). Kết quả (Bảng 3, trung bình ma trận cross-generator): JPEG95 53,91→67,17; JPEG80 50,62→59,37; JPEG60 50,58→55,07.
2. *Size constraint:* chỉ generator 512² (SD1.4/1.5/Wukong), ảnh thật có cạnh trong [450, 550], cân bằng theo lớp ImageNet; center-crop 450 rồi resize 224; suy luận: resize 512 → crop 450 → resize 224 (§4.2). Kết quả (Bảng 4): ResNet50 71,68→82,74; Swin-T 74,09→85,83.
3. Khuyến nghị cuối: ảnh thật và ảnh AI phải có phân phối nén và kích thước **gần bằng nhau**; huấn luyện detector trên cùng nguồn ảnh thật với generator (§5).
4. Bàn về augmentation JPEG ngẫu nhiên: không loại bỏ hoàn toàn thiên lệch vì chỉ dịch phân phối QF; ảnh đã nén bị nén lần hai còn ảnh chưa nén chỉ nén một lần; JPEG không idempotent (§4.1).

**So sánh với AEGIS (center-crop 224, offset bội số 8, PNG, `jpeg_harmonize`):**
- *Giống/ủng hộ:* `jpeg_harmonize` (nén ảnh AI theo phân phối chất lượng của ảnh Real) đúng tinh thần khuyến nghị "phân phối nén gần bằng nhau" (§5) và **tốt hơn** QF cố định 96 theo đúng lập luận của paper (§4.1: augmentation chỉ dịch phân phối). Lưu ý "ảnh AI bị nén một lần, ảnh Real một lần" là đối xứng — phù hợp quan ngại về double compression. Offset bội số 8 để giữ lưới khối JPEG 8×8: **không có trong tài liệu** (paper không bàn tới lưới 8×8; dự án nên tự kiểm chứng).
- *AEGIS thiếu hoặc khác:*
  1. **Kích thước/tỉ lệ đối tượng không được cân bằng.** Paper chỉ ra detector dùng cả đường tắt kích thước và nêu "detectors employing cropping ... are still subject to bias in terms of object size" (§3.2). Crop 224 trực tiếp từ ảnh gốc: Midjourney 1024² chỉ lấy ~4,8% diện tích, ADM/GLIDE 256² lấy ~76,6%, ảnh Real ~450–500² lấy ~20–25% (tính số học của tôi, không phải số của paper). Paper xử lý bằng crop 450 rồi resize 224 (đồng nhất lượng resize) và chọn ảnh thật cùng khoảng kích thước với generator. Đánh đổi: resize phá vỡ lưới 8×8 và dấu vết tần số cao, trong khi crop trực tiếp bảo toàn chúng (paper giới thiệu crop thay resize ở §2.1 là một cách tránh mất thông tin tần số, nhưng không đánh giá).
  2. Bài test `test_no_dataset_shortcut` chỉ dùng 3 đặc trưng chất lượng ảnh; không kiểm tra đường tắt về tỉ lệ đối tượng/độ phân giải nội dung (mà paper chứng minh tồn tại).
  3. Paper đánh giá bằng cách nén ảnh PNG test ở QF 96 để detector không gặp ảnh chưa nén lạ (§4.2). AEGIS tương đương nếu bật `jpeg_harmonize` cho mọi ảnh AI trong cả train/val/test; cần đảm bảo cờ này **bật nhất quán** ở cả S1–S4.
  4. Nhiễu nguồn ảnh thật (ImageNet vs LAION của SD; fingerprint của SD) không giải quyết được (§5 của paper); AEGIS chỉ có thể ghi chú đây là giới hạn.
  5. Paper kiểm tra thiên lệch trên detector huấn luyện từ đầu (ResNet50, Swin-T), **không** trên backbone foundation đóng băng — nên hiệu ứng trên CLIP/DINOv2 chưa có số liệu.
- *Mô tả thiếu ở §5 cần làm rõ:* (i) `jpeg_harmonize` ước lượng phân phối QF của ảnh Real bằng cách nào (đọc bảng lượng tử khi build dữ liệu là hợp lệ, vì điều cấm chỉ áp dụng cho `jpeg_quality_est` của gating); (ii) nén ảnh AI trước hay sau crop (nếu sau, lưới 8×8 bắt đầu tại gốc ảnh crop, vẫn khớp offset bội số 8 của Real); (iii) S2 nén lần hai cả Real lẫn AI — đối xứng, đúng.

### b. UnivFD — đặc trưng CLIP và tiền xử lý
- **[Paper]** Dùng "lớp cuối của bộ mã hóa thị giác CLIP:ViT-L/14", vector "768 chiều" (univfd_2023 §4: "we use the last layer of CLIP:ViT-L/14's visual encoder as ϕ"; "feature representations (of 768 dimensions)"). Lý do: huấn luyện trên 400M cặp ảnh-văn bản; patch 14 nhỏ hơn → có thể mô hình hóa chi tiết thấp (§4). Linear head: một lớp tuyến tính + sigmoid, BCE, ~768 tham số (§4); huấn luyện **có** augmentation Blur+JPEG p=0,5 (Phụ lục B). NN dùng khoảng cách cosine (§4).
- **Tiền xử lý ảnh: không có trong tài liệu** (paper chỉ nêu ảnh test 256×256, §5.1). Cũng không nói rõ 768-d là sau lớp chiếu hay trước chiếu.
- **Khớp với AEGIS?** Số chiều 768 khớp với embedding sau projection của ViT-L/14 mà AEGIS dùng; việc dùng 224×224 là chuẩn của CLIP nhưng **paper không xác nhận** (không có trong tài liệu). Hai khác biệt cần lưu ý: (1) UnivFD huấn luyện head có Blur+JPEG, AEGIS chỉ ảnh sạch (§5.5); (2) dự phòng ViT-B/16 **không được paper kiểm tra** — paper chỉ so CLIP ViT-L/14, CLIP ResNet-50, ImageNet ResNet-50, ImageNet ViT-B/16 (§6.3); CLIP ViT-L/14 tốt hơn CLIP RN50 và ImageNet-pretrained; pretraining data quan trọng hơn kiến trúc.

### c. F3Net / FreqNet và Frequency CNN 4-block ~1,2M tham số trên log-magnitude FFT
**Lựa chọn đầu vào tần số trong hai paper:**
- F3Net (f3net_2020): DCT (không phải FFT). FAD: 3 dải (1/16, 1/16–1/8, 7/8) + bộ lọc học được, IDCT về miền ảnh, xếp kênh (§3.1). LFS: SWDCT cửa sổ 10, bước 2, 6 dải, thống kê `log10 ‖·‖₁` → bản đồ 149×149×6 cho ảnh 299² (§3.2, Eq. 2). Backbone Xception ×2, pretrain ImageNet.
- FreqNet (freqnet_2024): FFT → lọc thông cao (bỏ vuông trung tâm |i|<W/4,|j|<H/4) → iFFT về miền ảnh (Eq. 3–5); thêm conv trên phổ biên độ/pha của feature (FCL, Eq. 7). Mạng residual nhẹ, **1,9M tham số**, huấn luyện từ đầu, Adam, batch 32, 100 epoch (§Implementation Details).

**Ủng hộ thiết kế AEGIS:**
- Mạng nhỏ huấn luyện từ đầu là khả thi: FreqNet 1,9M đạt 92,8 mAcc (17 GAN) so với Ojha/UnivFD 304M đạt 83,0 (Bảng 3) → cỡ ~1,2M của AEGIS cùng bậc độ lớn.
- `log` của độ lớn phổ để cân bằng dải (F3Net Eq. 2 dùng `log10`).
- Nhấn mạnh tần số cao: FAD-High 87,77% > FAD-Mid 87,57% > FAD-Low 86,95% (F3Net Bảng 3); FreqNet cố ý lọc thông cao (§Methodology). AEGIS có `high_freq_energy_ratio`.
- Tần số hỗ trợ độ bền nén trong F3Net: F3-Net LQ 90,43% vs Xception 86,86% (Bảng 1) — nhưng là nén video trên khuôn mặt.

**Mâu thuẫn hoặc cảnh báo:**
- F3Net nói thẳng: FFT/DCT toàn ảnh "không khớp với tính bất biến dịch chuyển và nhất quán cục bộ" nên CNN thông thường "might be infeasible" (§1). Bảng 4 (nhánh LFS, LQ): DCT toàn ảnh 76,16% → SWDCT cục bộ 82,47% → +Stat 84,89% → +D-Stat 86,16%. Ủng hộ thống kê tần số **cục bộ** hơn phổ toàn ảnh — AEGIS dùng phổ log-magnitude toàn ảnh.
- Trên GenImage, bộ phân loại nhận phổ tần số làm đầu vào (Spec) đạt Avg 68,8 (Bảng 3, huấn luyện SD v1.4) và ~50% ở **mọi** suy giảm (Bảng 5: LR112 50,0; LR64 49,9; JPEG65 50,8; JPEG30 50,4; blur σ3 49,9; σ5 49,9). F3Net trên GenImage: Avg 68,7 và 49,9–50,1 trên MJ/ADM/GLIDE (Bảng 3), JPEG65 89,0/JPEG30 74,4 nhưng LR 50,0 và blur σ3 57,9 (Bảng 5). Tức độ nhạy với JPEG **phụ thuộc biểu diễn** (DCT/thống kê tần số bền JPEG hơn phổ thô), còn resize/blur làm sụp cả hai — gần với tiền đề "Expert 3 nhạy suy giảm", nhưng không đồng nhất với "chỉ JPEG".
- FreqNet Bảng 1: các phương pháp dùng phổ FFT thô (Frank 2020 mean Acc 65,7 / AP 63,3; Durall 2020 68,7/65,0; 1-class) kém xa FreqNet (89,6/97,6), tức đầu vào phổ thô tổng quát hóa kém.
- Cả F3Net và FreqNet **không** đánh giá khuếch tán + JPEG/blur/resize đơn ảnh; không có bằng chứng cho "4-block" hay độ phân giải 224.
- **Kết luận [Đề xuất]:** Giữ Expert 3 như thiết kế §5.2 (nó phục vụ vai trò "chuyên gia nhạy, gating hạ trọng số"), nhưng chuẩn bị một ablation nội bộ rẻ: (i) log-magnitude FFT toàn ảnh (mặc định), (ii) ảnh high-pass (kiểu FreqNet HFRI) hoặc phổ theo dải/cục bộ (kiểu LFS). Cần người dùng quyết định (mục 5).

### d. Temperature scaling và energy score
- **[Paper] Temperature scaling (temperature_scaling_2017):** một tham số T tối ưu NLL trên validation, không đổi argmax/accuracy (§4.2 Eq. 9); ECE 15 bin (Bảng 1: ví dụ CIFAR-100 ResNet110 16,53→1,26%); vector/matrix scaling không tốt hơn, quá khớp với validation nhỏ (§5); giả thiết train/val/test cùng phân phối (§4). Nhị phân: Platt scaling `σ(a z + b)` (§4.1); TS không được mô tả riêng cho nhị phân — dùng `σ(z/T)` là **suy ra của tôi**.
- **[Paper] Energy (energy_ood_2020):** `E(x;f) = −T·logsumexp(f_i/T)` trên logit K lớp (Eq. 4); `log max softmax = E(x;f)+f_max` (§3.2) — softmax mất thông tin độ lớn logit; trên CIFAR-10 (WideResNet) FPR95 trung bình 51,04→33,01 (Bảng 1), nhưng AUROC chỉ 90,90→91,88 và AUPR 97,92→97,83; tham số-free với T=1 (§4.2).
- **Có lý do nên thêm energy cho head nhị phân không?** **Không có trong tài liệu** — paper chỉ thử đa lớp K=10/100 trên ảnh 32×32 với WideResNet, không có nhị phân, không có backbone đóng băng, không có ảnh suy giảm. Suy luận của tôi: với head một logit, energy hoặc dư thừa với entropy (đối xứng [−z/2, z/2]) hoặc phụ thuộc cách chọn logit tham chiếu (không đối xứng); chỉ có ý nghĩa mật độ khi head có hai đầu ra softmax tự do. Hiện §5.3 (entropy + cosine) là hợp lý; energy nên là **ablation tùy chọn**, tính từ logit thô (T=1, trước calibration) vì paper nói T≠1 làm giảm phân biệt (§4.2).
- Lưu ý: PROJECT_BRIEF §2.2 và `CLAUDE.md` nêu "Energy / Cosine OOD", còn §5.3 (chốt v2.0) không có energy — theo quy ước §5 thắng.

### e. GenImage — giao thức và số chính (ADM, SD v1.4, Midjourney, GLIDE)
- **Giao thức cross-generator (genimage_2023 §3.2):** huấn luyện trên một generator, test trên 8 generator; ma trận 8×8 (Bảng 2); báo cáo trung bình. Task 2: suy giảm chỉ ở test, huấn luyện SD v1.4 (§3.3).
- **Bảng 2 (ResNet-50, hàng huấn luyện → cột Midjourney / SD1.4 / ADM / GLIDE):** SD1.4 → 54,9 / 99,9 / 53,5 / 61,9; ADM → 58,6 / 53,1 / 99,0 / **97,1**; GLIDE → 50,7 / 50,0 / 56,0 / 99,9; Midjourney → 98,8 / 76,4 / 64,1 / 78,9.
- **Bảng 3 (huấn luyện SD v1.4; MJ / SD1.4 / ADM / GLIDE / Avg):** ResNet-50 54,9 / 99,9 / 53,5 / 61,9 / 72,1; Swin-T 62,1 / 99,9 / 49,8 / 67,6 / 74,8; F3Net 50,1 / 99,9 / 49,9 / 50,0 / 68,7; Spec 52,0 / 99,4 / 49,7 / 49,8 / 68,8.
- **Bảng 4 (trung bình 8 mô hình, theo generator test):** ResNet-50 MJ 59,0, SD1.4 72,3, ADM 59,7, GLIDE 73,1.
- **Bảng 5 (suy giảm; huấn luyện SD v1.4):** ResNet-50 JPEG65 51,9, JPEG30 51,2; blur σ3 97,9, σ5 69,4; LR112 96,2, LR64 57,4. CNNSpot (aug JPEG/blur lúc huấn luyện) JPEG65/30 97,3/97,3.
- **Hệ quả cho AEGIS [Đề xuất]:** Midjourney là generator khó nhất (cột MJ ≤ 64,1 với mọi huấn luyện ngoài chính nó); GLIDE dễ khi có ADM trong huấn luyện (97,1%) ⇒ khi ADM ∈ Seen, GLIDE có thể **không đủ OOD** để thể hiện lợi ích thích nghi; nên báo cáo S4 tách Midjourney và GLIDE. Tham số suy giảm của GenImage (JPEG 65/30, blur σ 3/5, LR 112/64) không trùng §5.5 nên không so số trực tiếp.

---

## 2. Đặc tả theo module

> Cột "Siêu tham số khuyến nghị" là **[Đề xuất]** trừ khi có trích dẫn paper. Mọi giá trị phải đọc từ `configs/default.yaml` (CLAUDE.md §5).

### 2.1 Expert 1 — CLIP (UnivFD-style)
| Mục | Nội dung |
|---|---|
| Phương pháp | Frozen CLIP ViT-L/14 → vector 768-d → head tuyến tính 768→1 (+ sigmoid), BCE. Tùy chọn kiểm chứng: ngân hàng NN-cosine. |
| Paper nguồn | univfd_2023 §4, §6.3; temperature_scaling_2017 (calibration phía sau). |
| Input | Ảnh 224×224 PNG (chuẩn hóa CLIP của backbone; **paper không nêu tiền xử lý**). `eval()`, `torch.no_grad()`. |
| Output | Logit z₁ + embedding cache `.pt` float16 trong `data/cache/`. |
| Siêu tham số | Head: BCE, AdamW lr ~1e-3, weight decay ~1e-4, batch 256 [Đề xuất]. (Paper: lr 5e-5, batch 256 chỉ cho baseline fine-tune toàn ViT:CLIP, Phụ lục B — không áp dụng.) Lr/epoch của linear probe UnivFD: không đọc được trong tài liệu. |
| Rủi ro | (1) Head chỉ ảnh sạch (khác UnivFD dùng Blur+JPEG) → S2/S3 có thể sụt mạnh — đây là điểm cần gating; (2) dự phòng ViT-B/16 không có bằng chứng paper; (3) không nêu rõ sau-projection hay trước-projection. |

### 2.2 Expert 2 — DINOv2 ViT-B/14
| Mục | Nội dung |
|---|---|
| Phương pháp | Frozen DINOv2 ViT-B/14 → CLS 768-d (tùy chọn nối average-pooled patch → 1536-d) → head tuyến tính. |
| Paper nguồn | dinov2_2023 §3–5, Bảng 4, Phụ lục B.3 (linear probe). |
| Input | 224×224 (paper đánh giá linear ở 224, chú thích Bảng 4; huấn luyện cuối ở 518). |
| Output | Logit z₂ + embedding cache. |
| Siêu tham số | Paper tìm lưới lr {1e-4…0,5}, output layers {1,4}, concat avg-pool {có/không} (B.3). Khuyến nghị: tương tự UnivFD head; ablation CLS vs CLS‖mean-patch [Đề xuất]. |
| Rủi ro | Không có bằng chứng trong paper rằng DINOv2 phát hiện ảnh AI; đặc trưng "bền" có thể che mất dấu vết; patch-level cache lớn. |

### 2.3 Expert 3 — Frequency CNN
| Mục | Nội dung |
|---|---|
| Phương pháp | CNN 4 block ~1,2M tham số, huấn luyện từ đầu trên `train_experts` sạch (ngoại lệ duy nhất). Đầu vào: log-magnitude FFT (hoặc SRM, theo §5.2). |
| Paper nguồn | freqnet_2024 (CNN nhẹ 1,9M, huấn luyện từ đầu); f3net_2020 (log, dải tần, thống kê cục bộ). |
| Input | **Không resize toàn cục**: ảnh 224×224 PNG đã crop (CLAUDE.md §2). |
| Output | Logit z₃ + vector đặc trưng penultimate (cho cosine distance). |
| Siêu tham số | Paper FreqNet: Adam, batch 32, 100 epoch, giảm lr 20% mỗi 10 epoch, lr "2e-2" (như in; nghi ngờ, không xác minh). Khuyến nghị: lr 1e-3, weight decay 1e-4, early stop theo hold-out trong `train_experts` [Đề xuất]. |
| Rủi ro | Nhạy JPEG/blur/resize (GenImage Bảng 5: Spec 49,9–50,8; F3Net LR 50,0); sụp trên generator khác (GenImage Bảng 3); `log(0)` → cộng epsilon; nguy cơ học đường tắt định dạng nếu dữ liệu chưa harmonize. |

### 2.4 OOD / Degradation Profiler
| Mục | Nội dung |
|---|---|
| Phương pháp | 3 đặc trưng chất lượng (`jpeg_quality_est` mù, `log_laplacian_var`, `high_freq_energy_ratio`) + 3 entropy + 3 cosine tới tâm `train_experts` + (tùy chọn) energy. |
| Paper nguồn | energy_ood_2020 (energy, §3.1–3.2); univfd_2023 (cosine/NN, §4); f3net_2020 (chia dải tần cho tỉ lệ năng lượng cao). Các đặc trưng chất lượng: **không có paper nguồn**. |
| Input | Ảnh (cho 3 đặc trưng chất lượng), logit/xác suất từng expert, embedding từng expert. |
| Output | Vector 12-d (§5.3) cho gating; điểm OOD trung bình cho cờ vàng (phân vị 95 của `val_gating`). |
| Siêu tham số | Entropy nhị phân (tính trên xác suất đã calibrate hay thô — cần thống nhất); cosine tới tâm tính từ `train_experts`; energy (nếu thêm) từ logit thô, T=1. |
| Rủi ro | `jpeg_quality_est` bị cấm đọc bảng lượng tử; energy dư thừa với entropy khi head một logit (mục 1.d); cosine trong không gian CNN-tần số kém ổn định. |

### 2.5 Calibration (per expert)
| Mục | Nội dung |
|---|---|
| Phương pháp | Temperature scaling một tham số T mỗi expert, tối ưu NLL trên `val_gating_fit`; ECE 15 bin để báo cáo. |
| Paper nguồn | temperature_scaling_2017 §4.1–4.2, §5, Bảng 1. |
| Input / Output | Logit thô → xác suất calibrate `σ(z/T)`. Lưu T (log T để ràng buộc T>0). |
| Siêu tham số | Tối ưu 1-D (paper: ~10 vòng CG hoặc line search); khởi T=1; ECE M=15 (§2, Bảng 1). |
| Rủi ro | Giả thiết cùng phân phối bị vi phạm ở S2/S3/S4 (paper §4); `val_gating_fit` trộn sạch + suy giảm nên T là thỏa hiệp; TS không đổi AUROC/accuracy (nên không giúp tách lớp); ECE đo bin ảnh hưởng bởi kích thước mẫu. |

### 2.6 Gating Controller
| Mục | Nội dung |
|---|---|
| Phương pháp | MLP 12→32→3 + `w = 0,05 + 0,85·softmax(z)`; `P_final = Σ wᵢPᵢ`; entropy regularization (§5.3). |
| Paper nguồn | **Không có trong 8 paper** (quyết định thiết kế của dự án). Gợi ý gián tiếp: temperature_scaling_2017 §5 (mô hình đơn giản hạn chế quá khớp tập validation nhỏ: 515 tham số của MLP, tính số học của tôi, so với 3 trọng số). |
| Input / Output | Input 12-d (mục 2.4). Output w₁..w₃ ∈ [0,05; 0,90], tổng 1. |
| Siêu tham số | Dropout 0,2; hệ số entropy regularization và lr: cần engineer chốt; theo dõi `val_gating_verify`. |
| Rủi ro | Quá khớp `val_gating_fit` (nhỏ); có thể không thắng Static Learned Weights nếu đặc trưng chất lượng không đủ tín hiệu; trọng số ép [0,05;0,9] giới hạn mức "tắt" expert. |

### 2.7 Few-shot adaptation (S4)
| Mục | Nội dung |
|---|---|
| Phương pháp | N<10: α=0 + log cảnh báo; 10≤N≤50: `w_adapted=(1−α)w_gating+α w_few_shot`, α=min(0,6, N/50), `w_few_shot` từ logistic regression có chính quy/grid search trên support set; N>50: refit gating trên `val_gating_fit` + support set. Không cập nhật prototype (quyết định 2026-11-18). |
| Paper nguồn | **Không có trong 8 paper.** Hỗ trợ gián tiếp: temperature_scaling_2017 §5 (mô hình ít tham số tránh quá khớp validation nhỏ → ủng hộ chỉ chỉnh 3 trọng số ở 10–50). univfd_2023 §6.5/Hình 8 (giảm dữ liệu huấn luyện 3–4 lần ít mất tổng quát hóa) — nhưng ở quy mô 8k–720k ảnh, **không** tương đương N=5–100. |
| Input / Output | Support (N/2 AI mới + N/2 Real) với nhãn thật; output trọng số điều chỉnh. |
| Rủi ro | N nhỏ → ước lượng ồn; chống rò rỉ: ảnh support và eval phải rời nhau; baseline "Static Learned Weights + support" là bắt buộc để không lẫn lợi ích dữ liệu thêm. |

### 2.8 Evaluation
| Mục | Nội dung |
|---|---|
| Phương pháp | Accuracy, Precision, Recall, F1, AUROC, ECE (15 bin), Selective Accuracy, Flagged Rate; Champion-Challenger trên `val_gating_verify`; baseline: từng expert, Naive Average, Static Learned Weights, (S4) Static + support. |
| Paper nguồn | temperature_scaling_2017 §2 (ECE/MCE/NLL, reliability diagram); univfd_2023 §5.3 (AP/Acc, ngưỡng chọn trên tập validation; baseline "Oracle" cho thấy cần cố định ngưỡng độc lập với test); genimage_2023 §3.2–3.3 (giao thức cross-generator và suy giảm); fake_or_jpeg_2024 (bằng chứng cần kiểm tra thiên lệch). |
| Siêu tham số | ECE M=15; ngưỡng 0,5 hoặc tune trên `val_gating_fit` (không dùng `test_benchmark`). |
| Rủi ro | Paper dùng Accuracy trung bình sau khi chọn ngưỡng trên validation; không đo Selective Accuracy/Flagged Rate (đây là chỉ số của dự án). |

---

## 3. Bảng so sánh các phương pháp (từ tài liệu)

| Phương pháp | Biểu diễn/backbone | Huấn luyện | Tham số (paper nói) | Chứng cứ chính | Điểm yếu thấy được |
|---|---|---|---|---|---|
| UnivFD (CLIP ViT-L/14 + linear/NN) | 768-d, đóng băng | Chỉ head tuyến tính (~768 tham số), có aug Blur+JPEG | 304M backbone (freqnet Bảng 3) | Total mAP LC 93,38; Avg acc NN k=9 84,25 (univfd Bảng 2/3) | Không báo ECE; tiền xử lý không nêu |
| DINOv2 ViT-B/14 | CLS 768-d (± patch) | Linear probe (đóng băng) | embed 768 (Bảng 17); số block đọc không chắc | ImageNet linear 84,5; Im-R 63,3 (Bảng 4, 6) | Không có thí nghiệm ảnh AI |
| F3Net | DCT, FAD+LFS, Xception×2 | Pretrain ImageNet + end-to-end | 48,9M (freqnet Bảng 3) | FF++ LQ 90,43% (Bảng 1); GenImage Avg 68,7 | Khuôn mặt/video; sụp cross-generator |
| FreqNet | Ảnh high-pass iFFT + conv trên phổ feature | Từ đầu | 1,9M | 17 GAN mAcc 92,8 (Bảng 3) | Chỉ GAN; không kiểm tra JPEG/blur |
| Spec (trong GenImage) | Phổ tần số → CNN | Từ đầu | — | GenImage Avg 68,8 | ~50% ở mọi suy giảm (Bảng 5) |
| ResNet-50 / Swin-T (GenImage) | Pixel | Từ đầu | — | Swin-T Avg 74,8 (SD1.4→) | JPEG65 51,9 (ResNet) |
| Temperature scaling | Logit | 1 tham số, NLL trên val | 1 | ECE 16,53→1,26 (CIFAR-100 R110) | Giả thiết cùng phân phối |
| Energy score | Logit đa lớp | Không huấn luyện | 0 | FPR95 51,04→33,01 (CIFAR-10) | Chưa kiểm cho nhị phân |

---

## 4. Rủi ro tổng hợp xuyên module
1. **Đường tắt dữ liệu** (fake_or_jpeg_2024): kích thước/tỉ lệ đối tượng chưa được cân bằng bằng crop trực tiếp; test shortcut chỉ kiểm 3 đặc trưng chất lượng.
2. **Dịch phân phối** vi phạm giả thiết của temperature scaling (S2/S3/S4).
3. **S4 có thể yếu:** ADM→GLIDE dễ (GenImage Bảng 2: 97,1%).
4. **Head chỉ ảnh sạch** khác với thông lệ trong UnivFD/CNNSpot (có Blur+JPEG), dẫn tới baseline đơn expert dễ sụp ở S2/S3 — vừa hỗ trợ câu chuyện gating, vừa có thể bị phản biện là "baseline bị làm yếu".
5. **Không có paper nguồn** cho gating, few-shot, Champion-Challenger, cờ vàng.

---

## 5. Điểm mâu thuẫn/khác biệt với §5 cần người dùng quyết
1. **Cân bằng kích thước/tỉ lệ đối tượng:** §5.1 center-crop 224 trực tiếp (không resize) vs paper crop-rồi-resize cùng khoảng kích thước (fake_or_jpeg §4.2). Giữ nguyên hay thêm biến thể?
2. **Head chỉ ảnh sạch (§5.5)** vs UnivFD/CNNSpot huấn luyện có Blur+JPEG (univfd Phụ lục B; genimage §3.3: CNNSpot JPEG65/30 = 97,3).
3. **Expert 3 trên log-magnitude FFT toàn ảnh** vs F3Net (thống kê cục bộ tốt hơn toàn ảnh; Bảng 4) và FreqNet (ảnh high-pass); phổ thô kém tổng quát (Spec ~50%; Frank/Durall thấp). Có chấp nhận ablation phụ?
4. **Energy:** `CLAUDE.md`/Brief §2.2 nêu energy; §5.3 không có; paper không ủng hộ cho head nhị phân một logit.
5. **ViT-B/16 làm dự phòng CLIP:** paper không kiểm tra CLIP ViT-B/16.
6. **S4/GLIDE:** có thể ít OOD hơn dự kiến vì ADM ∈ Seen.
7. **Tham số suy giảm** §5.5 khác GenImage Task 2 (JPEG 65/30, blur σ 3/5, LR 112/64) — chấp nhận, không so số trực tiếp.

---

## 6. Câu hỏi mở cho người dùng
1. `jpeg_harmonize`: ước lượng phân phối QF của ảnh Real bằng cách nào (đọc bảng lượng tử lúc build dữ liệu có được phép không)? Nén ảnh AI trước hay sau crop? Cờ này có bật nhất quán ở mọi tập?
2. Có thêm kiểm tra đường tắt thứ hai (tỉ lệ đối tượng/độ phân giải nội dung) vào `test_no_dataset_shortcut` không?
3. Embedding CLIP: lấy output sau projection (768-d) và áp dụng chuẩn hóa L2 hay không? Cần xác nhận tiền xử lý CLIP (paper UnivFD không nêu).
4. DINOv2: dùng CLS 768-d hay CLS‖mean-patch 1536-d?
5. Expert 3: log-magnitude FFT toàn ảnh (§5.2) hay thêm biến thể high-pass/phổ cục bộ? Có chấp nhận lệch khỏi "4-block ~1,2M" nếu ablation chỉ ra khác biệt lớn?
6. Entropy nhị phân tính trên xác suất đã calibrate hay thô? Có thêm energy (head 2 đầu ra) làm ablation?
7. Temperature: một T chung cho toàn bộ `val_gating_fit` hay T theo nhóm suy giảm? Có refit T trên support set ở S4?
8. S4: báo cáo riêng Midjourney và GLIDE? Có hạ ADM khỏi Seen để GLIDE thật sự OOD?
9. Ngưỡng quyết định: 0,5 cố định hay tune trên `val_gating_fit` (không bao giờ `test_benchmark`)?
10. Baseline "chỉ ảnh sạch" của expert đơn có cần bổ sung một bản có augmentation JPEG/blur để chứng minh AEGIS không chỉ thắng một baseline yếu?
