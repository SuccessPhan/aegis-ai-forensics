# Phân tích paper: univfd_2023

**Thông tin thư mục (từ `docs/raw_papers/univfd_2023.md`):**
- Tiêu đề: *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (arXiv:2302.10174v2 [cs.CV], 01/04/2024).
- Tác giả: Utkarsh Ojha*, Yuheng Li*, Yong Jae Lee (University of Wisconsin-Madison); * đóng góp ngang nhau.
- Venue: không nêu trong file. (fake_or_jpeg_2024 trích dẫn là CVPR 2023, pp. 24480–24489 — thông tin từ paper khác.)
- Mã/mô hình/dữ liệu: https://github.com/Yuheng-Li/UniversalFakeDetect (Abstract).
- Lưu ý đọc: Bảng 2–3 bị vỡ; số tổng (mAP/Acc trung bình) lấy từ văn bản; số từng cột chỉ dùng khi chắc chắn.

## 1. Research Problem
Detector real-vs-fake huấn luyện trên một họ generator (GAN) thất bại trên họ khác (khuếch tán, tự hồi quy). Bài toán: phát hiện ảnh giả từ generator chưa từng thấy khi chỉ có dữ liệu của **một** generator lúc huấn luyện (Abstract; §1).

## 2. Motivation
- Phân tích (§3.2): bộ phân loại huấn luyện trên ProGAN coi lớp "thật" như "lớp rác" (sink class) chứa mọi thứ không phải vân tay GAN; trên LDM/Guided/DALL-E độ chính xác "giả" chỉ 3,05–4,9% trong khi "thật" ~99% (Bảng 1: Real acc 98,64–99,61; Fake acc 62,91/59,1/3,05/4,67/4,9 cho CycleGAN/GauGAN/LDM/Guided/DALL-E).
- Giả thuyết: mạng dễ bám vào các dấu vết mức thấp (phổ tần số, Hình 3), nên ranh giới quyết định gắn chặt với miền giả đã thấy.

## 3. Methodology
- Ý tưởng: phân loại trong **không gian đặc trưng không được huấn luyện cho real/fake** (§4). Hai biến thể: **Nearest Neighbor** (khoảng cách cosine tới ngân hàng đặc trưng thật/giả) và **Linear probing** (một lớp tuyến tính + sigmoid, BCE).
- Đặc trưng ϕ: "lớp cuối của bộ mã hóa thị giác CLIP:ViT-L/14", **768 chiều** (§4: "we use the last layer of CLIP:ViT-L/14's visual encoder as ϕ"; "feature representations (of 768 dimensions)"). Paper **không nói rõ** đó là đầu ra sau lớp chiếu hay CLS token trước chiếu; chỉ nêu số chiều 768.
- Lý do chọn ViT-L/14: được huấn luyện trên 400M cặp ảnh-văn bản; patch 14×14 nhỏ hơn các biến thể ViT khác nên "có thể" mô hình hóa chi tiết mức thấp (§4).
- NN: dự đoán giả nếu khoảng cách cosine nhỏ nhất tới đặc trưng giả < tới đặc trưng thật (§4). Encoder luôn đóng băng.
- Linear: chỉ huấn luyện lớp tuyến tính (khoảng 768 tham số, §4). Huấn luyện **có augmentation Blur+JPEG** (xác suất 0,5), theo khuyến nghị của Wang et al. (Phụ lục B). Không áp dụng augmentation lúc test.
- Tiền xử lý ảnh (resize/crop/chuẩn hóa) vào CLIP: **Không đề cập trong paper** (chỉ nêu các generator sinh ảnh 256×256, §5.1).

## 4. Architecture
CLIP ViT-L/14 visual encoder (đóng băng) → vector 768-d → (a) NN theo cosine hoặc (b) Linear + sigmoid. Không có kiến trúc khác.

## 5. Dataset
- Huấn luyện: ProGAN (20 danh mục LSUN × 18k ảnh giả/danh mục; tổng 720k ảnh: 360k thật + 360k giả) (§3.1, §6.5). Thử thêm LDM + LAION: 400k thật + 400k giả (§6.4).
- Đánh giá: ProGAN, StyleGAN, BigGAN, CycleGAN, StarGAN, GauGAN, CRN, IMLE, SAN, SITD, DeepFakes; Guided (ImageNet); LDM (3 biến thể 200 step / 200 CFG / 100 step); Glide (3 biến thể 100-27 / 50-27 / 100-10); DALL-E-mini (§5.1). Ảnh thật cho LDM/Glide/DALL-E lấy từ LAION. Tất cả 256×256.
- Quy mô huấn luyện: giảm xuống 8k–200k vẫn gần như giữ tổng quát hóa (§6.5, Hình 8; có thể giảm 3–4 lần mà không mất nhiều).

## 6. Experimental Setup
- Metric: Average Precision (AP) và Accuracy; ngưỡng cho baseline tune trên tập validation của generator huấn luyện; thêm baseline "Oracle" tune ngưỡng trên từng tập test (§5.3).
- Baseline: ResNet-50 (Wang et al.) Blur+JPEG 0,1 và 0,5; ViT:CLIP huấn luyện đầy đủ (B+J 0,5); patch classifier (ResNet50-Layer1, Xception-Block2); co-occurrence; Freq-spec (§5.2).
- Siêu tham số: baseline ViT:CLIP — batch size 256, learning rate 5e-5 (Phụ lục B). Lr/số epoch của linear probe: câu mô tả bị cắt trong bản chuyển đổi ("We use the same ..."), **không đọc được**.
- Phần cứng: **Không đề cập** trong phần đã đọc.
- Bền vững: Gaussian blur và JPEG (Hình 10; số liệu trong hình, **không đọc được**).

## 7. Results
- Tổng quát (§1, §6.1): so với SoTA, NN cải thiện +15,05 mAP và +25,90% acc, linear +19,49 mAP và +23,39% acc trên nhóm khuếch tán + tự hồi quy (LDM+Glide+Guided+DALL-E). (Abstract ghi "+15,07 mAP"; §1 ghi "+15,05 mAP" — hai chỗ khác nhau trong cùng paper.)
- §6.1: trong họ GAN, NN ~93%, linear ~95% acc trung bình so với ~85% của trained deep network tốt nhất; trên khuếch tán/tự hồi quy NN 82–84%, linear ~82% so với 53–58% của baseline huấn luyện.
- Bảng 2 (mAP): Linear (LC) tổng "Total" 93,38; NN k=1 90,32; Trained deep network Blur+JPEG(0,1) 83,58; ViT:CLIP (B+J 0,5) 73,44.
- Bảng 3 (Avg. acc): NN k=9 84,25; NN k=5 83,72; NN k=1 82,30; LC 81,38; Oracle∗(B+J 0,5) 76,26; Trained Blur+JPEG(0,1) 69,58; ViT:CLIP (B+J 0,5) 60,57.
- §6.2: NN k=9 hơn baseline Oracle 7,99 điểm % (84,25 vs 76,26).
- §6.3 (Hình 5/6): CLIP:ViT-L/14 tốt hơn CLIP:ResNet-50, ImageNet:ResNet-50, ImageNet:ViT-B/16; pretraining data quan trọng hơn kiến trúc (số từng cột trong hình, không đọc được).
- §6.7: bền với blur/JPEG khi huấn luyện có aug; ở σ=2 ảnh "quá mờ", tác giả cho rằng bền ở mức cực đoan ít quan trọng.
- §6.4: huấn luyện trên LDM, phát hiện GAN trung bình 97,32 mAP; baseline 60,17 mAP (+37,16).

## 8. Limitations
- *Paper tự nêu (§7):* chưa hiểu "điểm chung" giữa ảnh giả GAN và khuếch tán; baseline thất bại nhưng nguyên nhân chỉ là giả thuyết.
- *Tôi nhận thấy:* (1) Không có cơ chế tiền xử lý nêu rõ; (2) linear probe huấn luyện với aug Blur+JPEG, khác với AEGIS (chỉ ảnh sạch); (3) đánh giá chủ yếu trên ảnh 256×256 sinh từ LAION/LSUN, không có đánh giá trên GenImage; (4) số liệu bền vững nằm trong hình; (5) không báo cáo hiệu chuẩn xác suất (ECE).

## 9. Reproducibility
- Công khai mã, mô hình, dữ liệu (Abstract). Thiếu: tiền xử lý ảnh, lr/epoch của linear head (đoạn bị cắt). Chạy lại trên RTX 4060 8 GB: rất khả thi (encoder đóng băng, head 768 tham số; trích đặc trưng một lần rồi cache).

## 10. AEGIS Integration
- **Expert 1 (CLIP, UnivFD-style):** ViT-L/14 đóng băng, head tuyến tính trên vector 768-d (xem `METHOD_SPECS.md`, câu b). Sinh ra xác suất P1 → Calibration (temperature scaling) → Gating.
- NN-cosine với tâm/ngân hàng đặc trưng: gợi ý cho **OOD Profiler** (cosine distance tới tâm `train_experts`) và cho **Few-shot** (prototype), nhưng PROJECT_BRIEF §5.4 đã chốt không cập nhật prototype trong few-shot.
- Với giới hạn local: cache embedding `.pt` float16.

## 11. Research Opportunity (đề xuất của tôi)
- UnivFD chưa kiểm tra ảnh bị nén/mờ khi head chỉ huấn luyện trên ảnh sạch → AEGIS S2/S3 sẽ cho số liệu mới cho đúng cấu hình "frozen CLIP + clean-only linear head".
- Có thể thử k-NN-cosine làm tín hiệu OOD thứ hai bên cạnh head tuyến tính (đề xuất, không phải của paper).
