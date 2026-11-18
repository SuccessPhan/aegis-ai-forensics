# Phân tích paper: freqnet_2024

**Thông tin thư mục (từ `docs/raw_papers/freqnet_2024.md`):**
- Tiêu đề: *Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning* (mô hình FreqNet; arXiv:2403.07240v1 [cs.CV], 12/03/2024).
- Tác giả: Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, Yunchao Wei (Beijing Jiaotong University; Yanshan University; A*STAR).
- Venue: bản quyền ghi "Copyright © 2024, Association for the Advancement of Artificial Intelligence (www.aaai.org)".
- Mã: https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection (Abstract).
- Lưu ý đọc: Bảng 1–2 vỡ thành cột; số "Mean" lấy từ hai cột cuối mỗi hàng; số tổng ở văn bản/Bảng 3 đáng tin hơn.

## 1. Research Problem
Detector tần số cho ảnh GAN dễ quá khớp vào dấu vết của nguồn huấn luyện; bài toán: detector deepfake tổng quát hóa sang nguồn chưa thấy với ít dữ liệu huấn luyện (Abstract; §Introduction).

## 2. Motivation
Phổ FFT trung bình khác nhau giữa các GAN và thậm chí giữa cùng một GAN huấn luyện trên dữ liệu khác (Hình 2; trung bình 2000 ảnh); CNN huấn luyện trên một nguồn quá khớp vào mẫu phổ đó. Các phương pháp tần số trước (Frank, Durall, F3Net, BiHPF, FrePGAN) khái quát kém (§Introduction).

## 3. Methodology
- **HFRI (High-Frequency Representation of Image):** FFT ảnh (dịch tần số 0 về tâm) → bộ lọc thông cao `B_h` (triệt phần tần số thấp ở trung tâm với |i|<W/4, |j|<H/4) → iFFT về **miền ảnh** → đầu vào CNN (Eq. 3–5). Tức đầu vào là **ảnh high-pass trong miền không gian**, không phải phổ log-magnitude.
- **HFRF (High-Frequency Representation of Feature):** áp dụng lọc thông cao lên feature map theo chiều không gian (W,H) và theo chiều kênh C (Eq. 6).
- **FCL (Frequency Convolutional Layer):** FFT feature map, áp lớp conv riêng lên **phổ biên độ** và **phổ pha**, iFFT quay lại (Eq. 7).
- Loss: cross-entropy chuẩn (Eq. 8). Không pretrain (§Implementation Details).

## 4. Architecture
CNN residual nhẹ, không tiền huấn luyện, xen các khối HFRI/HFRF/FCL, kết thúc global average pooling + FC (Hình 3). **1,9 triệu tham số** (Abstract; Bảng 3). Số block/kênh chi tiết: **Không đọc được từ văn bản** (nằm trong Hình 3).

## 5. Dataset
- Huấn luyện: ForenSynths (Wang et al. 2020): 20 danh mục, mỗi danh mục 18.000 ảnh ProGAN + số ảnh thật LSUN tương đương; dùng thiết lập 1/2/4 lớp: (horse), (chair, horse), (car, cat, chair, horse) (§Datasets).
- Test: bộ test ForenSynths (8 generator: ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake; ảnh thật từ 6 dataset) + 9 GAN tự tổng hợp (AttGAN, BEGAN, CramerGAN, InfoMaxGAN, MMDGAN, RelGAN, S3GAN, SNGAN, STGAN) → 17 mô hình; 36K ảnh test cân bằng; thêm tập khuôn mặt 20.000 thật (CelebA-HQ) + 60.000 giả (ProGAN, StyleGAN, StyleGAN2). **Không có generator khuếch tán, không có GenImage.**

## 6. Experimental Setup
Adam, learning rate khởi tạo "2 × 10^-2" (như in; giá trị cao bất thường, tôi không xác minh được), batch 32, 100 epoch, giảm lr 20% mỗi 10 epoch; PyTorch, RTX 3090; `torch.fft.fftn` (§Implementation Details). Chỉ số: Acc và AP. Baseline: Wang(2020), Frank(2020), Durall(2020), F3Net(2020), BiHPF(2022a), FrePGAN(2022c), LGrad(2023), Ojha(2023) (=UnivFD). **Không có đánh giá nén JPEG/blur/resize** (không xuất hiện từ khóa này trong file).

## 7. Results
- **Bảng 3 (17 mô hình, mAcc; tham số):** FreqNet 92,8 (1,9 M) vs Ojha 83,0 (304,0 M) → +9,8; F3Net 77,8 (48,9 M); LGrad 76,8 (46,6 M).
- **Bảng 1 (ForenSynths, mean Acc/AP):** 1-class — FreqNet 89,6/97,6; Ojha 89,2/97,8; F3Net 78,3/87,3; Frank 65,7/63,3; Durall 68,7/65,0. 2-class — FreqNet 91,0/97,9; Ojha 88,1/97,8. 4-class — FreqNet 91,5/98,5; Ojha 89,1/98,3; F3Net 80,4/86,2.
- **Bảng 2 (9 GAN tự tổng hợp, 4-class):** FreqNet mean Acc 94,0; Ojha 77,6.
- **Ablation (Bảng 4, mean Acc ForenSynths):** bản đầy đủ 91,5; bỏ lần lượt các module cho 82,0–87,8 (cột nào tương ứng module nào bị vỡ bảng, không gán chắc được).
- Mặt: ProGAN 98,7%, StyleGAN 99,0%, StyleGAN2 99,5% (§Deepfake Performance).
- *Mâu thuẫn nội bộ nhỏ:* văn bản nói gain so với Ojha là "2,9% (1-class) và 0,4% (2-class)", nhưng Bảng 1 cho 1-class +0,4 (89,6 vs 89,2) và 2-class +2,9 (91,0 vs 88,1) → có vẻ văn bản đảo thứ tự.

## 8. Limitations
- *Paper tự nêu:* hầu như không có mục limitations; chỉ ghi "in the revised version we will expound further" về ablation.
- *Tôi nhận thấy:* (1) chỉ GAN, không khuếch tán — mâu thuẫn với bối cảnh GenImage; (2) không kiểm tra JPEG/blur/resize; (3) lr "2e-2" cần kiểm tra; (4) "state-of-the-art +9,8%" so với Ojha trên tập 17 GAN chứ không phải dữ liệu khuếch tán; (5) cơ chế gốc là lọc cứng ở miền ảnh/feature — có thể nhạy JPEG (suy luận, không có số liệu trong paper).

## 9. Reproducibility
Mã công khai (Abstract). Thiếu: chi tiết kiến trúc dạng văn bản, seed. Chạy lại trên RTX 4060 8 GB: khả thi (1,9M tham số). Với AEGIS không cần chạy lại bản gốc; chỉ cần ý tưởng.

## 10. AEGIS Integration
- **Expert 3 (tần số)**: là minh chứng rằng một CNN ~1,9M tham số huấn luyện từ đầu có thể đạt kết quả cạnh tranh trên GAN — hỗ trợ lựa chọn mạng nhỏ ~1,2M của AEGIS.
- Khác biệt với AEGIS: FreqNet dùng ảnh high-pass trong miền không gian + conv trên phổ của feature, không dùng log-magnitude FFT làm đầu vào. Xem `METHOD_SPECS.md`, câu c.
- Gợi ý ablation nội bộ: thêm biến thể "ảnh high-pass iFFT" làm đầu vào thứ hai.

## 11. Research Opportunity (đề xuất của tôi)
- FreqNet chưa từng được thử trên khuếch tán + suy giảm; AEGIS (ADM/SD1.4 huấn luyện; MJ/GLIDE OOD) có thể cung cấp bằng chứng mới ở quy mô nhỏ cho họ phương pháp "tần số nhẹ".
- Hạ độ phức tạp: chỉ dùng HFRI (bước lọc thông cao đầu vào) vì không cần lớp FCL; ablation Bảng 4 gợi ý các module đều giúp nhưng khó tách số cụ thể.
