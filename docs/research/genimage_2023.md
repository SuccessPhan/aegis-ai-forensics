# Phân tích paper: genimage_2023

**Thông tin thư mục (từ `docs/raw_papers/genimage_2023.md`):**
- Tiêu đề: *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image* (arXiv:2306.08571v2 [cs.CV], 24/06/2023; ghi "Preprint. Under review").
- Tác giả: Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang (Huawei Noah's Ark Lab).
- Venue: không nêu trong file (preprint). (Một paper khác trong thư mục, fake_or_jpeg_2024, trích dẫn bản này với venue NeurIPS 36 — nhưng đó là thông tin từ paper khác, không phải từ file này.)
- Lưu ý đọc: bảng bị vỡ thành cột dọc; các số dưới đây gán cột theo thứ tự tiêu đề bảng. Hình 3 (ma trận) là ảnh, không đọc được.

## 1. Research Problem
Xây dựng benchmark quy mô triệu ảnh (thật ImageNet vs ảnh sinh bởi 8 generator) và đề xuất hai nhiệm vụ đánh giá: **cross-generator classification** và **degraded image classification** (Abstract; §1).

## 2. Motivation
Thiếu dataset lớn chứa ảnh từ các generator tiên tiến; dataset cũ tập trung khuôn mặt (UADFV, FakeSpotter, DFFD, ForgeryNet...) hoặc chỉ GAN (CNNSpot) hoặc quy mô nhỏ (Bảng 1). Con người chỉ đạt 61,3% khi phân biệt ảnh thật/giả (§1, trích dẫn từ nghiên cứu khác).

## 3. Methodology
- Dùng toàn bộ ảnh ImageNet làm ảnh thật; sinh ảnh giả theo 1000 nhãn ImageNet bằng prompt "photo of class" (§2.1). ADM và BigGAN dùng mô hình pretrain ImageNet với nhãn lớp; Wukong dịch prompt sang tiếng Trung (§2.1).
- Nhiệm vụ 1: huấn luyện trên một generator, test trên các generator khác (§3.2). Nhiệm vụ 2: huấn luyện trên SD v1.4, chỉ làm suy giảm ảnh test: giảm độ phân giải 112 và 64, JPEG QF 65 và 30, Gaussian blur σ=3 và σ=5 (§3.3, Bảng 5).
- Các detector đánh giá: ResNet-50, DeiT-S, Swin-T (**huấn luyện từ đầu**, §3.2), CNNSpot, Spec, F3Net, GramNet (§3.1).
- Hàm mất mát/siêu tham số huấn luyện: **Không đề cập trong bản chuyển đổi**.

## 4. Architecture
Không có kiến trúc mới (benchmark paper). Backbone baseline: ResNet-50 (CNN), DeiT-S, Swin-T (Transformer), có cùng mức chi phí tính toán/tham số (§3.2). Spec nhận phổ tần số làm input thay vì pixel (§3.2). F3Net có hai nhánh FAD và LFS (§3.2).

## 5. Dataset
- Tổng 2.681.167 ảnh: 1.331.167 thật (1.281.167 train + 50.000 test) và 1.350.000 giả (§2.1; Bảng 1: Real 1.331.167, Fake 1.350.000).
- Mỗi lớp ImageNet sinh 1350 ảnh giả: 1300 train, 50 test; mỗi generator ~162 train + 6 test mỗi lớp (riêng SD v1.5: 166 + 8) (§2.1). Ảnh thật không dùng chung giữa các subset.
- Độ phân giải generator (§2.2): Midjourney V5 1024×1024; Wukong 512×512; SD v1.4 và v1.5 512×512; ADM (có classifier guidance, pretrain ImageNet) — độ phân giải **không được nêu trong §2.2 của file này**; GLIDE 256×256 (classifier-free guidance); VQDM 256×256; BigGAN 128×128. Paper fake_or_jpeg_2024 (§3.2) ghi ADM = 256×256 (thông tin từ paper khác).
- Paper **không nhắc** định dạng JPEG/PNG; thiên lệch định dạng do fake_or_jpeg_2024 chỉ ra.

## 6. Experimental Setup
- Cross-generator: Bảng 2 (8 mô hình × 8 test), Bảng 3 (huấn luyện SD v1.4, test 8 generator, trung bình 8 số), Bảng 4 (8 mô hình huấn luyện, mỗi cột trung bình 8 mô hình; Avg là trung bình 64 số) (§3.2).
- Chỉ số: Accuracy (%). Phần cứng, learning rate, epoch: **Không đề cập**.
- CNNSpot dùng augmentation: blur σ∼U[0,3] xác suất 50%, JPEG xác suất 50% (§3.2).

## 7. Results (các số liên quan ADM, SD v1.4, Midjourney, GLIDE)
Thứ tự cột các bảng: Midjourney, SD V1.4, SD V1.5, ADM, GLIDE, Wukong, VQDM, BigGAN, Avg.

**Bảng 2 (ResNet-50, hàng = tập huấn luyện):**
- Huấn luyện **SD V1.4**: MJ 54,9 | SD1.4 99,9 | ADM 53,5 | GLIDE 61,9 (Avg 72,1).
- Huấn luyện **ADM**: MJ 58,6 | SD1.4 53,1 | ADM 99,0 | GLIDE **97,1** | BigGAN 88,3 (Avg 70,4).
- Huấn luyện **GLIDE**: MJ 50,7 | SD1.4 50,0 | ADM 56,0 | GLIDE 99,9 | BigGAN 74,0 (Avg 60,2).
- Huấn luyện **Midjourney**: MJ 98,8 | SD1.4 76,4 | ADM 64,1 | GLIDE 78,9 (Avg 71,1).

**Bảng 3 (huấn luyện SD v1.4):** ResNet-50: MJ 54,9 / SD1.4 99,9 / ADM 53,5 / GLIDE 61,9 / Avg 72,1. Swin-T: 62,1 / 99,9 / 49,8 / 67,6 / Avg 74,8. DeiT-S: 55,6 / 99,9 / 49,8 / 58,1 / Avg 71,6. F3Net: 50,1 / 99,9 / 49,9 / 50,0 / Avg 68,7. Spec: 52,0 / 99,4 / 49,7 / 49,8 / Avg 68,8. CNNSpot: 52,8 / 96,3 / 50,1 / 39,8 / Avg 64,2.

**Bảng 4 (trung bình 8 mô hình huấn luyện, test từng generator):** ResNet-50: MJ 59,0, SD1.4 72,3, ADM 59,7, GLIDE 73,1 (Avg 66,9). Swin-T: 61,7 / 76,0 / 61,3 / 76,9 (Avg 70,3). F3Net: 55,1 / 73,1 / 66,5 / 57,8 (Avg 64,6). Spec: 56,7 / 72,4 / 57,9 / 65,4 (Avg 65,1).

**Bảng 5 (suy giảm; huấn luyện SD v1.4; cột LR112, LR64, JPEG65, JPEG30, Blur σ3, Blur σ5, Avg):**
- ResNet-50: 96,2 / 57,4 / 51,9 / 51,2 / 97,9 / 69,4 / 70,6.
- Swin-T: 97,4 / 54,6 / 52,5 / 50,9 / 94,5 / 52,5 / 67,0.
- CNNSpot: 50,0 / 50,0 / 97,3 / 97,3 / 97,4 / 77,9 / 78,3.
- Spec (nhận phổ tần số): 50,0 / 49,9 / 50,8 / 50,4 / 49,9 / 49,9 / 50,1.
- F3Net: 50,0 / 50,0 / 89,0 / 74,4 / 57,9 / 51,7 / 62,1.
- GramNet: 98,8 / 94,9 / 68,8 / 53,4 / 95,9 / 81,6 / 82,2.
(Mức tin cậy việc gán cột: trung bình — bảng bị vỡ; khớp với văn bản §3.3 "JPEG ảnh hưởng mạnh, CNNSpot bền JPEG/blur".)

**Bảng 6–8:** tăng quy mô dữ liệu (SD v1.4; 1600 ảnh 10 lớp → 162.000 ảnh 1000 lớp: Cross-Generator Acc 60,9 → 72,1 và Identical-Generator Acc 73,6 → 99,9); tăng số lớp (10 lớp 78,8; 50 lớp 80,3; 100 lớp 81,2; 100 lớp×1300 ảnh 98,4); tổng quát hóa nội dung: khuôn mặt LFW 99,9, tranh Laion-Art 95,0 (huấn luyện SD v1.4).

## 8. Limitations
- *Paper tự nêu/ngụ ý:* detector chuyên dụng GAN/mặt (CNNSpot, Spec, F3Net, GramNet) kém backbone thông thường trên dữ liệu khuếch tán; cần thiết kế backbone riêng (§3.2). Midjourney là generator khó tổng quát hóa nhất (§4.4).
- *Tôi nhận thấy:* (1) paper không cân bằng JPEG/kích thước; fake_or_jpeg_2024 chứng minh đây là đường tắt (xem file đó); (2) cross-generator Bảng 3 chỉ huấn luyện trên SD v1.4, trong khi AEGIS huấn luyện ADM + SD1.4; (3) các detector đều huấn luyện từ đầu, không có backbone foundation đóng băng; (4) kịch bản suy giảm chỉ áp lên test, đối chiếu "huấn luyện trên ảnh sạch" — phù hợp với S2/S3 của AEGIS nhưng tham số khác (xem §10).

## 9. Reproducibility
- Dataset công khai: link không có trong file Markdown (Abstract không chứa URL). PROJECT_BRIEF §5.1 ghi nguồn tải thực tế (mirror HF và Baidu/GitHub) — đó là thông tin của dự án, không phải từ paper này.
- Thiếu: siêu tham số huấn luyện các baseline, tiền xử lý chi tiết (resize về 224 hay crop). Chạy lại baseline trên local: không cần thiết cho AEGIS.

## 10. AEGIS Integration
- Module: **Data Layer** (chọn ADM, SD v1.4 = Seen; Midjourney, GLIDE = Unseen cho S4) và **Evaluation** (giao thức cross-generator, suy giảm).
- Đối chiếu tham số suy giảm: GenImage dùng JPEG QF 65/30, blur σ 3/5, hạ độ phân giải 112/64 (cạnh ảnh gốc 256/512). AEGIS dùng JPEG QF 85/65/40, blur k=11 σ=1.5 và k=17 σ=2.5, resize 0.5× rồi phóng lại. Không trùng nên không so số trực tiếp được.
- Dữ liệu tham chiếu (ResNet-50, huấn luyện SD1.4, Bảng 3): tỉ lệ ~50–62% trên MJ/ADM/GLIDE → baseline "đơn expert kém ở OOD" có tiền lệ; riêng Bảng 2 cho thấy ADM→GLIDE đạt 97,1% (cùng độ phân giải 256, cùng nhóm khuếch tán không latent) → **GLIDE có thể không quá OOD đối với expert đã thấy ADM** (nhận xét của tôi, từ số liệu trong paper).

## 11. Research Opportunity (đề xuất của tôi)
- Dùng Bảng 2 làm tiền nghiệm chọn generator "khó nhất" cho S4: Midjourney luôn khó (cột MJ thấp 49,9–64,1 với mọi tập huấn luyện ngoài chính nó); GLIDE dễ khi có ADM trong huấn luyện.
- Báo cáo S4 tách riêng theo Midjourney và GLIDE (không gộp) để thể hiện hai mức OOD khác nhau.
