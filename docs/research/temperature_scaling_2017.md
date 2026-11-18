# Phân tích paper: temperature_scaling_2017

**Thông tin thư mục (từ `docs/raw_papers/temperature_scaling_2017.md`):**
- Tiêu đề: *On Calibration of Modern Neural Networks* (arXiv:1706.04599v2 [cs.LG], 03/08/2017).
- Tác giả: Chuan Guo*, Geoff Pleiss*, Yu Sun*, Kilian Q. Weinberger (Cornell University); * đóng góp ngang nhau.
- Venue: Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70, 2017.
- Mã tham chiếu: http://github.com/gpleiss/temperature_scaling (chú thích 4).
- Lưu ý đọc: Bảng 1 bị vỡ nhưng cột đọc theo thứ tự tiêu đề (Uncalibrated, Hist. Binning, Isotonic, BBQ, Temp. Scaling, Vector Scaling, Matrix Scaling).

## 1. Research Problem
Mạng nơ-ron hiện đại hiệu chuẩn kém (độ tự tin vượt độ chính xác thật); tìm nguyên nhân và phương pháp hậu xử lý để xác suất phản ánh đúng khả năng đúng của dự đoán (Abstract; §1–2).

## 2. Motivation
Quyết định trong hệ thống thực cần độ tin cậy có ý nghĩa xác suất (xe tự lái, chẩn đoán y tế) và để diễn giải/kết hợp với mô hình khác (§1). Mạng cũ (LeNet) hiệu chuẩn tốt; ResNet 110 lớp quá tự tin (Hình 1).

## 3. Methodology
- Định nghĩa hiệu chuẩn hoàn hảo `P(Ŷ=Y | P̂=p)=p` (Eq. 1); **ECE** chia M bin bằng nhau, `ECE = Σ_m |B_m|/n · |acc(B_m) − conf(B_m)|` (Eq. 3); MCE (Eq. 5); NLL (Eq. 6). Thực nghiệm dùng **M = 15 bin** (Bảng 1).
- Nguyên nhân (§3): tăng dung lượng (độ sâu, độ rộng), Batch Normalization, giảm weight decay làm tăng ECE; NLL quá khớp mà lỗi 0/1 không quá khớp (Hình 2–3).
- Phương pháp hậu xử lý (§4), cần tập hold-out validation, tham số mạng cố định:
  - Nhị phân (§4.1): histogram binning, isotonic regression, BBQ, **Platt scaling** `q̂ = σ(a·z + b)`, tối ưu NLL trên validation.
  - Đa lớp (§4.2): matrix/vector scaling; **temperature scaling** `q̂ = max_k softmax(z/T)`, một tham số T>0 cho mọi lớp, tối ưu NLL trên validation; T>1 làm mềm softmax; vì T không đổi argmax nên **không đổi độ chính xác**; T→∞ tiến về 1/K.
- Giả thiết ngầm: huấn luyện, validation và test **cùng phân phối** (§4: "We assume that the training, validation, and test sets are drawn from the same distribution").

## 4. Architecture
Không phải đề xuất kiến trúc; hậu xử lý thêm 1 tham số vô hướng (chèn `nn.MulConstant(1/T)` giữa logit và softmax trong Torch7; huấn luyện với T=1, rồi tìm T trên validation) (§5, "Ease of implementation").

## 5. Dataset
Ảnh: Caltech-UCSD Birds (5994/2897/2897 train/val/test), Stanford Cars (8041/4020/4020), ImageNet 2012 (1,3M/25.000/25.000), CIFAR-10/100 (45.000/5.000/10.000), SVHN (598.388/6.000/26.032). Văn bản: 20 News, Reuters, SST (§5).

## 6. Experimental Setup
ResNet, ResNet stochastic depth, Wide ResNet, DenseNet; Birds/Cars tinh chỉnh từ ImageNet; DAN và TreeLSTM cho văn bản; dùng tiền xử lý/siêu tham số theo từng paper gốc (§5). Metric chính ECE (15 bin); MCE/NLL/error ở phụ lục S3 (không có trong file chính).

## 7. Results
**Bảng 1 (ECE %; Uncalibrated → Temperature scaling):** CIFAR-100 ResNet 110: 16,53 → 1,26; CIFAR-100 Wide ResNet 32: 15,0 → 2,32; CIFAR-10 ResNet 110: 4,6 → 0,83; Birds ResNet 50: 9,19 → 1,85; ImageNet DenseNet 161: 6,28 → 1,99; ImageNet ResNet 152: 5,48 → 1,86; 20 News DAN 3: 8,02 → 4,11.
- Kết luận (§5): temperature scaling hiệu quả nhất trên vision, ngang các phương pháp khác trên NLP; vector scaling gần như trùng nghiệm với temperature (vector học ra gần hằng số → "miscalibration có chiều thấp nội tại"); matrix scaling kém khi nhiều lớp (quá khớp tập validation nhỏ); binning thường đổi dự đoán lớp → ảnh hưởng accuracy.
- Tốc độ: T tối ưu trong ~10 vòng conjugate gradient, "a fraction of a second" (§5).

## 8. Limitations
- *Paper tự nêu:* chưa hiểu vì sao dung lượng/chuẩn hóa/chính quy hóa tác động hiệu chuẩn (§6); Reuters/SVHN vốn đã hiệu chuẩn tốt nên TS không cải thiện (§5).
- *Tôi nhận thấy:* (1) giả thiết cùng phân phối train/val/test — **vi phạm trong S2/S3/S4 của AEGIS** (phân phối test khác val khi ảnh suy giảm hoặc OOD); (2) thực nghiệm toàn đa lớp (K ≥ 10); không có thí nghiệm nhị phân thuần với temperature scaling (chỉ định nghĩa Platt ở §4.1); (3) không thử trên backbone foundation đóng băng/head tuyến tính; (4) không xét ECE dưới dịch phân phối.

## 9. Reproducibility
Mã tham khảo công khai (chú thích 4). Cực nhẹ: một tham số, tối ưu trong chưa tới 1 giây. Chạy hoàn toàn khả thi trên CPU với logit đã cache.

## 10. AEGIS Integration
- **Calibration (per expert):** tối ưu T cho từng expert trên `val_gating_fit` (gồm cả ảnh suy giảm) bằng NLL; ECE 15 bin là chỉ số báo cáo (khớp danh mục chỉ số bắt buộc).
- **Evaluation:** thêm reliability diagram/ECE trước-sau calibration cho từng expert và cho P_final.
- Quan hệ với energy: xem `energy_ood_2020.md`; energy tính trên logit thô (T=1) — ghi chú ở `METHOD_SPECS.md`, câu d.
- Với head nhị phân một logit: dùng `σ(z/T)` — **đây là suy ra của tôi**; paper chỉ định nghĩa temperature scaling đa lớp (Eq. 9) và Platt scaling nhị phân (§4.1), và nói TS là "biến thể một tham số của Platt scaling" (§1).

## 11. Research Opportunity (đề xuất của tôi)
- Kiểm chứng thực nghiệm giả thiết "cùng phân phối": so sánh ECE sau calibration trên S1, S2, S3, S4 để lượng hóa việc T fit trên `val_gating_fit` suy yếu thế nào dưới dịch phân phối — đây là lý do biện hộ cho gating (kết hợp với OOD profiler).
- Thử T phụ thuộc lớp suy giảm (ví dụ nhóm theo `jpeg_quality_est`) so với một T chung — lưu ý nguy cơ quá khớp vì mỗi nhóm có ít mẫu hơn.
