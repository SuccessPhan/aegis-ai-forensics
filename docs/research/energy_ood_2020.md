# Phân tích paper: energy_ood_2020

**Thông tin thư mục (từ `docs/raw_papers/energy_ood_2020.md`):**
- Tiêu đề: *Energy-based Out-of-distribution Detection* (arXiv:2010.03759v4 [cs.LG], 26/04/2021).
- Tác giả: Weitang Liu (UC San Diego), Xiaoyun Wang (UC Davis), John D. Owens (UC Davis), Yixuan Li (UW-Madison).
- Venue: 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver.
- Mã: https://github.com/wetliu/energy_ood (chú thích 1).
- Lưu ý đọc: Bảng 1–2 đọc khá rõ; Phụ lục A, B (kết quả từng dataset, Hình 4 về nhiệt độ T, margin tối ưu) được tham chiếu nhưng **không có trong file Markdown**. File chứa một byte NUL nên công cụ grep coi là "binary" (không ảnh hưởng nội dung đọc).

## 1. Research Problem
Phát hiện đầu vào ngoài phân phối (OOD) cho bộ phân loại đã huấn luyện; điểm "softmax confidence" quá tự tin với OOD (Abstract; §1).

## 2. Motivation
Softmax có thể cho độ tin cậy tùy ý cao với đầu vào xa dữ liệu huấn luyện (§1). Ví dụ minh họa (Hình 2): mẫu trong phân phối và mẫu OOD (SVHN) có softmax 1,0 vs 0,99 nhưng negative energy 11,19 vs 7,11.

## 3. Methodology
- Năng lượng tự do (Eq. 4): `E(x; f) = −T · log Σ_i exp(f_i(x)/T)` trên logit f(x)∈R^K. Dùng điểm âm năng lượng `−E`: cao ⇒ trong phân phối. Phát hiện OOD: `g(x)=1 nếu −E(x;f) > τ` (Eq. 7; theo lời §3.1 "Examples with higher energies are considered as OOD").
- Quan hệ với softmax (§3.2): `log max_y p(y|x) = E(x; f(x)−f_max(x)) = E(x;f) + f_max(x)` khi T=1 — softmax là năng lượng sau khi dịch logit theo giá trị lớn nhất, làm mất thông tin độ lớn logit.
- **Dùng không cần huấn luyện lại** (inference-time, "parameter-free") cho mọi mạng đã huấn luyện; hoặc **energy-bounded fine-tuning** (Eq. 8–10): `CE + λ·L_energy` với hai hinge bình phương, `m_in` cho dữ liệu trong phân phối và `m_out` cho dữ liệu OOD phụ trợ (cần dữ liệu OOD).
- Nhiệt độ: T=1 là mặc định; T lớn làm phân phối đều hơn, energy kém phân biệt (§4.2).

## 4. Architecture
Không có kiến trúc mới. WideResNet cho CIFAR/SVHN (§4.1). Chỉ thêm `logsumexp` trên logit.

## 5. Dataset
In-dist: SVHN, CIFAR-10, CIFAR-100. OOD test (6 benchmark): Textures, SVHN, Places365, LSUN-Crop, LSUN-Resize, iSUN. Dữ liệu OOD phụ trợ cho fine-tune: 80 Million Tiny Images đã loại trùng CIFAR (§4.1).

## 6. Experimental Setup
Metric: FPR95, AUROC, AUPR (§4.1). Fine-tune: λ=0,1; 10 epoch; lr 0,001 cosine; batch 128 (in) / 256 (OOD); `m_in ∈ {−3,−5,−7}`, `m_out ∈ {−15,−19,−23,−27}` chọn theo FPR95 trên validation (§4.1, như Hendrycks et al.). Baseline: softmax score, ODIN, Mahalanobis, Outlier Exposure.

## 7. Results
- **Bảng 1 (CIFAR-10, WideResNet, pretrained; softmax / energy; trung bình 6 OOD):** FPR95 51,04 / 33,01; AUROC 90,90 / 91,88; AUPR 97,92 / 97,83.
- **Bảng 2 (trung bình 6 OOD):** CIFAR-10 — Energy 33,01 FPR95, AUROC 91,88; Mahalanobis 37,08 / 93,27; ODIN 35,71 / 91,09; OE 8,53 / 98,30; Energy fine-tune 3,32 / 98,92. CIFAR-100 — Softmax 80,41 / 75,53; Energy 73,60 / 79,56; Mahalanobis 54,04 / 84,12; Energy fine-tune 47,55 / 88,46.
- Tóm tắt: energy giảm FPR95 trung bình 18,03% so với softmax trên CIFAR-10 pretrained (Abstract); 8,69% trên SVHN in-dist (§4.2); fine-tune giảm thêm 5,20% so với OE (CIFAR-10) và 10,55% (CIFAR-100).
- Lưu ý: ở một số OOD (Textures, SVHN) và ở AUROC/AUPR, energy pretrained **không** hơn softmax (Bảng 1: AUROC Texture 88,50 vs 85,22; SVHN 91,89 vs 90,96; AUPR trung bình 97,92 vs 97,83).
- *Không nhất quán nhỏ:* Bảng 2 ghi test error của energy fine-tune 4,87%, văn bản ghi 4,98%.

## 8. Limitations
- *Paper tự nêu:* margin quá nhỏ cho dữ liệu trong phân phối gây khó tối ưu (§4.2).
- *Tôi nhận thấy:* (1) thí nghiệm đa lớp K=10/100 với WideResNet trên ảnh 32×32; **không có bài toán nhị phân**, không có backbone đóng băng, không có ảnh suy giảm; (2) ngưỡng τ chọn trên dữ liệu trong phân phối; (3) energy không mang lại lợi ích tuyệt đối ở mọi OOD; (4) bản fine-tune cần dữ liệu OOD phụ trợ — không dùng được ở AEGIS (không có nhãn OOD hợp lệ, tránh self-labeling).

## 9. Reproducibility
Mã công khai (chú thích 1). Phiên bản inference chỉ là `logsumexp`, trivially khả thi. Fine-tune ngoài phạm vi.

## 10. AEGIS Integration
- **OOD Profiler:** trả lời câu d trong `METHOD_SPECS.md`. Kết luận ngắn (suy luận đại số của tôi, không phải của paper): với **head nhị phân một logit z**, phải tự chọn cách ghép thành hai logit. Nếu đối xứng [−z/2, +z/2] thì `−E = |z|/2 + log(1+e^{−|z|})` là hàm đơn điệu của |z|, tức **dư thừa với entropy** (cũng chỉ phụ thuộc |z|). Nếu không đối xứng (ví dụ [0, z]) thì energy đơn điệu theo z (mang thông tin lớp dự đoán chứ không phải "độ tự tin") và phụ thuộc phép chọn tham chiếu tùy ý. Energy chỉ có ý nghĩa "mật độ" theo paper khi head có **hai đầu ra softmax tự do** (K=2), vì lúc đó độ lớn logit không bị ràng buộc. Paper không bàn trường hợp nhị phân.
- Nếu thêm: tính trên **logit thô (T=1, trước calibration)**, vì paper chỉ ra T≠1 làm giảm khả năng phân biệt (§4.2).

## 11. Research Opportunity (đề xuất của tôi)
- Thử thêm đặc trưng energy (K=2 logit) vào bộ điều phối như ablation nhỏ (12 → 15 chiều) và kiểm tra AUROC OOD của S4 so với entropy/cosine.
- Kiểm tra liệu "độ lớn logit" có tương quan với độ suy giảm JPEG/blur — điều paper chưa xét.
