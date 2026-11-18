# Phân tích paper: fake_or_jpeg_2024

**Thông tin thư mục (chỉ lấy từ file `docs/raw_papers/fake_or_jpeg_2024.md`):**
- Tiêu đề: *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets* (bản preprint, ngày 29/03/2024; arXiv:2403.17608v2 [cs.CV]).
- Tác giả: Patrick Grommelt, Louis Weiss (đóng góp ngang nhau), Franz-Josef Pfreundt, Janis Keuper. Đơn vị: Fraunhofer ITWM; Offenburg University (IMLA); University of Mannheim.
- Venue: preprint (không nêu hội nghị). Trang dữ liệu/mã: https://www.unbiased-genimage.org (ghi ở phần Abstract).
- Lưu ý đọc: bản chuyển đổi từ PDF; các Hình 1–5, 7–10 là ma trận/ảnh nên **không đọc được số liệu từ từng ô**; chỉ dùng được các bảng 1–4 và văn bản.

## 1. Research Problem
Chứng minh rằng các dataset phát hiện ảnh AI (cụ thể là GenImage) chứa hai thiên lệch "đường tắt": (i) **thiên lệch nén JPEG** — ảnh thật lưu JPEG, ảnh AI lưu PNG; (ii) **thiên lệch kích thước ảnh** — ảnh thật có nhiều kích thước, mỗi generator sinh một kích thước cố định. Detector học các đường tắt này, làm đánh giá cross-generator sai lệch và kém bền với JPEG (Abstract; §1; §3).

## 2. Motivation
- Benchmark chưa đáng tin vì detector có thể dựa vào JPEG/kích thước thay vì dấu vết sinh ảnh (§1, §3).
- Hệ quả hai chiều: (a) đánh giá bị thổi phồng vì tập test mang cùng thiên lệch; (b) detector không tổng quát hóa sang dữ liệu không có thiên lệch (thực tế) (§3).
- Khoảng trống: các công trình trước dùng augmentation JPEG ngẫu nhiên nhưng paper lập luận augmentation chỉ "dịch" phân phối QF chứ không cân bằng, và ảnh đã nén bị nén lần hai còn ảnh chưa nén chỉ nén một lần; JPEG không idempotent (§4.1).

## 3. Methodology
- **Chẩn đoán JPEG (§3.1):** (1) Lấy ResNet50 huấn luyện trên GenImage thô, nén ảnh AI ở test với QF giảm dần → độ chính xác giảm mạnh dù QF cao 95; precision giữ ~1, recall tụt (Phụ lục B) → ảnh AI bị nén bị xếp thành "thật". (2) Để loại trừ khả năng nén phá hủy dấu vết sinh ảnh, dùng ảnh thật **chưa nén** FFHQ (PNG, 1024×1024) và nén dần → độ chính xác phân loại "thật" tăng (Bảng 2).
- **Chẩn đoán kích thước (§3.2):** Đo độ chính xác trên ảnh thật ImageNet theo từng khoảng kích thước; độ chính xác thấp ở khoảng kích thước trùng kích thước ảnh AI mà detector đã học (Hình 3).
- **Khắc phục (§4):**
  - *JPEG constraint (§4.1):* chỉ chọn ảnh thật có QF = 96; chọn số lượng ảnh AI bằng nhau và nén chúng cùng QF 96; huấn luyện lại ResNet50. Với JPEG96, nén chỉ áp lên ảnh AI vì phần lớn ảnh thật đã ở QF 96 hoặc thấp hơn.
  - *Size constraint (§4.2):* chỉ dùng các tập generator 512×512 (SD1.4, SD1.5, Wukong) vì phần lớn ảnh thật GenImage có cạnh trong [450, 550]; chọn ảnh thật trong dải kích thước đó (gộp ảnh thật từ mọi tập train của GenImage), cân bằng số ảnh thật/AI trên mỗi lớp ImageNet; giảm tập huấn luyện từ ~320.000 xuống ~75.000 ảnh. Tiền xử lý: **center-crop 450 rồi resize 224** khi huấn luyện; khi suy luận mọi kích thước đều resize về 512 rồi crop 450 rồi resize 224. Khi đánh giá, ảnh PNG được nén JPEG QF 96.
- Hàm mất mát, optimizer, siêu tham số huấn luyện: **Không đề cập trong paper** (chỉ nói dùng code của GenImage và "ResNet50", "Swin-T").

## 4. Architecture
Không đề xuất kiến trúc mới. Detector là ResNet50 (baseline GenImage) và Swin-T, huấn luyện lại trên tập đã ràng buộc (§4.3). Đầu vào 224×224.

## 5. Dataset
- GenImage: 8 generator — Midjourney (1024×1024), SD1.4/SD1.5/Wukong (512×512), GLIDE/ADM/VQDM (256×256), BigGAN (128×128) (§3.2). Ảnh thật: ImageNet, JPEG, nhiều kích thước, đa số QF 96 (§3.1, Phụ lục A Hình 6 — trích từ nguồn blog [27], không phải thực nghiệm của paper).
- Bảng 1 (so sánh các dataset): GenImage — Natural: JPEG, kích thước đa dạng, ImageNet; Synthetic: PNG, 128/256/512/1024, nguồn LAION/ImageNet/không rõ. Ghi nhận cùng kiểu thiên lệch ở Wang et al. (PNG 256×256 vs "mostly JPEG saved as PNG"), DIRE, Epstein, Ricker, Ojha (Bảng 1).
- Split train/val/test của tập đã ràng buộc: dùng val set của GenImage không ràng buộc để đánh giá (§4.2). Số lượng chi tiết val: **Không đề cập trong paper**.

## 6. Experimental Setup
- Giao thức: ma trận cross-generator (huấn luyện trên một tập con GenImage, đánh giá trên các tập con còn lại), chỉ số Accuracy (%). Tái hiện kết quả GenImage (Hình 1) rồi so sánh Classic vs Ours (§3–§4).
- Độ bền JPEG: test nén với các QF khác nhau (Hình 2, Hình 4; Bảng 3 tổng hợp QF 95/80/60).
- Baseline so sánh: ResNet50 và Swin-T huấn luyện trên GenImage thô; GenDet (81,6% khi huấn luyện SD1.4, nêu ở §4.3).
- Phần cứng, learning rate, số epoch: **Không đề cập trong paper**.

## 7. Results
- **Bảng 2** (ResNet50 huấn luyện trên Midjourney; ảnh FFHQ 1024×1024): độ chính xác PNG 80,45; JPEG95 94,84; JPEG90 98,93; JPEG80 99,95; JPEG70 99,99; JPEG60 100,0 (%).
- **Bảng 3** (độ chính xác trung bình trên ma trận cross-generator): JPEG95 — Classic 53,91 vs JPEG96 67,17 (+13,26); JPEG80 — 50,62 vs 59,37 (+8,75); JPEG60 — 50,58 vs 55,07 (+4,49).
- **Bảng 4** (trung bình cross-generator, Classic → Ours): ResNet50 — SD1.5 72,16→83,90 (+11,74); SD1.4 71,27→83,39 (+12,12); Wukong 71,61→80,93 (+9,32); Tổng 71,68→82,74 (+11,06). Swin-T — SD1.5 74,14→85,90 (+11,76); SD1.4 74,93→86,80 (+11,87); Wukong 73,20→84,80 (+11,60); Tổng 74,09→85,83 (+11,74).
- §4.2: tăng tối đa 41,29 điểm %, trung bình 11,06 điểm %; chuyển SD→Wukong giảm ~7 điểm % khi bỏ thiên lệch kích thước (cho thấy đánh giá gốc đã thổi phồng).
- §4.1: nhận xét "bất ngờ" là việc giảm thiên lệch JPEG chỉ cải thiện nhẹ cross-generator, trái với kỳ vọng ban đầu.

## 8. Limitations
- *Paper tự nêu (§5):* mới thử baseline ResNet50/Swin-T, chưa thử các detector chuyên dụng; chưa đánh giá DIRE do hạn chế tính toán; dataset vẫn có thể còn thiên lệch khác — ảnh thật ImageNet vs ảnh LAION dùng huấn luyện SD, và SD gắn fingerprint vào ảnh sinh.
- *Tôi nhận thấy:* (1) cải thiện size-constraint chỉ thực hiện trên nhóm generator 512×512 (SD1.4/1.5/Wukong), không kiểm chứng trên ADM/GLIDE/Midjourney làm tập huấn luyện; (2) bảng 4 gộp JPEG96 và size constraint nên không tách được đóng góp riêng; (3) phân phối QF của ImageNet lấy từ blog bên ngoài; (4) mô hình học từ đầu, không đánh giá backbone foundation đóng băng; (5) các con số chi tiết nằm trong hình nên không kiểm chứng được từng ô.

## 9. Reproducibility
- Công khai: dataset và mã nguồn trên https://www.unbiased-genimage.org (ghi ở Abstract; **chưa kiểm tra được** nội dung website vì chỉ đọc file Markdown).
- Thiếu: siêu tham số huấn luyện, phần cứng. Khả thi chạy lại toàn bộ trên local: thấp (huấn luyện ResNet50/Swin-T trên 75k ảnh); nhưng **áp dụng ý tưởng tiền xử lý** (nén QF thống nhất, kiểm soát kích thước) hoàn toàn khả thi.

## 10. AEGIS Integration
- Module: **Data Layer / chuẩn hóa dữ liệu** (`data.normalize.jpeg_harmonize`, center-crop 224, PNG) và **test `test_no_dataset_shortcut`** (§5.1 PROJECT_BRIEF).
- Tái sử dụng: (i) kịch bản đo "nén ảnh AI làm tăng tỉ lệ bị xếp thành thật" — dùng làm chẩn đoán cho S2; (ii) ý tưởng bổ sung kiểm tra đường tắt về **kích thước/tỉ lệ đối tượng** ngoài 3 đặc trưng chất lượng; (iii) giải thích vì sao Expert FFT nhạy JPEG.
- Đối chiếu chi tiết với pipeline AEGIS: xem `METHOD_SPECS.md` mục "Trả lời câu hỏi a".

## 11. Research Opportunity (đề xuất của tôi, không phải nội dung paper)
- Khai thác thêm phân tích "double compression": ảnh thật của AEGIS bị nén 1 lần (nguồn) + 1 lần (S2), ảnh AI harmonize 1 lần + 1 lần (S2) → thiết kế S2 đối xứng hơn so với baseline của paper.
- Thêm một phép kiểm tra đường tắt thứ hai (nội dung/kích thước đối tượng gốc) vào bộ test shortcut.
- Báo cáo cặp "có/không `jpeg_harmonize`" như một ablation phụ để lượng hóa thiên lệch trên backbone đóng băng (paper chưa làm cho foundation model).
