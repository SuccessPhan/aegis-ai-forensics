// Build reports/AEGIS_BaoCao_DoAn.docx from numbers.json + figures/.
// Run from repo root:  node reports/report_build/build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  HeadingLevel, ShadingType, BorderStyle, LevelFormat, PageBreak, TableOfContents, Footer, PageNumber,
  PositionalTab, PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json"), "utf8"));
const FIG = (f) => path.join(__dirname, "figures", f);
// Vietnamese decimal comma in all text and tables (charts keep the dot).
const f3 = (x) => (x === null || x === undefined ? "–" : Number(x).toFixed(3).replace(".", ","));
const sgn = (x, d = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d).replace(".", ",");

const FONT = "Times New Roman";
const BODY = 26; // 13 pt
const CONTENT_W = 9638; // A4 width 11906 - 2 * 1134 margins
const ACCENT = "1F4E79", HEAD_FILL = "DCE6F1", MUTED = "595959";

// ---------------------------------------------------------------- helpers
function runs(text, base = {}) {
  // **bold**, _italic_ (only at word boundaries, so names like val_gating_fit stay intact) and `code` markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|(?<![\w])_[^_]+_(?![\w]))/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", ...base, size: (base.size || BODY) - 2 }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 300 },
  alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t),
  spacing: { after: 60, line: 288 }, alignment: AlignmentType.JUSTIFIED });
const NUM = (t, ref = "nums") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t),
  spacing: { after: 60, line: 288 }, alignment: AlignmentType.JUSTIFIED });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, caption, widthPx = 600) {
  const { w, h } = pngSize(file);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: runs(caption, { italics: true, size: 22, color: MUTED }) }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, isHead, align) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: isHead ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align || (isHead ? AlignmentType.CENTER : AlignmentType.LEFT),
      children: runs(String(text), { size: 22, bold: isHead || undefined }) })],
  });
  const align = opts.align || [];
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false, align[i])) })),
    ],
  });
}
const tcap = (t) => new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 },
  children: runs(t, { bold: true, size: 22 }) });
const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });
const C = AlignmentType.CENTER;

// ---------------------------------------------------------------- data shortcuts
const A = N.audit, MV = N.main_v3, OR = N.oracle_v2, FS = N.fewshot_mj_v3, FO = N.forest_v3, RB = N.robustness_v3_clean;
const SP = N.splits_v2, SEL = N.freq_selection_v2.auroc;
const ci = (r) => `${sgn(r.delta)} [${sgn(r.ci_lo)}; ${sgn(r.ci_hi)}]${r.sig ? " *" : ""}`;

// ---------------------------------------------------------------- content
const body = [];

// Title page
body.push(
  new Paragraph({ spacing: { before: 1800, after: 200 }, alignment: C,
    children: [new TextRun({ text: "BÁO CÁO ĐỒ ÁN", bold: true, size: 32, color: ACCENT })] }),
  new Paragraph({ alignment: C, spacing: { after: 200 },
    children: [new TextRun({ text: "AEGIS", bold: true, size: 56, color: ACCENT })] }),
  new Paragraph({ alignment: C, spacing: { after: 600 }, children: [new TextRun({
    text: "Hệ thống phát hiện ảnh do AI tạo ra dựa trên điều phối thích nghi nhiều chuyên gia",
    size: 32, bold: true })] }),
  new Paragraph({ alignment: C, spacing: { after: 120 }, children: [new TextRun({
    text: "Adaptive Multi-Expert AI-Generated Image Forensics", italics: true, size: 26, color: MUTED })] }),
  new Paragraph({ alignment: C, spacing: { before: 1200, after: 120 }, children: runs("Nhóm thực hiện: ……………………………………") }),
  new Paragraph({ alignment: C, spacing: { after: 120 }, children: runs("Giảng viên hướng dẫn: ……………………………………") }),
  new Paragraph({ alignment: C, spacing: { after: 120 }, children: runs("Ngày: 21/11/2026") }),
  pageBreak(),
  new Paragraph({ alignment: C, spacing: { after: 80 }, children: [new TextRun({ text: "MỤC LỤC", bold: true, size: 28 })] }),
  new Paragraph({ alignment: C, spacing: { after: 160 }, children: runs(
    "(Nếu mục lục trống hoặc sai số trang: bấm chuột phải vào mục lục → Update Field.)", { italics: true, size: 20, color: MUTED }) }),
  new TableOfContents("Mục lục", { hyperlink: true, headingStyleRange: "1-2" }),
);

// 1. Tóm tắt (starts on a new page without an extra break paragraph, so a full TOC page leaves no blank page)
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("1. Tóm tắt")] }),
  P("Các bộ phát hiện ảnh do AI tạo ra trong các bài báo thường hoạt động tốt trên ảnh sạch nhưng suy giảm mạnh khi ảnh bị nén JPEG, làm mờ, thu nhỏ, hoặc đến từ một mô hình sinh ảnh (generator) chưa từng thấy. **AEGIS** không cố huấn luyện một mô hình lớn hơn, mà xây một **lớp ra quyết định** đặt trên ba chuyên gia (expert) đóng băng có điểm mù khác nhau: **CLIP ViT-L/14** (ngữ nghĩa), **DINOv2-base** (cấu trúc cục bộ) và một **CNN miền tần số** nhỏ (~1,17 triệu tham số). Một bộ điều phối (gating) đọc hồ sơ chất lượng ảnh và độ bất định của từng expert để chia trọng số động cho từng ảnh; một cơ chế few-shot thích nghi theo số mẫu mới; và mọi cập nhật phải được kiểm chứng trước khi áp dụng."),
  P(`Thực nghiệm trên **16.000 ảnh GenImage** (cân bằng 50/50 Real/AI), qua **ba vòng thực nghiệm chính** (v1, v2, v3) và nhiều vòng kiểm tra dữ liệu, cho thấy: (1) ba expert thật sự hỏng ở các kiểu suy giảm khác nhau; (2) bộ điều phối học đúng hướng — chuyển trọng số sang DINOv2 khi ảnh mờ và giảm trọng số expert tần số khi ảnh bị nén; (3) AEGIS **thắng có ý nghĩa thống kê** ở kịch bản ảnh mờ/thu nhỏ (head sạch: ${ci(FO["clean|S3 vs Static LR"])} AUROC so với Static LR) và ở **thích nghi few-shot trên Midjourney** với N=20 (${ci(FO["clean|S4 MJ N=20 vs Static LR+support"])} so với baseline dùng cùng lượng nhãn); hai kết quả này **lặp lại** khi đổi dữ liệu và cách chia tập. Ngược lại, AEGIS **hòa hoặc thua** Static LR ở ảnh sạch và ảnh nén JPEG. Phân tích “mức trần oracle” cho thấy AEGIS đã cách cận trên của mọi cách trộn có trọng số cố định không quá ~0,01 AUROC, nên giới hạn còn lại nằm ở chất lượng expert chứ không ở bộ điều phối.`),
  P("Báo cáo trình bày trung thực cả kết quả tích cực lẫn tiêu cực, các lỗi đã phát hiện và cách khắc phục qua từng vòng, cùng những giới hạn về tính toàn vẹn của thực nghiệm."),
);

// 2. Bài toán
body.push(H1("2. Bài toán và động lực"),
  P("**Bài toán:** phân loại nhị phân một ảnh tĩnh là **Real** (ảnh thật) hay **AI** (do mô hình sinh ảnh tạo ra). Phạm vi giai đoạn này chỉ gồm ảnh tĩnh; không gồm video, không khoanh vùng chỉnh sửa."),
  P("**Ba khó khăn thực tế** mà hệ thống được thiết kế để giải quyết:"),
  B("**Ảnh bị suy giảm chất lượng:** ảnh lan truyền qua mạng xã hội thường bị nén JPEG, thu nhỏ hoặc làm mờ, xóa đi dấu vết mà nhiều bộ phát hiện dựa vào."),
  B("**Generator mới chưa từng thấy (OOD):** mô hình sinh ảnh mới xuất hiện liên tục; bộ phát hiện huấn luyện trên generator cũ thường kém trên generator mới."),
  B("**Dữ liệu mới rất ít (few-shot):** khi gặp generator mới, thường chỉ thu được vài chục ảnh có nhãn."),
  P("**Triết lý thiết kế:** _“Một hệ thống phát hiện thông minh không chỉ nhận ra điều nó biết, mà còn hiểu điều nó không biết, quyết định khi nào cần thích nghi, và kiểm chứng việc thích nghi có thật sự cải thiện độ tin cậy hay không.”_ Do giới hạn phần cứng (laptop RTX 4060, 8 GB VRAM), dự án chọn hướng **đóng băng backbone lớn, chỉ học các phần nhỏ** thay vì huấn luyện lại hay học liên tục (dễ gây quên thảm họa và quá tải phần cứng)."),
);

// 3. Phương pháp
body.push(H1("3. Phương pháp"),
  H2("3.1. Ba chuyên gia bổ trợ"),
  table(["Expert", "Mô hình", "Huấn luyện", "Kỳ vọng"], [
    ["1 – Ngữ nghĩa", "CLIP ViT-L/14 (768 chiều, kiểu UnivFD)", "Đóng băng; chỉ học head tuyến tính", "Mạnh về logic tổng thể"],
    ["2 – Cấu trúc", "DINOv2 ViT-B/14 (768 chiều)", "Đóng băng; chỉ học head tuyến tính", "Mạnh về bề mặt, chi tiết cục bộ"],
    ["3 – Tần số", "CNN 4 khối, 1.173.473 tham số, đầu vào SRM high-pass", "Ngoại lệ duy nhất được học từ đầu", "Bắt dấu vết tần số cao; dễ hỏng khi nén"],
  ], [1700, 2900, 2600, 2438]),
  spacer(),
  P("Mỗi expert có bước tiền xử lý riêng; expert tần số không bao giờ nhận ảnh đã bị resize ở bước của nó. Embedding của CLIP/DINOv2 được trích một lần và lưu đệm (float16), nhờ đó các bước sau chạy trong vài giây. Mỗi head có hai biến thể: **clean** (chỉ học trên ảnh sạch) và **aug** (học có augmentation blur/JPEG như UnivFD) để so với baseline mạnh nhất."),
  H2("3.2. Hồ sơ ảnh và độ bất định"),
  P("Với mỗi ảnh, hệ thống tính **3 đặc trưng chất lượng** chỉ từ điểm ảnh (không đọc header file): ước lượng mức nén JPEG mù (blockiness), độ sắc nét (log phương sai Laplacian) và tỉ lệ năng lượng tần số cao. Với mỗi expert, tính thêm **entropy dự đoán** (trên xác suất đã hiệu chỉnh bằng temperature scaling) và **khoảng cách cosine** tới tâm cụm ảnh huấn luyện."),
  H2("3.3. Bộ điều phối (Adaptive Gating)"),
  P("Đầu vào 12 chiều = 3 đặc trưng chất lượng + 3 entropy + 3 khoảng cách cosine + 3 xác suất đã hiệu chỉnh. Mô hình là MLP 2 lớp (12 → 32 → 3) có regularization entropy. Trọng số được ràng buộc **w = 0,05 + 0,85·softmax(z)**, nên tổng bằng 1 và mỗi wᵢ ∈ [0,05; 0,90] — không expert nào bị loại hẳn. Kết quả cuối P = Σ wᵢ·Pᵢ."),
  H2("3.4. Thích nghi few-shot theo lượng dữ liệu"),
  table(["Số mẫu mới N", "Hành vi", "Lý do"], [
    ["N < 10", "Giữ nguyên trọng số (α = 0), ghi cảnh báo khan hiếm", "Quá ít mẫu, cập nhật sẽ gây nhiễu"],
    ["10 ≤ N ≤ 50", "Neo: w = (1−α)·w_gating + α·w_few_shot, α = min(0,6; N/50)", "Tin dần dữ liệu mới nhưng không quên kiến thức nền"],
    ["N > 50", "Huấn luyện lại bộ điều phối trên dữ liệu cũ + mẫu mới", "Đủ dữ liệu để học lại"],
  ], [1800, 4600, 3238]),
  spacer(),
  P("Tập hỗ trợ luôn gồm N/2 ảnh AI mới + N/2 ảnh Real. Mọi cập nhật trọng số được đưa qua cơ chế **Champion–Challenger** (chỉ đánh giá trên tập `val_gating_verify`) và tự động **rollback** nếu làm giảm hiệu năng."),
  H2("3.5. Cờ vàng (cảnh báo bất định)"),
  P("Hệ thống luôn trả về nhãn và độ tin cậy, nhưng bật **cờ vàng** khi các expert mâu thuẫn gay gắt (có Pᵢ > 0,8 và Pⱼ < 0,2 với độ lệch chuẩn > 0,3), khi P cuối nằm trong vùng lưỡng lự [0,4; 0,6], hoặc khi điểm OOD vượt phân vị 95 của tập validation."),
);

// 4. Dữ liệu
const aud = (k, g) => f3(A[k][g]);
body.push(H1("4. Dữ liệu và kiểm soát đường tắt"),
  H2("4.1. Nguồn dữ liệu"),
  P("Dữ liệu lấy từ bộ **GenImage** (NeurIPS 2023, arXiv 2306.08571), qua file `genimage_test.zip` trên bản mirror Hugging Face `jzousz/GenImage` (bản chính thức phát hành qua Baidu Yunpan). Với mỗi generator, nhóm lấy ngẫu nhiên có seed 2.000 ảnh AI + 2.000 ảnh Real (ImageNet), rồi **tự chia lại** thành các tập riêng của AEGIS — không dùng cách chia gốc của GenImage. Giấy phép CC BY-NC-SA 4.0 (chỉ dùng phi thương mại)."),
  table(["Vai trò", "Generator", "Ghi chú"], [
    ["Seen (huấn luyện + kiểm thử S1–S3)", "ADM, Stable Diffusion v1.4", "Hai họ diffusion khác nhau"],
    ["Unseen (chỉ dùng cho S4)", "Midjourney, VQDM", "GLIDE bị thay bằng VQDM sau vòng v1 (xem mục 6)"],
    ["Loại bỏ", "BigGAN", "Ảnh 128×128, không chuẩn hóa được về 224"],
  ], [3300, 3000, 3338]),
  H2("4.2. Phát hiện quan trọng: dataset “nói dối”"),
  P("Danh mục file xác nhận **100% ảnh AI là PNG, 100% ảnh Real là JPEG**, và mỗi generator có một kích thước cố định (ADM 256², SD 512², Midjourney 1024²), trong khi ảnh Real có kích thước đa dạng. Một mô hình có thể đoán đúng nhãn chỉ nhờ định dạng/kích thước mà không cần học dấu vết AI. Để đo điều này, nhóm xây **bài audit đường tắt**: một Logistic Regression chỉ dùng 3 đặc trưng chất lượng ảnh. Nếu không có đường tắt, AUROC của nó phải ≈ 0,50."),
  ...figure(FIG("fig1_audit.png"), "Hình 1. AUROC của bài audit (chỉ từ 3 đặc trưng chất lượng) theo từng generator qua các lần xử lý dữ liệu."),
  tcap("Bảng 1. Kết quả audit đường tắt (AUROC, càng gần 0,50 càng tốt)"),
  table(["Cách xử lý", "Gộp", "ADM", "SD v1.4", "Midjourney", "GLIDE", "VQDM"], [
    ["Chỉ crop 224", aud("crop", "pooled"), aud("crop", "adm"), aud("crop", "sd_v1_4"), "–", "–", "–"],
    ["resize_crop", aud("resize_crop", "pooled"), aud("resize_crop", "adm"), aud("resize_crop", "sd_v1_4"), "–", "–", "–"],
    ["v1: resize_crop + cân bằng", aud("v1", "pooled"), aud("v1", "adm"), aud("v1", "sd_v1_4"), aud("v1", "midjourney"), aud("v1", "glide"), "–"],
    ["v3: dữ liệu có VQDM", aud("v2", "pooled"), aud("v2", "adm"), aud("v2", "sd_v1_4"), aud("v2", "midjourney"), "–", aud("v2", "vqdm")],
  ], [2738, 1150, 1150, 1150, 1150, 1150, 1150], { align: [undefined, C, C, C, C, C, C] }),
  spacer(),
  P("**Các biện pháp và hiệu quả:**"),
  B("**Hòa đồng lịch sử JPEG** (`jpeg_harmonize`): nén ảnh AI một lần với mức chất lượng lấy theo phân phối của ảnh Real. Sau bước này, riêng đặc trưng JPEG chỉ còn AUROC 0,54–0,62 — dấu vết định dạng gần như đã bị xóa."),
  B("**resize_crop** (thu cạnh ngắn về 256 rồi crop 224, theo hướng của bài “Fake or JPEG?”, arXiv 2403.17608) thay cho crop trực tiếp: giảm AUROC gộp từ " + aud("crop", "pooled") + " xuống " + aud("resize_crop", "pooled") + "."),
  B("**Phần còn lại không xóa được** nằm ở _lịch sử thu nhỏ_: ảnh Real (~500 px) và SD (512 px) cùng bị thu nhỏ ~2 lần nên SD gần sạch, còn ADM (256 px gốc) không bị thu nhỏ nên mềm hơn ảnh Real. Một phần chênh lệch có thể là khác biệt thật giữa ảnh chụp và ảnh AI."),
  B("**Quyết định:** dùng resize_crop; báo cáo audit theo từng generator thay vì dùng làm cổng chặn cứng; thêm baseline **quality_only** (chính LR của audit) vào mọi bảng kết quả như **mức sàn** mà mọi phương pháp phải vượt xa."),
  B("**GLIDE bị loại khỏi vai trò OOD** vì chỉ đo độ sắc nét đã phân biệt được 97,9% — bài kiểm tra quá dễ, không chứng minh được khả năng thích nghi."),
);

// 5. Thiết lập
body.push(H1("5. Thiết lập thực nghiệm"),
  H2("5.1. Chia tập không rò rỉ"),
  P("Ảnh được khử trùng lặp bằng SHA-256 rồi chia **theo nhóm**: mọi biến thể suy giảm của một ảnh gốc luôn nằm cùng một tập. Mọi tập được cân bằng 50/50 Real/AI. Tập `test_benchmark` chỉ dùng cho bảng kết quả cuối; mọi lựa chọn mô hình chỉ dựa trên `val_gating_verify`."),
  tcap("Bảng 2. Kích thước các tập (ảnh sạch, dữ liệu v3)"),
  table(["Tập", "Vai trò", "Real", "AI"], [
    ["train_experts", "Huấn luyện head của expert", SP.train_experts.real, SP.train_experts.ai],
    ["val_gating_fit", "Hiệu chỉnh xác suất + huấn luyện bộ điều phối", SP.val_gating_fit.real, SP.val_gating_fit.ai],
    ["val_gating_verify", "Champion–Challenger, chọn mô hình", SP.val_gating_verify.real, SP.val_gating_verify.ai],
    ["test_benchmark", "Đánh giá cuối (S1–S3), chỉ chạy một lần", SP.test_benchmark.real, SP.test_benchmark.ai],
    ["s4_support", "Mẫu few-shot của generator unseen", SP.s4_support.real, SP.s4_support.ai],
    ["s4_eval", "Đánh giá S4 (tách khỏi s4_support)", SP.s4_eval.real, SP.s4_eval.ai],
  ], [2200, 4838, 1300, 1300], { align: [undefined, undefined, C, C] }),
  spacer(),
  P("Tập validation chứa biến thể suy giảm với tham số **ngẫu nhiên liên tục** (JPEG Q ∈ [30; 95], blur σ ∈ [0,5; 3,0]), còn tập test dùng các **mức cố định** dưới đây — để bộ điều phối không “học trước đề thi”. Tổng số bản ghi (gồm biến thể) là 32.170."),
  tcap("Bảng 3. Bốn kịch bản đánh giá"),
  table(["Kịch bản", "Nội dung"], [
    ["S1 – Sạch", "Ảnh gốc của ADM, SD v1.4 và Real"],
    ["S2 – Nén JPEG", "Q = 85, 65, 40"],
    ["S3 – Mờ / thu nhỏ", "Gaussian blur (k=11, σ=1,5) và (k=17, σ=2,5); thu nhỏ 0,5× rồi phóng lại"],
    ["S4 – Generator mới + few-shot", "Midjourney và VQDM, N ∈ {5, 20, 100}, 5 seed"],
  ], [2800, 6838]),
  H2("5.2. Các phương pháp so sánh"),
  B("**Từng expert đơn lẻ:** CLIP, DINOv2, Freq (SRM)."),
  B("**Naive average:** trung bình cộng 3 xác suất."),
  B("**Static LR:** bộ trọng số cố định học bằng logistic regression trên `val_gating_fit` — baseline mạnh nhất vì cũng được thấy ảnh suy giảm."),
  B("**Static LR + support** (chỉ S4): học lại trên cùng tập hỗ trợ mà AEGIS dùng — để so sánh với **cùng lượng dữ liệu có nhãn**."),
  B("**quality_only:** mức sàn đường tắt (mục 4.2)."),
  B("**AEGIS** và **AEGIS không thích nghi** (để tách riêng lợi ích của cơ chế few-shot)."),
  H2("5.3. Độ đo và kiểm định"),
  P("Accuracy, Precision, Recall, F1, **AUROC** (độ đo chính), ECE (độ hiệu chỉnh), **Selective Accuracy** (độ chính xác trên ảnh không bị gắn cờ) và **Flagged Rate**. Mọi chênh lệch được kiểm định bằng **paired bootstrap 2.000 lần**, lấy mẫu theo cụm ảnh gốc (ảnh gốc và các biến thể của nó được lấy cùng nhau) để khoảng tin cậy không bị hẹp giả tạo. Dấu * trong các bảng nghĩa là khoảng tin cậy 95% không chứa 0."),
);

// 6. Hành trình thực nghiệm
body.push(H1("6. Hành trình thực nghiệm: phát hiện và khắc phục"),
  P("Mục này ghi lại các vòng thực nghiệm theo thứ tự thời gian: mỗi vòng phát hiện gì, nhóm khắc phục thế nào, và kết quả lưu ở đâu. Toàn bộ dữ liệu và kết quả của các vòng đều được giữ lại để có thể kiểm tra lại."),
  tcap("Bảng 4. Nhật ký các vòng thực nghiệm"),
  table(["Vòng", "Phát hiện", "Hướng khắc phục / quyết định", "Lưu tại"], [
    ["0. Hạ tầng", "Xây pipeline, hợp đồng dữ liệu và 61 bài test trên dữ liệu giả", "Quy tắc: mỗi module test < 15 giây; không rò rỉ giữa các tập", "commit de7e198"],
    ["0b. Rà soát 8 bài báo", "Center-crop trực tiếp làm lệch “độ zoom”; UnivFD huấn luyện có augmentation; expert tần số nên dùng high-pass", "So sánh 2 cách chuẩn hóa bằng audit; báo cáo cả head clean và aug; chọn FFT/SRM trên val", "docs/research/"],
    ["1. Audit dữ liệu thật", `Lộ nhãn qua định dạng/kích thước: AUROC ${aud("crop", "pooled")}`, `jpeg_harmonize + resize_crop → ${aud("resize_crop", "pooled")}; thêm mức sàn quality_only`, "data/normalized_crop, normalized_resize_crop"],
    ["1b. Cân bằng", "Real nhiều gấp ~1,8 lần AI trong các tập", "Phân bổ Real theo số AI từng tập → 50/50", "data/normalized_v1"],
    ["2. Lỗi ẩn SRM", `Expert SRM chỉ đạt AUROC ${f3(N.srm_bug.before_train_auroc)} trên chính tập train dù loss = ${N.srm_bug.train_loss_final}`, `Thống kê BatchNorm bị lệch → ước lượng lại sau huấn luyện → ${f3(N.srm_bug.after_val_auroc_300)} trên val; thêm test hồi quy`, "commit a580571"],
    ["2b. Lỗi quản lý mã", "Mẫu .gitignore “data/” vô tình bỏ qua cả src/data", "Neo mẫu về thư mục gốc", "commit f017ab6"],
    ["v1. Kết quả thật đầu tiên", "AEGIS thắng ở S3 và S4 Midjourney; hòa ở S1/S2. GLIDE quá dễ (audit 0,979)", "Quyết định thay GLIDE bằng VQDM", "reports/v1"],
    ["v2. Kiểm định & chẩn đoán", "Bootstrap CI; biểu đồ trọng số; mức trần oracle: AEGIS cách cận trên ≤ 0,011", "Bác bỏ 2 giả thuyết của chính nhóm (mục 8.3); không cải tiến bộ đo JPEG vì hết dư địa", "reports/v2 (khớp v1 từng byte)"],
    ["Cải tiến vòng 1", "Thử trộn theo logit thay vì trộn xác suất", "Chọn chỉ trên val: logit không tốt hơn → giữ trộn xác suất", "reports/fusion_selection_v3"],
    ["v3. Kết quả chính", "Hai kết quả tích cực lặp lại; S2 head sạch thua có ý nghĩa; VQDM bão hòa", "Báo cáo trung thực; xác định hướng tiếp theo (mục 11)", "reports/v3, data/normalized_v2"],
  ], [1500, 3000, 3338, 1800]),
);

// 7. Kết quả chính
const mrow = (hv, key) => { const v = MV[`${hv}|${key}`]; return [key, f3(v.clip), f3(v.dinov2), f3(v.freq), f3(v.static_lr), f3(v.aegis), f3(v.quality_only)]; };
const MKEYS = ["S1 sạch", "S2 JPEG", "S3 mờ/resize", "S4 Midjourney N=20"];
body.push(H1("7. Kết quả chính (v3)"),
  P("Kết quả trong mục này là của vòng v3: dữ liệu có VQDM, cách chia tập mới, tập test chạy đúng một lần. Trước khi chạy test, các lựa chọn được quyết định **chỉ trên tập validation**: expert tần số dùng SRM (AUROC val " + f3(SEL.clean.srm) + " so với FFT " + f3(SEL.clean.log_magnitude_fft) + "), trộn theo xác suất (trộn theo logit không cải thiện AEGIS trên val)."),
  H2("7.1. Các expert hỏng ở những chỗ khác nhau"),
  ...figure(FIG("fig2_robustness.png"), "Hình 2. AUROC của từng expert và AEGIS theo mức suy giảm (test v3, head sạch)."),
  P(`Expert tần số đạt ${f3(RB["Nén JPEG"].freq[0])} trên ảnh sạch nhưng rơi xuống ${f3(RB["Nén JPEG"].freq[3])} khi nén JPEG Q40 và ${f3(RB["Làm mờ Gaussian"].freq[1])} khi làm mờ — gần mức đoán ngẫu nhiên. CLIP sụp mạnh khi ảnh mờ (${f3(RB["Làm mờ Gaussian"].clip[0])} → ${f3(RB["Làm mờ Gaussian"].clip[2])}), trong khi DINOv2 bền hơn với ảnh mờ (${f3(RB["Làm mờ Gaussian"].dinov2[2])} ở σ=2,5). Đây là bằng chứng trực tiếp cho giả thuyết nền của dự án: ba expert có điểm mù khác nhau, nên việc chia trọng số theo tình trạng ảnh là có cơ sở.`),
  H2("7.2. Bảng kết quả"),
  ...figure(FIG("fig3_main_v3.png"), "Hình 3. AUROC của các phương pháp theo kịch bản (v3). Vạch xám là mức sàn quality_only."),
  tcap("Bảng 5. AUROC theo kịch bản — head sạch (ở S4, cột Static LR là bản học lại trên cùng support set)"),
  table(["Kịch bản", "CLIP", "DINOv2", "Freq", "Static LR", "AEGIS", "Sàn"], MKEYS.map((k) => mrow("clean", k)),
    [2738, 1150, 1150, 1150, 1150, 1150, 1150], { align: [undefined, C, C, C, C, C, C] }),
  tcap("Bảng 6. AUROC theo kịch bản — head augmentation"),
  table(["Kịch bản", "CLIP", "DINOv2", "Freq", "Static LR", "AEGIS", "Sàn"], MKEYS.map((k) => mrow("aug", k)),
    [2738, 1150, 1150, 1150, 1150, 1150, 1150], { align: [undefined, C, C, C, C, C, C] }),
  H2("7.3. Ý nghĩa thống kê"),
  ...figure(FIG("fig4_forest_v3.png"), "Hình 4. Chênh lệch AUROC giữa AEGIS và baseline mạnh nhất, kèm khoảng tin cậy 95%."),
  tcap("Bảng 7. ΔAUROC = AEGIS − baseline [CI 95%] (* = có ý nghĩa thống kê)"),
  table(["So sánh", "Head sạch", "Head augmentation"], [
    "S1 vs Static LR", "S2 vs Static LR", "S3 vs Static LR", "S4 MJ N=20 vs Static LR+support",
    "S4 MJ N=100 vs Static LR+support", "S4 VQDM N=20 vs Static LR+support",
  ].map((k) => [k, ci(FO[`clean|${k}`]), ci(FO[`aug|${k}`])]), [3638, 3000, 3000], { align: [undefined, C, C] }),
  spacer(),
  H2("7.4. Thích nghi few-shot"),
  ...figure(FIG("fig5_fewshot_v3.png"), "Hình 5. AUROC trên Midjourney (chưa từng thấy) theo số mẫu few-shot N, trung bình 5 seed."),
  P(`Với N=5, AEGIS cố ý **không thay đổi** (đúng thiết kế khi dữ liệu khan hiếm), nên trùng với bản không thích nghi. Với N=20 (chế độ neo), AEGIS tăng từ ${f3(FS.clean.aegis_noadapt)} lên ${f3(FS.clean.aegis[1])} (head sạch) và từ ${f3(FS.aug.aegis_noadapt)} lên ${f3(FS.aug.aegis[1])} (head augmentation), trong khi baseline học lại trên **cùng** tập hỗ trợ gần như đứng yên (${f3(FS.clean.static_learned_support[1])} và ${f3(FS.aug.static_learned_support[1])}). Như vậy lợi thế đến từ **cơ chế thích nghi**, không chỉ từ việc có thêm mẫu có nhãn. Với head sạch, AEGIS chỉ nhỉnh hơn CLIP đơn lẻ (${f3(FS.clean.clip)}) ở N=20 và thua nhẹ ở N=100 — CLIP là đối thủ rất mạnh trên Midjourney.`),
  P("Trên **VQDM**, mọi phương pháp đều ≈ 0,998: VQDM không lộ nhãn qua chất lượng ảnh (audit 0,727) nhưng lại quá dễ với expert ngữ nghĩa, nên không phân biệt được các phương pháp. Bằng chứng S4 vì vậy dựa chủ yếu vào Midjourney."),
);

// 8. Phân tích sâu
const oro = (hv, k) => { const v = OR[`${hv}|${k}`]; return [k, f3(v.static_lr), f3(v.aegis), f3(v.oracle), sgn(v.aegis - v.oracle)]; };
const OKEYS = ["sạch", "Q85", "Q65", "Q40", "σ1.5", "σ2.5", "resize"];
body.push(H1("8. Phân tích sâu"),
  H2("8.1. Bộ điều phối có chia trọng số đúng hướng không?"),
  P("Trên `val_gating_verify` (tập bộ điều phối không được học), nhóm vẽ trọng số trung bình của từng expert theo mức JPEG và mức mờ, đặt cạnh AUROC của từng expert trên cùng nhóm ảnh."),
  ...figure(path.join(ROOT, "reports/v3/weights_by_blur.png"), "Hình 6. Trọng số và AUROC của từng expert theo mức mờ (v3)."),
  ...figure(path.join(ROOT, "reports/v3/weights_by_jpeg.png"), "Hình 7. Trọng số và AUROC của từng expert theo mức nén JPEG (v3)."),
  B("**Khi ảnh mờ (head sạch):** trọng số chuyển từ CLIP sang DINOv2 (trên tập test: w_DINOv2 tăng từ 0,15 ở ảnh sạch lên 0,67 ở σ=1,5) — đúng với việc CLIP sụp và DINOv2 bền khi mờ. AEGIS nhờ đó nằm trên cả hai đường."),
  B("**Khi ảnh bị nén (head augmentation):** trọng số expert tần số giảm đều theo mức nén (0,39 → 0,28 → 0,20 ở Q85 → Q40), song song với AUROC của nó giảm."),
  B("**Khi ảnh mờ, expert tần số bị hạ về sàn** (~0,07) vì nó gần như đoán ngẫu nhiên."),
  B("Xu hướng này **lặp lại** ở cả v1, v2 và v3."),
  H2("8.2. Mức trần: bộ điều phối còn bao nhiêu dư địa?"),
  P("Câu hỏi: khi AEGIS hòa Static LR, là do bộ điều phối kém, hay do không còn gì để thắng? Nhóm tính một **oracle**: với mỗi mức suy giảm, tìm bộ trọng số cố định **tốt nhất có thể** (dò lưới bước 0,05 trên chính lát test). Đây là cận trên lạc quan, **chỉ dùng để chẩn đoán**, không phải hiệu năng thật."),
  ...figure(FIG("fig7_oracle.png"), "Hình 8. AEGIS, Static LR và cận trên oracle theo từng mức suy giảm (dự đoán test v2)."),
  tcap("Bảng 8. Dư địa so với oracle — head sạch (AUROC, test v2)"),
  table(["Mức", "Static LR", "AEGIS", "Oracle", "AEGIS − Oracle"], OKEYS.map((k) => oro("clean", k)),
    [2438, 1800, 1800, 1800, 1800], { align: [undefined, C, C, C, C] }),
  spacer(),
  P("AEGIS cách cận trên không quá ~0,01 AUROC ở **mọi** mức suy giảm (head augmentation: 0,001–0,011), dù nó **không được biết** ảnh bị suy giảm kiểu gì. Kết luận: trong họ phương pháp “trung bình có trọng số”, bộ điều phối đã gần tối ưu; ở ảnh nén JPEG, trọng số tốt nhất gần như không đổi giữa các mức nên trọng số cố định cũng đủ tốt. **Giới hạn còn lại nằm ở chất lượng của các expert**, không nằm ở cách điều phối. Chính phân tích này đã giúp nhóm **không** đầu tư vào một cải tiến không có dư địa."),
  H2("8.3. Những giả thuyết của nhóm đã bị bác bỏ"),
  NUM("**“Bộ đo chất lượng nhầm ảnh nén với ảnh mờ.”** Từ biểu đồ trọng số, nhóm nghi bộ điều phối chuyển nhầm sang DINOv2 khi ảnh bị nén. Kiểm tra cho thấy 3 đặc trưng chất lượng phân biệt nén với mờ rất tốt (AUROC 0,94–0,97; dưới 1% ảnh JPEG bị nhận nhầm là ảnh mờ). Giả thuyết **sai**."),
  NUM("**“Trộn DINOv2 khi ảnh bị nén là sai lầm.”** Trên test, ở JPEG Q40 (head sạch), AEGIS đạt 0,869 cao hơn CLIP một mình (0,840): hai expert bổ trợ nhau kể cả khi DINOv2 yếu hơn. Điểm AEGIS thấp hơn CLIP ở một nhóm Q trên biểu đồ chỉ là nhiễu do mẫu nhỏ (152 ảnh). Giả thuyết **sai**."),
  NUM("**“Trộn theo logit sẽ tốt hơn trộn xác suất.”** Kiểm tra chỉ trên val: ΔAUROC −0,0001 (head sạch) và −0,0012 (head augmentation), khoảng tin cậy đều chứa 0. Giả thuyết **không được ủng hộ**; giữ trộn xác suất."),
  H2("8.4. Kết quả có lặp lại không?"),
  ...figure(FIG("fig6_replication.png"), "Hình 9. Chênh lệch AEGIS − Static LR ở v1 (GLIDE, cách chia cũ) và v3 (VQDM, cách chia mới)."),
  P("Hai kết quả tích cực **lặp lại** trên dữ liệu và cách chia khác: S3 với head sạch (giảm từ +0,023 xuống +0,013 nhưng vẫn có ý nghĩa) và few-shot Midjourney N=20 (+0,013 đến +0,016 ở cả hai lần, cả hai head). Ngược lại, kết quả **không ổn định** ở S2 với head sạch: v1 hòa, v3 thua có ý nghĩa (−0,009) — một điểm nhóm chưa giải thích được đầy đủ."),
  H2("8.5. Cờ vàng có giúp ích không?"),
  P("Trên ảnh không bị gắn cờ, độ chính xác luôn cao hơn độ chính xác chung (ví dụ head sạch: S3 từ 0,728 lên 0,780 với 30% ảnh bị gắn cờ; S2 từ 0,739 lên 0,791 với 25,6% ảnh bị gắn cờ). Tỉ lệ gắn cờ tăng theo mức suy giảm (7,4% ở S1, 25,6% ở S2, 30,0% ở S3) — hệ thống “biết” khi nào nó không chắc chắn."),
);

// 9. Nhóm hiểu gì
body.push(H1("9. Nhóm đã làm được gì và hiểu được gì"),
  H2("9.1. Những gì đã làm được"),
  B("Một pipeline hoàn chỉnh, tái lập được: chạy lại cho kết quả trùng **từng byte**; 113 bài test tự động; mọi lần chạy gắn với một commit git."),
  B("Một quy trình kiểm soát dữ liệu: khử trùng lặp, chia tập theo nhóm, cân bằng, audit đường tắt có đối chứng âm/dương."),
  B("Bộ điều phối và cơ chế few-shot có **lợi thế có ý nghĩa thống kê và lặp lại được** ở ảnh mờ và ở generator mới (Midjourney)."),
  B("Các công cụ phân tích: kiểm định bootstrap theo cụm, biểu đồ trọng số, mức trần oracle."),
  H2("9.2. Những điều nhóm hiểu ra"),
  NUM("**Dataset có thể “nói dối”.** GenImage lộ nhãn qua định dạng và kích thước. Phải đo và báo cáo điều này, không được che đi.", "nums2"),
  NUM("**Điểm mù bổ trợ là có thật và đo được** — đây là điều kiện cần để việc điều phối có ý nghĩa.", "nums2"),
  NUM("**Điều phối chỉ thắng khi trọng số tối ưu thay đổi theo tình trạng ảnh.** Ở ảnh mờ, trọng số tốt nhất thay đổi nhiều (AEGIS thắng); ở ảnh nén, nó gần như cố định (trọng số tĩnh đã đủ).", "nums2"),
  NUM("**Biết mức trần trước khi tối ưu** giúp tránh lãng phí công sức vào hướng không còn dư địa.", "nums2"),
  NUM("**Few-shot chỉ có ích khi đủ mẫu** và phải được so với baseline dùng cùng lượng nhãn mới công bằng.", "nums2"),
  NUM("**Chọn bộ kiểm tra OOD khó hơn tưởng:** GLIDE quá dễ vì chất lượng ảnh, VQDM quá dễ vì ngữ nghĩa; chỉ Midjourney thật sự phân biệt được các phương pháp.", "nums2"),
  NUM("**Kiểm tra giả thuyết của chính mình:** hai giả thuyết của nhóm đã bị dữ liệu bác bỏ, và điều đó giúp hiểu đúng hơn về hệ thống.", "nums2"),
);

// 10. Hạn chế
body.push(H1("10. Hạn chế và tính toàn vẹn của thực nghiệm"),
  B("**Đã xem kết quả test trước khi cải tiến.** Các quyết định ở vòng cải tiến (thay GLIDE, thử trộn theo logit) được đưa ra sau khi xem kết quả test v1/v2. Vòng v3 dùng cách chia tập mới và chỉ chạy test một lần, nhưng không thể coi là hoàn toàn “mù”."),
  B("**Đường tắt còn sót:** audit trên ADM vẫn ở 0,754 (do lịch sử thu nhỏ ảnh). Mức sàn quality_only được đưa vào mọi bảng để người đọc thấy rõ."),
  B("**AEGIS không vượt trội ở mọi nơi:** hòa ở ảnh sạch, hòa hoặc thua ở ảnh nén JPEG; CLIP đơn lẻ rất mạnh trên Midjourney."),
  B("**Champion–Challenger và rollback** mới được kiểm chứng bằng test tự động trên dữ liệu giả, chưa có thí nghiệm trình diễn trên dữ liệu thật."),
  B("**Quy mô:** 2.000 ảnh mỗi lớp mỗi generator, 4 generator; chưa thử các generator mới hơn hoặc ảnh thật ngoài ImageNet."),
  B("**Số liệu trích từ bài báo** được chuyển từ PDF sang văn bản; một số bảng bị vỡ định dạng và cần đối chiếu với PDF gốc trước khi trích dẫn."),
);

// 11. Hướng phát triển
body.push(H1("11. Hướng phát triển"),
  NUM("**Giải thích kết quả S2 (head sạch)** chỉ bằng tập validation, không chạy lại test.", "nums3"),
  NUM("**Trình diễn Champion–Challenger và rollback trên dữ liệu thật**, ví dụ coi mỗi lần thích nghi few-shot là một challenger.", "nums3"),
  NUM("**Nâng chất lượng expert** (head MLP thay vì tuyến tính, đặc trưng nhiều vùng crop) — hướng duy nhất còn dư địa theo phân tích mức trần.", "nums3"),
  NUM("**Generator unseen khó hơn** để S4 phân biệt được các phương pháp tốt hơn.", "nums3"),
  NUM("**Bản demo web** cho người dùng gửi ảnh để kiểm tra, với cảnh báo rõ ràng rằng ảnh ngoài thực tế khác xa dữ liệu huấn luyện.", "nums3"),
);

// Phụ lục
body.push(H1("Phụ lục A. Tái lập kết quả"),
  table(["Thành phần", "Vị trí"], [
    ["Mã nguồn và lịch sử", "Kho git của dự án (commit kết quả chính v3: be13961)"],
    ["Cấu hình", "configs/default.yaml (manifest v2, cache v2, Freq = SRM, fusion = prob)"],
    ["Dữ liệu chuẩn hóa", "data/normalized_v1 (v1/v2), data/normalized_v2 (v3)"],
    ["Embedding & mô hình", "data/cache/features, data/cache/heads, data/cache/bundles_v1, data/cache/bundles"],
    ["Kết quả", "reports/v1, reports/v2, reports/v3, reports/fusion_selection_v3"],
    ["Biểu đồ báo cáo", "reports/report_build/make_figures.py (đọc trực tiếp từ các file kết quả)"],
  ], [3000, 6638]),
  spacer(),
  P("Chuỗi lệnh chính (chạy từ thư mục gốc, môi trường `.venv`):"),
  B("`python -m src.data.ingest --mode resize_crop --out data/normalized_v2`"),
  B("`python -m src.models.extract --manifest <manifest> --expert clip|dinov2`"),
  B("`python -m src.models.heads --manifest <manifest> --expert clip|dinov2|freq --variant clean|aug`"),
  B("`python -m src.models.bundle --manifest <manifest> --split <tập> --head-variant clean|aug`"),
  B("`python -m src.evaluation.select_fusion` (chỉ đọc validation)"),
  B("`python -m src.evaluation.ablation --s4-seeds 0 1 2 3 4 --out reports/v3 --save-predictions`"),
  B("`python -m src.evaluation.significance` và `python -m src.evaluation.weights_analysis`"),
  H1("Phụ lục B. Tài liệu tham khảo"),
  P("Danh sách dưới đây ghi tên bài và mã arXiv đã được tra cứu trực tiếp trên arXiv."),
  NUM("GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image. arXiv:2306.08571 (NeurIPS 2023).", "refs"),
  NUM("Towards Universal Fake Image Detectors that Generalize Across Generative Models (UnivFD). arXiv:2302.10174.", "refs"),
  NUM("DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193.", "refs"),
  NUM("Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues (F3Net). arXiv:2007.09355.", "refs"),
  NUM("Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning (FreqNet). arXiv:2403.07240.", "refs"),
  NUM("On Calibration of Modern Neural Networks. arXiv:1706.04599.", "refs"),
  NUM("Energy-based Out-of-distribution Detection. arXiv:2010.03759.", "refs"),
  NUM("Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. arXiv:2403.17608.", "refs"),
);

// ---------------------------------------------------------------- document
const numCfg = (ref, fmt, text) => ({ reference: ref, levels: [{ level: 0, format: fmt, text, alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] });
const doc = new Document({
  creator: "AEGIS", title: "Báo cáo đồ án AEGIS", description: "Báo cáo đồ án AEGIS",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24 },
        paragraph: { spacing: { before: 0, after: 20, line: 264 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24 },
        paragraph: { spacing: { before: 0, after: 20, line: 264 }, indent: { left: 280 } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
    numCfg("nums", LevelFormat.DECIMAL, "%1."), numCfg("nums2", LevelFormat.DECIMAL, "%1."),
    numCfg("nums3", LevelFormat.DECIMAL, "%1."), numCfg("refs", LevelFormat.DECIMAL, "[%1]"),
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Trang ", PageNumber.CURRENT], size: 20, color: MUTED })] })] }) },
    children: body,
  }],
});

const out = path.join(ROOT, "reports", "AEGIS_BaoCao_DoAn.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length, "bytes"); });
