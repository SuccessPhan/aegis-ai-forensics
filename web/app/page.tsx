import Demo from "./Demo";

const STEPS = [
  { t: "Ba chuyên gia", d: "CLIP (ngữ nghĩa), DINOv2 (cấu trúc chi tiết) và mạng phân tích tần số SRM cùng đánh giá ảnh một cách độc lập." },
  { t: "Hồ sơ ảnh", d: "Hệ thống đo mức nén JPEG, độ sắc nét và năng lượng tần số cao, đồng thời xem ảnh có khác xa dữ liệu đã học không." },
  { t: "Bộ điều phối", d: "Dựa trên hồ sơ đó, bộ điều phối chia trọng số cho từng chuyên gia: tin ai nhiều hơn với ảnh này." },
  { t: "Kết quả và cờ vàng", d: "Xác suất cuối cùng được tính từ ý kiến có trọng số. Nếu kết quả đáng ngờ, hệ thống bật cờ vàng để bạn thận trọng." },
];

export default function Home() {
  return (
    <>
      <Demo>
        <section className="wrap section" aria-labelledby="how-h">
          <h2 id="how-h" className="h2">Cách AEGIS hoạt động</h2>
          <p className="lead small">
            Mục tiêu nghiên cứu: hệ thống không chỉ nhận ra điều đã biết mà còn biết khi nào mình
            không chắc chắn.
          </p>
          <ol className="steps">
            {STEPS.map((s, i) => (
              <li key={s.t} className="step">
                <span className="num" aria-hidden="true">{i + 1}</span>
                <h3>{s.t}</h3>
                <p>{s.d}</p>
              </li>
            ))}
          </ol>
        </section>

        <section className="wrap section" aria-labelledby="note-h">
          <h2 id="note-h" className="h2">Lưu ý</h2>
          <div className="notes">
            <div className="note">
              <h3>Quyền riêng tư</h3>
              <p>
                Ảnh của bạn chỉ được xử lý trong bộ nhớ để trả kết quả và bị xóa ngay sau đó. Hệ
                thống không lưu ảnh và không lưu kết quả.
              </p>
            </div>
            <div className="note">
              <h3>Giới hạn của bản demo</h3>
              <ul>
                <li>Đây là sản phẩm demo của đồ án nghiên cứu, kết quả không phải kết luận chắc chắn.</li>
                <li>Hệ thống được huấn luyện trên ImageNet/GenImage; ảnh chụp điện thoại hoặc đã chỉnh sửa có thể bị đánh giá sai.</li>
                <li>Ảnh AI từ công cụ mới (ví dụ Midjourney) thường bị nhận nhầm là ảnh thật.</li>
                <li>Chỉ hoạt động trong giờ demo.</li>
              </ul>
            </div>
          </div>
        </section>
      </Demo>
      <footer className="foot">
        <div className="wrap foot-in">
          <span>AEGIS – đồ án nghiên cứu phát hiện ảnh do AI tạo ra</span>
          <a href="https://github.com/SuccessPhan/aegis-ai-forensics" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </div>
      </footer>
    </>
  );
}
