import Demo from "./Demo";

export default function Home() {
  return (
    <>
      <main className="wrap">
        <header>
          <h1>AEGIS – Kiểm tra ảnh do AI tạo ra</h1>
          <p className="badge">Bản demo nghiên cứu</p>
          <p>
            Tải một ảnh lên để hệ thống ước lượng khả năng ảnh đó do AI tạo ra. Đây là sản phẩm
            demo của đồ án nghiên cứu, <strong>kết quả không phải kết luận chắc chắn</strong>. Hệ
            thống được huấn luyện trên ảnh ImageNet/GenImage, nên ảnh ngoài thực tế (ảnh chụp bằng
            điện thoại, ảnh đã chỉnh sửa, ảnh từ các công cụ tạo ảnh mới) có thể bị đánh giá sai.
            Đặc biệt, ảnh AI từ các công cụ mới (ví dụ Midjourney) thường bị hệ thống{" "}
            <strong>nhận nhầm là ảnh thật</strong>.
          </p>
          <p className="privacy" role="note">
            <strong>Quyền riêng tư:</strong> ảnh của bạn chỉ được xử lý trong bộ nhớ để trả kết quả
            và bị xóa ngay sau đó. Hệ thống không lưu ảnh và không lưu kết quả.
          </p>
        </header>
        <Demo />
        <section id="about" aria-labelledby="about-h">
          <h2 id="about-h">Về dự án</h2>
          <p>
            AEGIS là hệ thống phát hiện ảnh do AI tạo ra theo hướng đa chuyên gia thích nghi. Ba
            chuyên gia độc lập — CLIP (ngữ nghĩa), DINOv2 (cấu trúc chi tiết) và một mạng phân tích
            tần số (SRM) — cùng đánh giá ảnh. Một bộ điều phối sẽ tự điều chỉnh trọng số của từng
            chuyên gia dựa trên chất lượng ảnh và mức độ chắc chắn, đồng thời bật cờ vàng khi kết
            quả đáng ngờ. Mục tiêu nghiên cứu là hệ thống không chỉ nhận ra điều đã biết mà còn
            biết khi nào mình không chắc chắn.
          </p>
        </section>
      </main>
      <footer className="wrap foot">
        <span>AEGIS – đồ án nghiên cứu phát hiện ảnh AI</span>
        <a href="#about">Về dự án</a>
      </footer>
    </>
  );
}
