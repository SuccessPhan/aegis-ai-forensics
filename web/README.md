# AEGIS Web Demo (frontend)

Giao diện web (Next.js, App Router, TypeScript) cho bản demo AEGIS. Trình duyệt gọi **thẳng** máy chủ FastAPI (xem `docs/WEB_API_CONTRACT.md`), không đi qua hàm serverless của Vercel (tránh giới hạn ~4,5 MB). Web không lưu ảnh hay kết quả.

## Chạy local
```bash
cd web
cp .env.example .env.local   # NEXT_PUBLIC_AEGIS_API_URL=http://localhost:8000
npm install
npm run dev                  # http://localhost:3000
```
Backend cần cho phép origin `http://localhost:3000` trong `AEGIS_ALLOWED_ORIGINS`. Nếu biến `NEXT_PUBLIC_AEGIS_API_URL` trống, trang chỉ hiện thông báo thiếu cấu hình (không giả kết quả).

Kiểm tra: `npm run lint`, `npm run build`.

## Deploy lên Vercel
1. Import repo, đặt **Root Directory = `web`**.
2. Vào Settings > Environment Variables, thêm `NEXT_PUBLIC_AEGIS_API_URL` = URL tunnel (ví dụ `https://xxxx.trycloudflare.com`, không có dấu `/` ở cuối).
3. Deploy. Thêm URL Vercel vào `AEGIS_ALLOWED_ORIGINS` của backend (CORS).

## Lưu ý về Cloudflare quick tunnel
Quick tunnel **đổi URL mỗi lần khởi động lại**. Biến `NEXT_PUBLIC_AEGIS_API_URL` được nhúng vào lúc build, nên mỗi khi URL đổi phải: cập nhật biến trên Vercel rồi **Redeploy**. Khi laptop/tunnel tắt, người dùng thấy "Máy chủ demo đang tạm nghỉ".
