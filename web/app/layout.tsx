import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AEGIS – Kiểm tra ảnh do AI tạo ra",
  description:
    "Bản demo nghiên cứu: hệ thống AEGIS phát hiện ảnh do AI tạo ra. Ảnh không được lưu.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
