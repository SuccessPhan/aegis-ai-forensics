import type { Metadata, Viewport } from "next";
import { Be_Vietnam_Pro } from "next/font/google";
import "./globals.css";

const font = Be_Vietnam_Pro({
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

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
    <html lang="vi" className={font.className}>
      <body>{children}</body>
    </html>
  );
}
