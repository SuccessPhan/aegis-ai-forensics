import type { Metadata, Viewport } from "next";
import { JetBrains_Mono, Lexend } from "next/font/google";
import Nav from "./Nav";
import "./globals.css";

const sans = Lexend({
  subsets: ["latin", "vietnamese"],
  weight: ["300", "400", "500", "700"],
  display: "swap",
  variable: "--f-sans",
});

const mono = JetBrains_Mono({
  subsets: ["latin", "vietnamese"],
  weight: ["400", "700"],
  display: "swap",
  variable: "--f-mono",
});

export const metadata: Metadata = {
  title: "AEGIS – Kiểm tra ảnh do AI tạo ra",
  description: "AEGIS: ba chuyên gia và một bộ điều phối cùng đánh giá ảnh có phải do AI tạo ra hay không.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0b1020",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <Nav />
        {children}
      </body>
    </html>
  );
}
