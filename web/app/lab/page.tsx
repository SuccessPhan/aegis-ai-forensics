import type { Metadata } from "next";
import Lab from "./Lab";

export const metadata: Metadata = {
  title: "AEGIS – Bộ điều phối",
  description: "Trọng số ba chuyên gia của AEGIS thay đổi thế nào khi ảnh bị làm mờ hoặc nén mạnh.",
  robots: { index: false, follow: false },
};

export default function LabPage() {
  return <Lab />;
}
