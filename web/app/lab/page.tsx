import type { Metadata } from "next";
import Lab from "./Lab";

export const metadata: Metadata = {
  title: "AEGIS – Chế độ phòng thí nghiệm",
  description: "Xem trọng số ba chuyên gia của AEGIS thay đổi khi ảnh bị làm mờ hoặc nén mạnh.",
  robots: { index: false, follow: false },
};

export default function LabPage() {
  return <Lab />;
}
