import type { Metadata } from "next";
import Results from "./Results";

export const metadata: Metadata = {
  title: "AEGIS – Số liệu",
  description: "AUROC của AEGIS, từng chuyên gia và các mô hình học máy trên cùng tập kiểm thử.",
  robots: { index: false, follow: false },
};

export default function StatsPage() {
  return <Results />;
}
