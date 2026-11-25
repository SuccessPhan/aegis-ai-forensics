// AUROC on test_benchmark / s4_eval, copied from reports/ml_comparison/summary.md (commit 4a1f225).
// Columns: S1 clean, S2 JPEG (all levels), S3 blur/resize (all levels), S4 Midjourney (no few-shot adaptation).

export type Scenario = "S1" | "S2" | "S3" | "S4";
export type Head = "clean" | "aug";
export type Row = { name: string; vals: [number, number, number, number]; main?: boolean };
export type Group = { title: string; rows: Row[] };

export const SCENARIOS: { key: Scenario; label: string; sub: string }[] = [
  { key: "S1", label: "Ảnh gốc", sub: "S1" },
  { key: "S2", label: "Nén JPEG", sub: "S2 · Q 85/65/40" },
  { key: "S3", label: "Mờ / thu nhỏ", sub: "S3 · σ 1.5/2.5 · 0.5×" },
  { key: "S4", label: "Midjourney", sub: "S4 · chưa thấy khi học" },
];

export const HEADS: { key: Head; label: string }[] = [
  { key: "clean", label: "Học trên ảnh sạch" },
  { key: "aug", label: "Học có tăng cường" },
];

export const DATA: Record<Head, Group[]> = {
  clean: [
    {
      title: "Mô hình của AEGIS",
      rows: [
        { name: "AEGIS (bộ điều phối)", vals: [0.993, 0.865, 0.822, 0.957], main: true },
        { name: "Static LR (trọng số cố định)", vals: [0.993, 0.869, 0.808, 0.946] },
        { name: "Chuyên gia CLIP", vals: [0.994, 0.839, 0.682, 0.961] },
        { name: "Chuyên gia DINOv2", vals: [0.891, 0.774, 0.805, 0.802] },
        { name: "Chuyên gia tần số", vals: [0.985, 0.548, 0.541, 0.829] },
      ],
    },
    {
      title: "Học máy trên 92 đặc trưng thủ công",
      rows: [
        { name: "SVM (RBF)", vals: [0.948, 0.746, 0.544, 0.789] },
        { name: "Random Forest", vals: [0.899, 0.681, 0.542, 0.81] },
        { name: "XGBoost", vals: [0.947, 0.753, 0.607, 0.798] },
      ],
    },
    {
      title: "Học máy trên đặc trưng CLIP",
      rows: [
        { name: "SVM (RBF)", vals: [0.994, 0.861, 0.726, 0.962] },
        { name: "Random Forest", vals: [0.969, 0.804, 0.765, 0.93] },
        { name: "XGBoost", vals: [0.985, 0.84, 0.76, 0.941] },
      ],
    },
    {
      title: "Học máy trên đặc trưng DINOv2",
      rows: [
        { name: "SVM (RBF)", vals: [0.885, 0.797, 0.82, 0.811] },
        { name: "Random Forest", vals: [0.845, 0.722, 0.768, 0.784] },
        { name: "XGBoost", vals: [0.837, 0.762, 0.781, 0.76] },
      ],
    },
  ],
  aug: [
    {
      title: "Mô hình của AEGIS",
      rows: [
        { name: "AEGIS (bộ điều phối)", vals: [0.996, 0.928, 0.868, 0.918], main: true },
        { name: "Static LR (trọng số cố định)", vals: [0.996, 0.932, 0.871, 0.916] },
        { name: "Chuyên gia CLIP", vals: [0.987, 0.899, 0.823, 0.928] },
        { name: "Chuyên gia DINOv2", vals: [0.874, 0.782, 0.813, 0.783] },
        { name: "Chuyên gia tần số", vals: [0.967, 0.803, 0.564, 0.774] },
      ],
    },
    {
      title: "Học máy trên 92 đặc trưng thủ công",
      rows: [
        { name: "SVM (RBF)", vals: [0.94, 0.77, 0.678, 0.773] },
        { name: "Random Forest", vals: [0.895, 0.737, 0.657, 0.791] },
        { name: "XGBoost", vals: [0.934, 0.775, 0.679, 0.766] },
      ],
    },
    {
      title: "Học máy trên đặc trưng CLIP",
      rows: [
        { name: "SVM (RBF)", vals: [0.991, 0.901, 0.834, 0.947] },
        { name: "Random Forest", vals: [0.961, 0.825, 0.788, 0.924] },
        { name: "XGBoost", vals: [0.981, 0.874, 0.808, 0.917] },
      ],
    },
    {
      title: "Học máy trên đặc trưng DINOv2",
      rows: [
        { name: "SVM (RBF)", vals: [0.859, 0.794, 0.812, 0.775] },
        { name: "Random Forest", vals: [0.833, 0.748, 0.764, 0.766] },
        { name: "XGBoost", vals: [0.818, 0.757, 0.771, 0.747] },
      ],
    },
  ],
};
