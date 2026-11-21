export const API_URL = (process.env.NEXT_PUBLIC_AEGIS_API_URL ?? "").replace(/\/+$/, "");

export const MAX_BYTES = 10 * 1024 * 1024;
export const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export type FlagReason = "conflict" | "uncertain" | "ood";

export interface Expert {
  p_ai: number;
  weight: number;
}

export interface PredictResponse {
  label: "AI" | "Real";
  p_ai: number;
  confidence: number;
  warning_flag: boolean;
  flag_reasons: FlagReason[];
  experts: { clip: Expert; dinov2: Expert; freq: Expert };
  quality: {
    jpeg_quality_est: number;
    log_laplacian_var: number;
    high_freq_energy_ratio: number;
  };
  ood_score: number;
  ood_threshold: number;
  model_version: string;
  latency_ms: number;
}

/** Trả về thông báo lỗi tiếng Việt nếu file không hợp lệ, ngược lại null. */
export function validateFile(file: File): string | null {
  const name = file.name.toLowerCase();
  if (file.type === "image/heic" || file.type === "image/heif" || /\.(heic|heif)$/.test(name)) {
    return "Ảnh HEIC (iPhone) chưa được hỗ trợ. Hãy chuyển sang JPEG: vào Cài đặt > Máy ảnh > Định dạng > Tương thích nhất, hoặc xuất/chụp màn hình rồi tải lại.";
  }
  if (!ALLOWED_TYPES.includes(file.type)) {
    return "Định dạng không được hỗ trợ. Vui lòng chọn ảnh JPEG, PNG hoặc WebP.";
  }
  if (file.size > MAX_BYTES) {
    return "Ảnh lớn hơn 10 MB. Vui lòng chọn ảnh nhỏ hơn.";
  }
  if (file.size === 0) {
    return "File rỗng. Vui lòng chọn ảnh khác.";
  }
  return null;
}

export class ApiError extends Error {}

const STATUS_FALLBACK: Record<number, string> = {
  400: "Không đọc được ảnh. Vui lòng thử ảnh khác.",
  413: "Ảnh lớn hơn 10 MB. Vui lòng chọn ảnh nhỏ hơn.",
  429: "Bạn gửi quá nhiều yêu cầu. Vui lòng đợi khoảng một phút rồi thử lại.",
  503: "Máy chủ đang khởi động hoặc đang bận. Vui lòng thử lại sau ít phút.",
};

export async function predict(file: File, signal?: AbortSignal): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file, file.name);
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/predict`, { method: "POST", body, signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError("Máy chủ demo đang tạm nghỉ. Vui lòng thử lại sau.");
  }
  if (!res.ok) {
    let msg = "";
    try {
      const j = await res.json();
      if (j && typeof j.message === "string") msg = j.message;
    } catch {
      /* bỏ qua */
    }
    throw new ApiError(msg || STATUS_FALLBACK[res.status] || "Đã có lỗi xảy ra. Vui lòng thử lại.");
  }
  try {
    return (await res.json()) as PredictResponse;
  } catch {
    throw new ApiError("Máy chủ trả về dữ liệu không hợp lệ.");
  }
}
