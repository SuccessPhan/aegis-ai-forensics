"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, ApiError, FlagReason, PredictResponse, predict, validateFile } from "./api";

const REASON_TEXT: Record<FlagReason, string> = {
  conflict: "Các chuyên gia mâu thuẫn mạnh",
  uncertain: "Kết quả nằm trong vùng lưỡng lự",
  ood: "Ảnh khác xa dữ liệu hệ thống đã học",
};

const EXPERTS: { key: "clip" | "dinov2" | "freq"; name: string }[] = [
  { key: "clip", name: "CLIP – ngữ nghĩa" },
  { key: "dinov2", name: "DINOv2 – cấu trúc" },
  { key: "freq", name: "Tần số – SRM" },
];

const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;

function describeQuality(q: PredictResponse["quality"]) {
  const j = q.jpeg_quality_est;
  const jpeg =
    j >= 90
      ? "gần như không bị nén hoặc nén rất nhẹ"
      : j >= 70
        ? "đã bị nén JPEG ở mức vừa phải"
        : "đã bị nén JPEG khá mạnh";
  const l = q.log_laplacian_var;
  const sharp =
    l >= 6 ? "khá sắc nét, nhiều chi tiết" : l >= 4 ? "độ sắc nét trung bình" : "khá mờ hoặc ít chi tiết";
  const h = q.high_freq_energy_ratio;
  const hf =
    h >= 0.05
      ? "nhiều chi tiết mịn/nhiễu tần số cao"
      : h >= 0.015
        ? "lượng chi tiết mịn ở mức vừa"
        : "ít chi tiết mịn (ảnh mượt)";
  return [
    { k: "Mức nén JPEG ước lượng", v: `${j.toFixed(0)}/100`, d: jpeg },
    { k: "Độ sắc nét", v: l.toFixed(2), d: sharp },
    { k: "Năng lượng tần số cao", v: h.toFixed(3), d: hf },
  ];
}

export default function Demo() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const run = useCallback(async (f: File) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await predict(f, ac.signal);
      setResult(r);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e instanceof ApiError ? e.message : "Đã có lỗi xảy ra. Vui lòng thử lại.");
    } finally {
      if (abortRef.current === ac) setLoading(false);
    }
  }, []);

  const pick = useCallback(
    (f: File | undefined | null) => {
      if (!f) return;
      const err = validateFile(f);
      setResult(null);
      if (err) {
        setError(err);
        setFile(null);
        setPreview(null);
        return;
      }
      setError(null);
      setFile(f);
      setPreview(URL.createObjectURL(f));
      void run(f);
    },
    [run],
  );

  if (!API_URL) {
    return (
      <section className="card err" role="alert">
        <h2>Thiếu cấu hình</h2>
        <p>
          Biến môi trường <code>NEXT_PUBLIC_AEGIS_API_URL</code> chưa được đặt nên trang không biết
          gọi máy chủ nào. Xem <code>web/README.md</code> để cấu hình.
        </p>
      </section>
    );
  }

  return (
    <section aria-label="Kiểm tra ảnh">
      {!file && (
        <div
          className={`drop${drag ? " drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            pick(e.dataTransfer.files?.[0]);
          }}
        >
          <p>Kéo và thả ảnh vào đây, hoặc</p>
          <label className="btn" htmlFor="file">
            Chọn ảnh từ thiết bị
          </label>
          <input
            id="file"
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="sr"
            onChange={(e) => pick(e.target.files?.[0])}
          />
          <p className="hint">JPEG, PNG hoặc WebP · tối đa 10 MB · cạnh ngắn nhất từ 224 px</p>
          <p className="hint">Ảnh được gửi nguyên bản, không bị nén hay thu nhỏ.</p>
        </div>
      )}

      {error && (
        <div className="card err" role="alert">
          <strong>Lỗi:</strong> {error}
          {file && (
            <p>
              <button className="btn" onClick={() => run(file)}>
                Thử lại
              </button>
            </p>
          )}
        </div>
      )}

      {preview && (
        <div className="card">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="preview" src={preview} alt="Ảnh bạn đã chọn" />
          {loading && (
            <p className="loading" role="status" aria-live="polite">
              <span className="spin" aria-hidden="true" /> Đang phân tích ảnh…
            </p>
          )}
        </div>
      )}

      {result && <Result r={result} />}

      {(file || error) && !loading && (
        <p>
          <button className="btn" onClick={reset}>
            Kiểm tra ảnh khác
          </button>
        </p>
      )}
    </section>
  );
}

function Result({ r }: { r: PredictResponse }) {
  const isAI = r.label === "AI";
  const q = describeQuality(r.quality);
  return (
    <div className="card" aria-live="polite">
      <h2>Kết quả</h2>
      <p className={`label ${isAI ? "ai" : "real"}`}>
        {isAI ? "Nhiều khả năng là ảnh AI" : "Nhiều khả năng là ảnh thật"}{" "}
        <span className="tag">{r.label}</span>
      </p>
      <div className="row">
        <span>Xác suất ảnh AI (p_ai)</span>
        <strong>{pct(r.p_ai, 1)}</strong>
      </div>
      <div
        className="bar"
        role="meter"
        aria-label="Xác suất ảnh AI"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(r.p_ai * 100)}
      >
        <div className="fill" style={{ width: pct(r.p_ai) }} />
        <div className="mid" />
      </div>
      <div className="scale">
        <span>Thật</span>
        <span>AI</span>
      </div>
      <p>
        Độ tin cậy: <strong>{pct(r.confidence, 1)}</strong>
      </p>

      {r.warning_flag && (
        <div className="flag" role="alert">
          <strong>⚠ Cờ vàng – hãy thận trọng với kết quả này</strong>
          <ul>
            {r.flag_reasons.map((x) => (
              <li key={x}>{REASON_TEXT[x] ?? x}</li>
            ))}
          </ul>
        </div>
      )}

      <h3>Ý kiến của ba chuyên gia</h3>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Chuyên gia</th>
              <th scope="col">p_ai</th>
              <th scope="col">Trọng số</th>
            </tr>
          </thead>
          <tbody>
            {EXPERTS.map(({ key, name }) => {
              const e = r.experts[key];
              return (
                <tr key={key}>
                  <th scope="row">{name}</th>
                  <td>{pct(e.p_ai)}</td>
                  <td>
                    <div className="wrow">
                      <div className="wbar" aria-hidden="true">
                        <div style={{ width: pct(e.weight) }} />
                      </div>
                      <span>{e.weight.toFixed(2)}</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>Hồ sơ chất lượng ảnh</h3>
      <ul className="quality">
        {q.map((x) => (
          <li key={x.k}>
            <strong>{x.k}:</strong> {x.v} – {x.d}
          </li>
        ))}
      </ul>

      <p className="meta">
        Mô hình {r.model_version} · {r.latency_ms} ms
      </p>
    </div>
  );
}
