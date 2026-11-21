"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ServerStatus, { useServerState } from "./ServerStatus";
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

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 2.5l11.5 4v8.2c0 7.2-4.8 12.6-11.5 14.8C9.3 27.3 4.5 21.9 4.5 14.7V6.5z" fill="var(--accent)" />
      <circle cx="16" cy="14.5" r="5" fill="none" stroke="var(--accent-fg)" strokeWidth="2" />
      <circle cx="16" cy="14.5" r="1.8" fill="var(--accent-fg)" />
    </svg>
  );
}

export default function Demo({ children }: { children?: React.ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const server = useServerState();

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

  return (
    <>
      <header className="top">
        <div className="wrap top-in">
          <div className="brand">
            <Logo />
            <span className="brand-name">AEGIS</span>
            <span className="badge">Bản demo nghiên cứu</span>
          </div>
          {API_URL && <ServerStatus state={server} />}
        </div>
      </header>

      <main>
        <section className="wrap hero">
          <h1>Ảnh này do AI tạo ra, hay là ảnh thật?</h1>
          <p className="lead">
            Tải một ảnh lên để AEGIS ước lượng khả năng ảnh do AI tạo ra, kèm mức độ chắc chắn và
            cảnh báo khi kết quả đáng ngờ.
          </p>
        </section>

        <section className="wrap demo" aria-label="Kiểm tra ảnh">
          {!API_URL ? (
            <div className="card err" role="alert">
              <h2 className="h3">Thiếu cấu hình</h2>
              <p>
                Biến môi trường <code>NEXT_PUBLIC_AEGIS_API_URL</code> chưa được đặt nên trang không
                biết gọi máy chủ nào. Xem <code>web/README.md</code> để cấu hình.
              </p>
            </div>
          ) : (
            <>
              {server === "off" && !file && (
                <p className="notice" role="note">
                  Máy chủ demo hiện đang tắt (ngoài giờ demo). Bạn vẫn có thể chọn ảnh, nhưng kết
                  quả chỉ có khi máy chủ hoạt động. Vui lòng quay lại trong giờ demo.
                </p>
              )}

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
                  <svg className="drop-ico" width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <rect x="3" y="4" width="18" height="16" rx="3" />
                    <circle cx="9" cy="10" r="1.6" />
                    <path d="M21 16l-5-5-8 8" />
                  </svg>
                  <p className="drop-t">{drag ? "Thả ảnh để bắt đầu" : "Kéo và thả ảnh vào đây"}</p>
                  <p className="hint">hoặc</p>
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
                <div className="grid">
                  <div className="card pv">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img className="preview" src={preview} alt="Ảnh bạn đã chọn" />
                    <p className="fname" title={file?.name}>
                      {file?.name}
                    </p>
                    {loading && (
                      <p className="loading" role="status" aria-live="polite">
                        <span className="spin" aria-hidden="true" /> Đang phân tích ảnh…
                      </p>
                    )}
                    {!loading && (
                      <button className="btn ghost" onClick={reset}>
                        Kiểm tra ảnh khác
                      </button>
                    )}
                  </div>
                  <div>{result && <Result r={result} />}</div>
                </div>
              )}

              {!preview && error && (
                <p>
                  <button className="btn ghost" onClick={reset}>
                    Kiểm tra ảnh khác
                  </button>
                </p>
              )}
            </>
          )}
        </section>

        {children}
      </main>
    </>
  );
}

function Gauge({ p }: { p: number }) {
  return (
    <svg className="gauge" viewBox="0 0 120 70" aria-hidden="true">
      <path d="M10 62 A50 50 0 0 1 110 62" pathLength={100} className="g-track" />
      <path d="M10 62 A50 50 0 0 1 110 62" pathLength={100} className="g-fill" strokeDasharray={`${Math.max(p * 100, 0.5)} 100`} />
      <line x1="60" y1="6" x2="60" y2="16" className="g-mid" />
    </svg>
  );
}

function Result({ r }: { r: PredictResponse }) {
  const isAI = r.label === "AI";
  const q = describeQuality(r.quality);
  return (
    <div className="card result" aria-live="polite">
      <div className={`verdict ${isAI ? "ai" : "real"}`}>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          {isAI ? <path d="M12 3l2.2 5.6L20 11l-5.8 2.4L12 19l-2.2-5.6L4 11l5.8-2.4z" /> : <path d="M5 12.5l4.5 4.5L19 7.5" />}
        </svg>
        <span>{isAI ? "Nhiều khả năng là ảnh AI" : "Nhiều khả năng là ảnh thật"}</span>
        <span className="tag">{r.label}</span>
      </div>

      <div className={`meter ${isAI ? "ai" : "real"}`}>
        <Gauge p={r.p_ai} />
        <div
          className="m-val"
          role="meter"
          aria-label="Xác suất ảnh do AI tạo ra"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(r.p_ai * 100)}
        >
          <strong>{pct(r.p_ai, 1)}</strong>
          <span>xác suất ảnh do AI tạo ra</span>
        </div>
        <div className="scale" aria-hidden="true">
          <span>0% Thật</span>
          <span>100% AI</span>
        </div>
      </div>
      <p className="conf">
        Độ tin cậy: <strong>{pct(r.confidence, 1)}</strong>
      </p>

      {r.warning_flag && (
        <div className="flag" role="alert">
          <strong>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 3l10 18H2z" />
              <path d="M12 10v5M12 18.2v.1" />
            </svg>
            Cờ vàng – hãy thận trọng với kết quả này
          </strong>
          <ul>
            {r.flag_reasons.map((x) => (
              <li key={x}>{REASON_TEXT[x] ?? x}</li>
            ))}
          </ul>
        </div>
      )}

      <h3 className="h3">Ý kiến của ba chuyên gia</h3>
      <ul className="experts">
        {EXPERTS.map(({ key, name }) => {
          const e = r.experts[key];
          return (
            <li key={key}>
              <div className="e-name">{name}</div>
              <div className="e-line">
                <span className="e-lbl">Khả năng AI</span>
                <div className="track" aria-hidden="true">
                  <div className="f-ai" style={{ width: pct(e.p_ai) }} />
                </div>
                <span className="e-num">{pct(e.p_ai)}</span>
              </div>
              <div className="e-line">
                <span className="e-lbl">Trọng số</span>
                <div className="track" aria-hidden="true">
                  <div className="f-w" style={{ width: pct(e.weight) }} />
                </div>
                <span className="e-num">{e.weight.toFixed(2)}</span>
              </div>
            </li>
          );
        })}
      </ul>

      <h3 className="h3">Hồ sơ chất lượng ảnh</h3>
      <ul className="quality">
        {q.map((x) => (
          <li key={x.k}>
            <span className="q-k">{x.k}</span>
            <strong className="q-v">{x.v}</strong>
            <span className="q-d">{x.d}</span>
          </li>
        ))}
      </ul>

      <p className="meta">
        Mô hình {r.model_version} · {r.latency_ms} ms
      </p>
    </div>
  );
}
