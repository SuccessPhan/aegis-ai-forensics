"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, ApiError, FlagReason, PredictResponse, predict, validateFile } from "./api";

const REASON_TEXT: Record<FlagReason, string> = {
  conflict: "các chuyên gia mâu thuẫn",
  uncertain: "kết quả lưỡng lự",
  ood: "ảnh khác xa dữ liệu đã học",
};

const EXPERTS: { key: "clip" | "dinov2" | "freq"; name: string; role: string }[] = [
  { key: "clip", name: "CLIP", role: "ngữ nghĩa" },
  { key: "dinov2", name: "DINOv2", role: "cấu trúc" },
  { key: "freq", name: "Tần số", role: "dấu vết SRM" },
];

const pct = (x: number) => `${Math.round(x * 100)}%`;

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
  // On phones the result panel sits below the upload column: bring it into view once it arrives.
  useEffect(() => {
    if (!result || window.innerWidth >= 900) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.querySelector(".result")?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }, [result]);

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
      setResult(await predict(f, ac.signal));
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
    <main className="wrap home">
      <section className="home-l" aria-label="Tải ảnh lên">
        <h1 className="h1">Ảnh này có phải do AI tạo ra?</h1>
        <p className="sub">Ngữ nghĩa, cấu trúc và tần số: ba góc nhìn, một bộ điều phối chia trọng số cho từng ảnh.</p>

        {!API_URL ? (
          <div className="panel err" role="alert">
            Chưa đặt biến <code>NEXT_PUBLIC_AEGIS_API_URL</code>.
          </div>
        ) : !file ? (
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
            <svg className="drop-ico" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="4" width="18" height="16" rx="3" />
              <circle cx="9" cy="10" r="1.6" />
              <path d="M21 16l-5-5-8 8" />
            </svg>
            <p className="drop-t">{drag ? "Thả ảnh để phân tích" : "Kéo ảnh vào đây"}</p>
            <label className="btn" htmlFor="file">
              Chọn ảnh
            </label>
            <input
              id="file"
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="sr"
              onChange={(e) => pick(e.target.files?.[0])}
            />
            <p className="hint">JPEG · PNG · WebP · tối đa 10 MB</p>
          </div>
        ) : (
          <div className="panel pv">
            {preview && (
              // eslint-disable-next-line @next/next/no-img-element
              <img className="preview" src={preview} alt="Ảnh đã chọn" />
            )}
            <div className="pv-row">
              <span className="fname" title={file.name}>
                {file.name}
              </span>
              {loading ? (
                <span className="loading" role="status" aria-live="polite">
                  <span className="spin" aria-hidden="true" /> đang phân tích
                </span>
              ) : (
                <button className="btn ghost" onClick={reset}>
                  Ảnh khác
                </button>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="panel err" role="alert">
            {error}
            <div className="err-act">
              {file && (
                <button className="btn ghost" onClick={() => run(file)}>
                  Thử lại
                </button>
              )}
              <button className="btn ghost" onClick={reset}>
                Chọn ảnh khác
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="home-r" aria-live="polite" aria-label="Kết quả">
        {result ? <Result r={result} /> : <Empty loading={loading} />}
      </section>
    </main>
  );
}

function Empty({ loading }: { loading: boolean }) {
  return (
    <div className="panel result empty">
      <div className="verdict">
        <span className="big dim">{loading ? "…" : "—"}</span>
        <span className="pval">p(ai) = –</span>
      </div>
      <div className="meter" aria-hidden="true" />
      <Rows />
      <p className="flag">{loading ? "đang chạy 3 chuyên gia" : "chưa có ảnh"}</p>
    </div>
  );
}

function Rows({ r }: { r?: PredictResponse }) {
  return (
    <div className="rows">
      <div className="row head">
        <span>chuyên gia</span>
        <span>ý kiến p(ai)</span>
        <span />
        <span>trọng số</span>
      </div>
      {EXPERTS.map(({ key, name, role }) => {
        const e = r?.experts[key];
        return (
          <div className="row" key={key}>
            <span className="ename">
              {name}
              <small>{role}</small>
            </span>
            <span className="bar" aria-hidden="true">
              <span style={{ width: e ? `${e.p_ai * 100}%` : 0 }} />
            </span>
            <span className="num">{e ? e.p_ai.toFixed(2) : "–"}</span>
            <span className="num w">{e ? pct(e.weight) : "–"}</span>
          </div>
        );
      })}
    </div>
  );
}

function Result({ r }: { r: PredictResponse }) {
  const isAI = r.label === "AI";
  return (
    <div className="panel result">
      <div className="verdict">
        <span className={`big ${isAI ? "ai" : "real"}`}>{isAI ? "AI" : "Ảnh thật"}</span>
        <span className="pval">p(ai) = {r.p_ai.toFixed(3)}</span>
      </div>
      <div
        className="meter"
        role="meter"
        aria-label="Xác suất ảnh do AI tạo ra"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(r.p_ai * 100)}
      >
        <span className="needle" style={{ left: `${r.p_ai * 100}%` }} />
      </div>
      <div className="scale" aria-hidden="true">
        <span>thật</span>
        <span>0.5</span>
        <span>AI</span>
      </div>
      <Rows r={r} />
      <p className={`flag${r.warning_flag ? " on" : ""}`}>
        {r.warning_flag
          ? `cờ vàng: ${r.flag_reasons.map((x) => REASON_TEXT[x] ?? x).join(", ")}`
          : "cờ vàng: tắt"}
      </p>
      <p className="meta">
        ood {r.ood_score.toFixed(2)} / {r.ood_threshold.toFixed(2)} · jpeg≈{r.quality.jpeg_quality_est.toFixed(0)} ·{" "}
        {r.latency_ms} ms
      </p>
    </div>
  );
}
