"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import ServerStatus, { useServerState } from "../ServerStatus";
import { API_URL, ApiError, FlagReason, LabRunResponse, LabSample, LabVariant, labRun, labSamples } from "../api";

const REASON_TEXT: Record<FlagReason, string> = {
  conflict: "Các chuyên gia mâu thuẫn",
  uncertain: "Kết quả lưỡng lự",
  ood: "Ảnh lạ so với dữ liệu đã học",
};

const EXPERTS: { key: "clip" | "dinov2" | "freq"; name: string; cls: string }[] = [
  { key: "clip", name: "CLIP", cls: "w-clip" },
  { key: "dinov2", name: "DINOv2", cls: "w-dino" },
  { key: "freq", name: "Tần số SRM", cls: "w-freq" },
];

const SOURCE: Record<string, string> = { adm: "ADM", sd_v1_4: "Stable Diffusion v1.4" };
const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;

function answerText(truth: "Real" | "AI", source: string) {
  return truth === "Real" ? "Ảnh thật" : `Ảnh AI (${SOURCE[source] ?? source})`;
}

export default function Lab() {
  const server = useServerState();
  const [samples, setSamples] = useState<LabSample[] | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runErr, setRunErr] = useState<string | null>(null);
  const [result, setResult] = useState<LabRunResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(() => {
    if (!API_URL) return;
    const ac = new AbortController();
    setSamples(null);
    setLoadErr(null);
    labSamples(ac.signal)
      .then(setSamples)
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setLoadErr(e instanceof ApiError ? e.message : "Đã có lỗi xảy ra. Vui lòng thử lại.");
      });
    return () => ac.abort();
  }, []);

  useEffect(() => load(), [load]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const run = async () => {
    if (!sel) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setRunning(true);
    setRunErr(null);
    setResult(null);
    try {
      setResult(await labRun(sel, ac.signal));
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setRunErr(e instanceof ApiError ? e.message : "Đã có lỗi xảy ra. Vui lòng thử lại.");
    } finally {
      if (abortRef.current === ac) setRunning(false);
    }
  };

  const choose = (id: string) => {
    setSel(id);
    setResult(null);
    setRunErr(null);
  };

  return (
    <>
      <header className="top">
        <div className="wrap top-in">
          <div className="brand">
            <span className="brand-name">AEGIS</span>
            <span className="badge">Phòng thí nghiệm</span>
          </div>
          <div className="top-r">
            <Link className="lablink" href="/">← Quay lại trang chính</Link>
            {API_URL && <ServerStatus state={server} />}
          </div>
        </div>
      </header>

      <main className="lab wrap">
        <h1 className="lab-h">Khi ảnh bị làm mờ, AEGIS tin chuyên gia nào hơn?</h1>

        {!API_URL ? (
          <div className="card err" role="alert">
            Biến môi trường <code>NEXT_PUBLIC_AEGIS_API_URL</code> chưa được đặt nên trang không biết gọi máy chủ nào.
          </div>
        ) : (
          <>
            {server === "off" && (
              <p className="notice lab-notice" role="note">
                Máy chủ demo hiện đang tắt (ngoài giờ demo). Kết quả chỉ có khi máy chủ hoạt động.
              </p>
            )}

            <section className="picker card" aria-label="Chọn ảnh mẫu">
              {loadErr ? (
                <div className="lab-err" role="alert">
                  <span>{loadErr}</span>
                  <button className="btn ghost" onClick={load}>Tải lại</button>
                </div>
              ) : !samples ? (
                <p className="loading" role="status"><span className="spin" aria-hidden="true" /> Đang tải ảnh mẫu…</p>
              ) : (
                <div className="pick-row">
                  <fieldset className="pick-set">
                    <legend>Chọn một ảnh mẫu (ảnh kiểm thử, chưa dùng khi huấn luyện)</legend>
                    <div className="pick-grid">
                      {samples.map((s, i) => (
                        <label key={s.id} className={`pick${sel === s.id ? " on" : ""}`}>
                          <input
                            type="radio"
                            name="sample"
                            className="sr"
                            checked={sel === s.id}
                            onChange={() => choose(s.id)}
                          />
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={s.thumb} alt={`Ảnh mẫu ${i + 1}`} />
                          <span className="pick-n">{i + 1}</span>
                        </label>
                      ))}
                    </div>
                  </fieldset>
                  <button className="btn go" onClick={run} disabled={!sel || running}>
                    {running ? "Đang chạy…" : "Chạy kiểm tra"}
                  </button>
                </div>
              )}
              {samples && !sel && !result && (
                <p className="hint pick-hint">Dùng chuột hoặc phím mũi tên để chọn, rồi bấm “Chạy kiểm tra”.</p>
              )}
            </section>

            {runErr && (
              <div className="card err lab-run-err" role="alert">
                <strong>Lỗi:</strong> {runErr}
              </div>
            )}
            {running && (
              <p className="loading lab-loading" role="status" aria-live="polite">
                <span className="spin" aria-hidden="true" /> Đang kiểm tra 4 phiên bản của ảnh…
              </p>
            )}

            {result && (
              <section aria-label="Kết quả" aria-live="polite">
                <ul className="vgrid">
                  {result.variants.map((v) => (
                    <VariantCard key={v.key} v={v} />
                  ))}
                </ul>
                <p className="answer">
                  Đáp án: <strong>{answerText(result.truth, result.source)}</strong>
                  <span className="ans-meta"> · mô hình {result.model_version} · {result.latency_ms} ms</span>
                </p>
              </section>
            )}
          </>
        )}
      </main>
    </>
  );
}

function VariantCard({ v }: { v: LabVariant }) {
  const isAI = v.label === "AI";
  const [shown, setShown] = useState(false);
  // Chiều rộng bắt đầu từ 0 rồi chuyển mượt tới giá trị thật.
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);
  const total = EXPERTS.reduce((a, e) => a + v.experts[e.key].weight, 0) || 1;
  return (
    <li className="vcard card">
      <h2 className="v-title">{v.title}</h2>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="v-img" src={v.thumb} alt={`${v.title}: ảnh sau xử lý`} />
      <div className={`v-verdict ${isAI ? "ai" : "real"}`}>
        <span className="v-lab">{isAI ? "Ảnh AI" : "Ảnh thật"}</span>
        <span className="v-p">
          {pct(v.p_ai)} <small>AI</small>
        </span>
      </div>
      <div
        className="wbar"
        role="img"
        aria-label={`Trọng số chuyên gia: ${EXPERTS.map((e) => `${e.name} ${pct(v.experts[e.key].weight)}`).join(", ")}`}
      >
        {EXPERTS.map((e) => {
          const w = v.experts[e.key].weight;
          const width = shown ? (w / total) * 100 : 0;
          return (
            <span key={e.key} className={`seg ${e.cls}`} style={{ width: `${width}%` }}>
              {w >= 0.15 ? pct(w) : ""}
            </span>
          );
        })}
      </div>
      <ul className="wlegend">
        {EXPERTS.map((e) => (
          <li key={e.key}>
            <span className={`sw ${e.cls}`} aria-hidden="true" />
            <span>{e.name}</span>
            <strong>{pct(v.experts[e.key].weight)}</strong>
          </li>
        ))}
      </ul>
      <div className="v-flagslot">
        {v.warning_flag && (
          <div className="vflag" role="alert">
            <strong>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 3l10 18H2z" />
                <path d="M12 10v5M12 18.2v.1" />
              </svg>
              Cờ vàng
            </strong>
            <span>{v.flag_reasons.map((x) => REASON_TEXT[x] ?? x).join("; ") || "Hãy thận trọng"}</span>
          </div>
        )}
      </div>
    </li>
  );
}
