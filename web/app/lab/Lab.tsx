"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, ApiError, FlagReason, LabRunResponse, LabSample, LabVariant, labRun, labSamples } from "../api";

const REASON_TEXT: Record<FlagReason, string> = {
  conflict: "mâu thuẫn",
  uncertain: "lưỡng lự",
  ood: "ảnh lạ",
};

const EXPERTS: { key: "clip" | "dinov2" | "freq"; name: string; cls: string }[] = [
  { key: "clip", name: "CLIP", cls: "w-clip" },
  { key: "dinov2", name: "DINOv2", cls: "w-dino" },
  { key: "freq", name: "Tần số", cls: "w-freq" },
];

const SOURCE: Record<string, string> = { adm: "ADM", sd_v1_4: "Stable Diffusion v1.4" };
const pct = (x: number) => `${Math.round(x * 100)}%`;

function answerText(truth: "Real" | "AI", source: string) {
  return truth === "Real" ? "ảnh thật" : `ảnh AI (${SOURCE[source] ?? source})`;
}

export default function Lab() {
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
    <main className="wrap lab">
      <h1 className="h1 lab-h">Khi ảnh bị làm mờ hay nén, bộ điều phối tin chuyên gia nào?</h1>

      {!API_URL ? (
        <div className="panel err" role="alert">
          Chưa đặt biến <code>NEXT_PUBLIC_AEGIS_API_URL</code>.
        </div>
      ) : (
        <>
          <section className="panel picker" aria-label="Chọn ảnh mẫu">
            {loadErr ? (
              <div className="lab-err" role="alert">
                <span>{loadErr}</span>
                <button className="btn ghost" onClick={load}>
                  Tải lại
                </button>
              </div>
            ) : !samples ? (
              <p className="loading" role="status">
                <span className="spin" aria-hidden="true" /> đang tải ảnh mẫu
              </p>
            ) : (
              <div className="pick-row">
                <fieldset className="pick-set">
                  <legend>Chọn một ảnh</legend>
                  <div className="pick-grid">
                    {samples.map((s, i) => (
                      <label key={s.id} className={`pick${sel === s.id ? " on" : ""}`}>
                        <input type="radio" name="sample" className="sr" checked={sel === s.id} onChange={() => choose(s.id)} />
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
          </section>

          {runErr && (
            <div className="panel err" role="alert">
              {runErr}
            </div>
          )}
          {running && (
            <p className="loading lab-loading" role="status" aria-live="polite">
              <span className="spin" aria-hidden="true" /> đang chạy 4 phiên bản của ảnh
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
              </p>
            </section>
          )}
        </>
      )}
    </main>
  );
}

function VariantCard({ v }: { v: LabVariant }) {
  const isAI = v.label === "AI";
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);
  const total = EXPERTS.reduce((a, e) => a + v.experts[e.key].weight, 0) || 1;
  return (
    <li className="panel vcard">
      <h2 className="v-title">{v.title}</h2>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="v-img" src={v.thumb} alt={`${v.title}: ảnh sau xử lý`} />
      <div className="v-verdict">
        <span className={`v-lab ${isAI ? "ai" : "real"}`}>{isAI ? "AI" : "Thật"}</span>
        <span className="v-p">p = {v.p_ai.toFixed(2)}</span>
      </div>
      <div
        className="wbar"
        role="img"
        aria-label={`Trọng số: ${EXPERTS.map((e) => `${e.name} ${pct(v.experts[e.key].weight)}`).join(", ")}`}
      >
        {EXPERTS.map((e) => {
          const w = v.experts[e.key].weight;
          return (
            <span key={e.key} className={`seg ${e.cls}`} style={{ width: `${shown ? (w / total) * 100 : 0}%` }}>
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
      <p className={`flag sm${v.warning_flag ? " on" : ""}`}>
        {v.warning_flag ? `cờ vàng: ${v.flag_reasons.map((x) => REASON_TEXT[x] ?? x).join(", ")}` : "cờ vàng: tắt"}
      </p>
    </li>
  );
}
