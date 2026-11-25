"use client";

import { useState } from "react";
import { DATA, HEADS, Head, SCENARIOS, Scenario } from "./data";

const IDX: Record<Scenario, number> = { S1: 0, S2: 1, S3: 2, S4: 3 };
// Bars start at 0.5 (chance) and end at 1.0.
const width = (v: number) => `${Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * 100}%`;

export default function Results() {
  const [sc, setSc] = useState<Scenario>("S3");
  const [head, setHead] = useState<Head>("clean");
  const i = IDX[sc];
  const groups = DATA[head];
  const aegis = groups[0].rows[0].vals[i];

  return (
    <main className="wrap stats">
      <h1 className="h1">AEGIS so với các mô hình học máy</h1>

      <div className="controls">
        <div className="seg-ctl" role="tablist" aria-label="Kịch bản">
          {SCENARIOS.map((s) => (
            <button
              key={s.key}
              role="tab"
              aria-selected={sc === s.key}
              className={sc === s.key ? "on" : undefined}
              onClick={() => setSc(s.key)}
            >
              {s.label}
              <small>{s.sub}</small>
            </button>
          ))}
        </div>
        <div className="seg-ctl small" role="group" aria-label="Cách huấn luyện head">
          {HEADS.map((h) => (
            <button key={h.key} aria-pressed={head === h.key} className={head === h.key ? "on" : undefined} onClick={() => setHead(h.key)}>
              {h.label}
            </button>
          ))}
        </div>
      </div>

      <section className="panel chart" aria-label="Biểu đồ AUROC">
        <div className="axis" aria-hidden="true">
          <span />
          <span className="ticks">
            <span>0.5</span>
            <span>0.75</span>
            <span>1.0</span>
          </span>
          <span>AUROC</span>
        </div>
        {groups.map((g) => (
          <div className="grp" key={g.title}>
            <h2 className="grp-t">{g.title}</h2>
            {g.rows.map((r) => {
              const v = r.vals[i];
              const d = v - aegis;
              return (
                <div className={`crow${r.main ? " main" : ""}`} key={g.title + r.name}>
                  <span className="cname">{r.name}</span>
                  <span className="ctrack">
                    <span style={{ width: width(v) }} />
                  </span>
                  <span className="cval">
                    {v.toFixed(3)}
                    {!r.main && <small className={d > 0 ? "up" : undefined}>{(d >= 0 ? "+" : "−") + Math.abs(d).toFixed(3)}</small>}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </section>
    </main>
  );
}
