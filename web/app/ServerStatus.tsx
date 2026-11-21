"use client";

import { useEffect, useState } from "react";
import { API_URL } from "./api";

type State = "checking" | "ok" | "loading" | "off";

const TEXT: Record<State, string> = {
  checking: "Đang kiểm tra máy chủ…",
  ok: "Đang hoạt động",
  loading: "Đang khởi động",
  off: "Ngoài giờ demo — máy chủ đang tắt",
};

export function useServerState(): State {
  const [state, setState] = useState<State>("checking");
  useEffect(() => {
    if (!API_URL) return;
    let alive = true;
    let ac: AbortController | null = null;
    const check = async () => {
      ac = new AbortController();
      const t = setTimeout(() => ac?.abort(), 8000);
      try {
        const res = await fetch(`${API_URL}/api/health`, { signal: ac.signal, cache: "no-store" });
        let status = "";
        try {
          status = (await res.json()).status;
        } catch {
          /* bỏ qua */
        }
        if (!alive) return;
        setState(res.ok && status === "ok" ? "ok" : res.status === 503 || status === "loading" ? "loading" : "off");
      } catch {
        if (alive) setState("off");
      } finally {
        clearTimeout(t);
      }
    };
    void check();
    const id = setInterval(check, 30000);
    return () => {
      alive = false;
      ac?.abort();
      clearInterval(id);
    };
  }, []);
  return state;
}

function Icon({ state }: { state: State }) {
  const p = { width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (state === "ok")
    return (
      <svg {...p}>
        <circle cx="8" cy="8" r="6.5" />
        <path d="M5 8.3l2 2 4-4.3" />
      </svg>
    );
  if (state === "off")
    return (
      <svg {...p}>
        <circle cx="8" cy="8" r="6.5" />
        <path d="M5.5 8h5" />
      </svg>
    );
  return (
    <svg {...p}>
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 4.5V8l2.3 1.5" />
    </svg>
  );
}

export default function ServerStatus({ state }: { state: State }) {
  return (
    <span className={`status ${state}`} role="status" aria-live="polite">
      <Icon state={state} />
      {TEXT[state]}
    </span>
  );
}
