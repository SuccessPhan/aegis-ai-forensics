"use client";

import { useEffect, useState } from "react";
import { API_URL } from "./api";

type State = "checking" | "ok" | "loading" | "off";

const TEXT: Record<State, string> = {
  checking: "đang kết nối",
  ok: "sẵn sàng",
  loading: "đang khởi động",
  off: "máy chủ tắt",
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

export default function ServerStatus({ state }: { state: State }) {
  return (
    <span className={`status ${state}`} role="status" aria-live="polite">
      <span className="dot" aria-hidden="true" />
      {TEXT[state]}
    </span>
  );
}
