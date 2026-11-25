"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ServerStatus, { useServerState } from "./ServerStatus";
import { API_URL } from "./api";

const LINKS = [
  { href: "/", text: "Kiểm tra ảnh" },
  { href: "/lab", text: "Bộ điều phối" },
  { href: "/so-lieu", text: "Số liệu" },
];

export default function Nav() {
  const path = usePathname();
  const server = useServerState();
  return (
    <header className="top">
      <div className="wrap top-in">
        <Link href="/" className="logo" aria-label="AEGIS – trang chủ">
          AEGIS
        </Link>
        <nav className="nav" aria-label="Các trang">
          {LINKS.map((l) => {
            const on = l.href === "/" ? path === "/" : path?.startsWith(l.href);
            return (
              <Link key={l.href} href={l.href} className={on ? "on" : undefined} aria-current={on ? "page" : undefined}>
                {l.text}
              </Link>
            );
          })}
        </nav>
        {API_URL && <ServerStatus state={server} />}
      </div>
    </header>
  );
}
