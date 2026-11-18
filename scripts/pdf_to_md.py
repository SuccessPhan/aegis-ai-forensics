"""Convert docs/pdfs/*.pdf -> docs/raw_papers/*.md using pymupdf. Skips already-converted files.

Usage: python scripts/pdf_to_md.py [--src docs/pdfs] [--dst docs/raw_papers]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_config, resolve_path  # noqa: E402
from src.core.logger import get_logger  # noqa: E402

log = get_logger("pdf_to_md")


def convert_pdf(pdf: Path, out: Path) -> None:
    import pymupdf

    parts = [f"# {pdf.stem}\n\n> Auto-converted from `{pdf.name}` (pymupdf plain text; verify against the PDF).\n"]
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            parts.append(f"\n\n<!-- page {i} -->\n\n{page.get_text('text')}")
    out.write_text("".join(parts), encoding="utf-8")


def convert_all(src: Path, dst: Path) -> int:
    """Returns number of newly converted files."""
    if not src.is_dir():
        log.info("no PDF directory at %s; nothing to do", src)
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted(src.glob("*.pdf")):
        out = dst / f"{pdf.stem}.md"
        if out.exists():
            log.info("skip (exists): %s", out.name)
            continue
        try:
            convert_pdf(pdf, out)
            n += 1
            log.info("converted %s -> %s", pdf.name, out.name)
        except Exception as e:  # noqa: BLE001
            log.error("failed %s: %s", pdf.name, e)
    if n == 0:
        log.info("no new PDFs converted")
    return n


def main(argv: Optional[List[str]] = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(resolve_path(cfg, "pdf_dir")))
    ap.add_argument("--dst", default=str(resolve_path(cfg, "raw_papers_dir")))
    a = ap.parse_args(argv)
    convert_all(Path(a.src), Path(a.dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
