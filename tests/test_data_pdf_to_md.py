import importlib.util
from pathlib import Path

import pymupdf

SPEC = importlib.util.spec_from_file_location("pdf_to_md", Path(__file__).resolve().parents[1] / "scripts" / "pdf_to_md.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_empty_or_missing_dir_is_ok(tmp_path):
    (tmp_path / "pdfs").mkdir()
    assert mod.convert_all(tmp_path / "pdfs", tmp_path / "md") == 0
    assert mod.convert_all(tmp_path / "missing", tmp_path / "md") == 0


def test_convert_and_skip_existing(tmp_path):
    (tmp_path / "pdfs").mkdir()
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello aegis")
    doc.save(tmp_path / "pdfs" / "p.pdf")
    doc.close()
    assert mod.convert_all(tmp_path / "pdfs", tmp_path / "md") == 1
    assert "hello aegis" in (tmp_path / "md" / "p.md").read_text(encoding="utf-8")
    assert mod.convert_all(tmp_path / "pdfs", tmp_path / "md") == 0
