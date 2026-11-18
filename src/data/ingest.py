"""GenImage ingestion (project-normalised layout) and the full data-prep CLI.

Layout under raw_root (generator dir names must match cfg data.generators seen+unseen):
    <generator>/ai/      AI images of that generator   -> label FAKE, generator = dir name
    <generator>/nature/  paired ImageNet Real images   -> label REAL, generator = cfg data.real_name
Unknown generator directories (e.g. biggan) are skipped with a warning. Extensions are matched
case-insensitively (.png .jpg .jpeg .JPEG ...). Identical Real images appearing under several
generators' nature/ dirs are dropped by the SHA-256 dedup in prepare_records (count is logged).
Note: with max_per_class, the subset is drawn BEFORE dedup, so a duplicated Real may leave a
generator's Real count marginally below max_per_class.

CLI:  python -m src.data.ingest [--root data/raw/genimage] [--max-per-class N] [--out data/normalized] [--mode crop|resize_crop]
Exit codes: 0 ok, 1 audit failed (only with --strict), 2 bad input layout.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.core.config import load_config
from src.core.contracts import LABEL_FAKE, LABEL_REAL
from src.core.logger import get_logger
from src.data.degradations import materialize_variants
from src.data.pipeline import RawItem, prepare_records
from src.data.shortcut_audit import audit_full
from src.data.split import assign_splits, split_label_table

log = get_logger(__name__)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LAYOUT_HINT = ("Expected layout:\n  <root>/<generator>/ai/*.png\n  <root>/<generator>/nature/*.JPEG\n"
               "with <generator> in {gens} (e.g. data/raw/genimage/adm/ai and data/raw/genimage/adm/nature).")


class IngestError(RuntimeError):
    """User-facing data-layout error (message is meant to be shown as-is)."""


def _list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def scan_genimage(root: str | Path, cfg: dict, max_per_class: Optional[int] = None) -> List[RawItem]:
    """Scan the normalised GenImage tree into RawItems (optionally a seeded, per-generator balanced subset)."""
    root = Path(root)
    gens = list(cfg["data"]["generators"]["seen"]) + list(cfg["data"]["generators"]["unseen"])
    hint = LAYOUT_HINT.format(gens=gens)
    if not root.is_dir():
        raise IngestError(f"raw root not found: {root}\n{hint}")
    real_name = cfg["data"]["real_name"]
    seed = int(cfg.get("seed", 0))
    known = {g.lower(): g for g in gens}

    items: List[RawItem] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        gen = known.get(d.name.lower())
        if gen is None:
            log.warning("skipping unknown generator directory %s (not in cfg seen+unseen)", d)
            continue
        ai = _list_images(d / "ai")
        nat = _list_images(d / "nature")
        if not ai or not nat:
            log.warning("%s: ai=%d nature=%d images (both expected)", gen, len(ai), len(nat))
        if max_per_class is not None:
            k = min(int(max_per_class), len(ai), len(nat))
            rng = np.random.default_rng([seed, zlib.crc32(gen.encode())])
            ai = [ai[i] for i in sorted(rng.permutation(len(ai))[:k])]
            nat = [nat[i] for i in sorted(rng.permutation(len(nat))[:k])]
        items += [RawItem(p, gen, LABEL_FAKE) for p in ai] + [RawItem(p, real_name, LABEL_REAL) for p in nat]
        log.info("scanned %s: %d ai, %d nature", gen, len(ai), len(nat))
    present = {i.generator for i in items}
    missing = [g for g in gens if g not in present]
    if missing:
        log.warning("configured generators with no images: %s", missing)
    if not items:
        raise IngestError(f"no images found under {root}\n{hint}")
    return items


def _stats_table(records) -> dict:
    c = Counter((str(r.split), r.generator, "REAL" if r.label == LABEL_REAL else "FAKE")
                for r in records if r.is_clean)
    return {f"{s}|{g}|{l}": n for (s, g, l), n in sorted(c.items())}


def _format_table(table: dict) -> str:
    lines = [f"{'split':<18}{'real':>8}{'ai':>8}"]
    for s, d in table.items():
        lines.append(f"{s:<18}{d['real']:>8}{d['ai']:>8}")
    return "\n".join(lines)


def run_ingest(root: str | Path, out: str | Path, cfg: dict, max_per_class: Optional[int] = None) -> dict:
    """scan -> prepare_records -> assign_splits -> materialize_variants -> save manifest -> shortcut audit."""
    out = Path(out)
    mode = str(cfg["data"]["normalize"].get("mode", "crop"))
    prev = out / "manifest.json"
    if prev.exists():   # two modes must never overwrite each other
        try:
            prev_mode = json.loads(prev.read_text(encoding="utf-8")).get("meta", {}).get("normalize_mode")
        except (OSError, ValueError):
            prev_mode = None
        if prev_mode is not None and prev_mode != mode:
            raise IngestError(f"{out} already holds a '{prev_mode}' normalisation; use a different --out for mode '{mode}'")
    items = scan_genimage(root, cfg, max_per_class)
    records = prepare_records(items, out, cfg)
    log.info("ingest: %d raw items -> %d records (%d dropped as invalid/duplicate, incl. cross-generator Real dups)",
             len(items), len(records), len(items) - len(records))
    man = assign_splits(records, cfg, root=str(out))
    materialize_variants(man, out, cfg)
    man.meta["source"] = "genimage"
    man.meta["normalize_mode"] = mode
    man.meta["resize_short_side"] = int(cfg["data"]["normalize"].get("resize_short_side", 256)) if mode == "resize_crop" else None
    man.save(out / "manifest.json")
    stats = _stats_table(man.records)
    for k, n in stats.items():
        log.info("stats split|generator|label = %s : %d", k, n)
    try:
        audit = audit_full(man, cfg)   # clean seen-generator images (S1): pooled + per generator
    except ValueError as e:
        audit = {"status": "skipped", "reason": str(e), "passed": None}
    log.info("shortcut audit (%s): %s", audit.get("status"), json.dumps(audit))
    table = split_label_table(man)
    log.info("final split x label table (clean images)\n%s", _format_table(table))
    audit["mode"] = mode
    summary = {"mode": mode, "n_items": len(items), "n_records": len(man.records), "stats": stats, "split_label_table": table,
               "shortcut_audit": audit}
    with open(out / "shortcut_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AEGIS GenImage ingestion: scan, normalise, split, audit")
    ap.add_argument("--root", default=None, help="raw GenImage root (default: cfg paths.raw_root)")
    ap.add_argument("--max-per-class", type=int, default=None, help="seeded subset per generator and class")
    ap.add_argument("--out", default=None, help="output dir (default: cfg paths.normalized_dir)")
    ap.add_argument("--mode", choices=["crop", "resize_crop"], default=None,
                    help="normalisation mode (overrides cfg data.normalize.mode). Without --out the output dir is "
                         "<normalized_dir>_<mode>, so two modes never overwrite each other")
    ap.add_argument("--config", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 if the shortcut audit fails")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = args.root or cfg["paths"]["raw_root"]
    if args.mode:
        cfg["data"]["normalize"]["mode"] = args.mode
    out = args.out or (f"{cfg['paths']['normalized_dir']}_{args.mode}" if args.mode else cfg["paths"]["normalized_dir"])
    try:
        summary = run_ingest(root, out, cfg, args.max_per_class)
    except IngestError as e:
        log.error("%s", e)
        return 2
    return 1 if args.strict and summary["shortcut_audit"].get("passed") is False else 0


if __name__ == "__main__":
    sys.exit(main())
