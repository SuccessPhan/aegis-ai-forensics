"""Image validation, SHA-256 hashing, SHA-256 deduplication."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image

from src.core.contracts import ImageRecord
from src.core.logger import get_logger

log = get_logger(__name__)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_image(path: str | Path, min_size: int = 1) -> Tuple[bool, str]:
    """Return (ok, reason). Checks the file decodes fully and both sides >= min_size."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() invalidates the handle; reopen to fully decode
            im.load()
            w, h = im.size
    except Exception as e:  # noqa: BLE001 - any decode failure means invalid
        return False, f"unreadable: {type(e).__name__}"
    if min(w, h) < min_size:
        return False, f"too_small: {w}x{h} < {min_size}"
    return True, "ok"


def dedup_paths(paths: Sequence[str | Path]) -> Tuple[List[Path], List[Tuple[Path, Path]]]:
    """Dedup raw files by SHA-256. Returns (kept, [(dropped, duplicate_of)]). Deterministic (sorted)."""
    seen: dict[str, Path] = {}
    kept: List[Path] = []
    dropped: List[Tuple[Path, Path]] = []
    for p in sorted(Path(x) for x in paths):
        s = sha256_file(p)
        if s in seen:
            dropped.append((p, seen[s]))
        else:
            seen[s] = p
            kept.append(p)
    if dropped:
        log.info("dedup: dropped %d duplicate file(s)", len(dropped))
    return kept, dropped


def dedup_records(records: Iterable[ImageRecord]) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Dedup ImageRecords by sha256 (first occurrence wins)."""
    seen: set[str] = set()
    kept: List[ImageRecord] = []
    dropped: List[ImageRecord] = []
    for r in records:
        if r.sha256 in seen:
            dropped.append(r)
        else:
            seen.add(r.sha256)
            kept.append(r)
    return kept, dropped
