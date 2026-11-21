"""Download a seeded random subset of the GenImage *test* split without downloading the whole 25 GB archive.

Source: genimage_test.zip on the community Hugging Face mirror `jzousz/GenImage` (the official release is on
Baidu Yunpan, see https://github.com/GenImage-Dataset/GenImage). Licence: CC BY-NC-SA 4.0, non-commercial only.

The archive's central directory is read through HTTP range requests, then only the selected members are fetched,
CRC-checked and written to:

    <out>/<generator>/{ai,nature}/<original file name>

which is the layout `python -m src.data.ingest` expects. Re-running resumes (complete files are skipped).

Usage (from the repo root):
    .venv/Scripts/python scripts/fetch_genimage.py --out data/raw/genimage --per-class 2000
    .venv/Scripts/python scripts/fetch_genimage.py --out data/raw/genimage --per-class 2000 --generators adm sd_v1_4
Use few workers (default 4): the mirror answers HTTP 429 when hit too hard; just re-run to fill gaps.
"""
from __future__ import annotations

import argparse
import io
import random
import struct
import time
import urllib.request
import zipfile
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # same convention as the other scripts
from src.core.logger import get_logger  # noqa: E402

log = get_logger("fetch_genimage")
URL = "https://huggingface.co/datasets/jzousz/GenImage/resolve/main/genimage_test.zip"
FOLDERS = {"adm": "adm_imagenet", "sd_v1_4": "sdv4_imagenet", "midjourney": "midjourney_imagenet",
           "glide": "glide_imagenet", "vqdm": "vqdm_imagenet", "biggan": "biggan_imagenet",
           "wukong": "wukong_imagenet", "sd_v1_5": "sdv5_imagenet"}
DEFAULT_GENERATORS = ["adm", "sd_v1_4", "midjourney", "vqdm"]   # configs/default.yaml seen + unseen


class HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile to read the central directory)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else n
        if n == 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req).read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def fetch_member(info: zipfile.ZipInfo, dst: Path, tries: int = 5) -> int:
    """Download one archive member with a single range request; return bytes written, 0 if present, -1 on failure."""
    if dst.exists() and dst.stat().st_size == info.file_size:
        return 0
    err = None
    for t in range(tries):
        try:
            end = info.header_offset + 30 + len(info.filename.encode()) + 2048 + info.compress_size
            req = urllib.request.Request(URL, headers={"Range": f"bytes={info.header_offset}-{end}"})
            blob = urllib.request.urlopen(req, timeout=120).read()
            n, m = struct.unpack("<HH", blob[26:30])
            raw = blob[30 + n + m:30 + n + m + info.compress_size]
            data = raw if info.compress_type == 0 else zlib.decompress(raw, -15)
            if zlib.crc32(data) != info.CRC:
                raise IOError("crc mismatch")
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dst)
            return len(data)
        except Exception as e:  # network errors, HTTP 429
            err = e
            time.sleep(2 * (t + 1))
    log.warning("failed %s: %s", info.filename, err)
    return -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/genimage")
    ap.add_argument("--per-class", type=int, default=2000, help="images per class (ai / nature) per generator")
    ap.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS, choices=sorted(FOLDERS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    zf = zipfile.ZipFile(io.BufferedReader(HTTPFile(URL), buffer_size=1 << 20))
    wanted = {FOLDERS[g]: g for g in a.generators}
    groups: dict = {}
    for info in zf.infolist():
        parts = info.filename.split("/")
        if len(parts) == 4 and parts[1] in wanted and parts[2] in ("ai", "nature") and not info.is_dir():
            groups.setdefault((wanted[parts[1]], parts[2]), []).append(info)

    jobs = []
    for (gen, cls), items in sorted(groups.items()):
        items.sort(key=lambda x: x.filename)
        pick = random.Random(f"aegis-42-{gen}-{cls}").sample(items, min(a.per_class, len(items)))
        d = Path(a.out) / gen / cls
        d.mkdir(parents=True, exist_ok=True)
        jobs += [(i, d / i.filename.split("/")[-1]) for i in pick]
    log.info("%d files planned for %s", len(jobs), a.generators)

    done = failed = total = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda j: fetch_member(*j), jobs):
            done += 1
            failed += r < 0
            total += max(r, 0)
            if done % 500 == 0:
                log.info("%d/%d files, %.2f GB new, %d failed", done, len(jobs), total / 1e9, failed)
    log.info("done: %d files, %d failed, %.2f GB new%s", done, failed, total / 1e9,
             " - re-run to retry the failed files" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
