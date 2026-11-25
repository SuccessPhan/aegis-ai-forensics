"""Download a Hugging Face image dataset into the external-test-set folder layout (real/, <generator>/).

Low-memory by design: the dataset's parquet files are downloaded to disk one by one (resumable, cached by
huggingface_hub) and read a few rows at a time with pyarrow, so a laptop never holds a whole multi-hundred-MB file in
RAM. Image bytes are written as they are (no re-encoding: JPEG harmonisation happens later in src.data.external).

Works with datasets stored as parquet with an image column ({bytes, path}) and a label column (0/1 or
"real"/"fake"/"ai"), plus optionally a generator column. Check each dataset's licence and terms before use.

    pip install pyarrow huggingface_hub          # usually already installed (datasets / transformers)
    python scripts/fetch_external.py --dataset Robo531/ai-detector-benchmark-test-data --split train \\
        --out data/raw/external/newgen --generator-column generator

Then: python -m src.data.external --root data/raw/external/newgen --name newgen
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image as PILImage

REAL_VALUES = {"0", "real", "nature", "authentic", "human", "false"}


def _is_real(value) -> bool:
    return str(value).strip().lower() in REAL_VALUES


def parquet_files(dataset: str, split: str, config: Optional[str] = None) -> List[str]:
    """Repository paths of the parquet files of one split (HF layouts: data/<split>-*.parquet, <config>/<split>/...,
    and the auto-converted refs/convert/parquet branch)."""
    from huggingface_hub import HfApi
    api = HfApi()
    for revision in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(dataset, repo_type="dataset", revision=revision)
        except Exception:                               # noqa: BLE001 - branch may not exist
            continue
        out = []
        for f in files:
            if not f.endswith(".parquet"):
                continue
            parts = f.split("/")
            name = parts[-1]
            if config and config not in parts:
                continue
            if name.startswith(f"{split}-") or name.startswith(f"{split}.") or split in parts[:-1]:
                out.append(f)
        if out:
            return sorted(out)
    raise FileNotFoundError(f"no parquet files for split {split!r} in {dataset} (check --split / --config-name)")


def download(dataset: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    revision = None
    try:
        return Path(hf_hub_download(dataset, filename, repo_type="dataset"))
    except Exception:                                   # noqa: BLE001 - auto-converted parquet lives on a branch
        revision = "refs/convert/parquet"
    return Path(hf_hub_download(dataset, filename, repo_type="dataset", revision=revision))


def iter_rows(path: Path, columns: List[str]) -> Iterator[dict]:
    """Rows of a parquet file, one row group at a time (memory ~ one row group, freed before the next)."""
    import gc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path), buffer_size=8 << 20, pre_buffer=False, memory_map=False)
    cols = [c for c in columns if c in pf.schema_arrow.names]
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g, columns=cols, use_threads=False)
        for k in range(table.num_rows):
            yield {c: table.column(c)[k].as_py() for c in cols}
        del table
        gc.collect()


def image_bytes(value) -> Optional[bytes]:
    if isinstance(value, dict):
        if value.get("bytes"):
            return value["bytes"]
        if value.get("path") and Path(value["path"]).is_file():
            return Path(value["path"]).read_bytes()
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="Hugging Face dataset id")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config-name", default=None, help="dataset configuration (subset), if any")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--generator-column", default=None, help="column naming the generator of AI images")
    ap.add_argument("--fake-name", default="ai", help="folder name for AI images without a generator column")
    ap.add_argument("--max-per-generator", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        import pyarrow  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("this script needs pyarrow and huggingface_hub: pip install pyarrow huggingface_hub")
        return 1
    out = Path(a.out)
    counts: Counter = Counter()
    columns = [a.image_column, a.label_column] + ([a.generator_column] if a.generator_column else [])
    files = parquet_files(a.dataset, a.split, a.config_name)
    print(f"{len(files)} parquet file(s) for split '{a.split}'")
    i = 0
    for fname in files:
        print("downloading", fname, "...", flush=True)
        path = download(a.dataset, fname)
        for ex in iter_rows(path, columns):
            i += 1
            real = _is_real(ex.get(a.label_column))
            if real:
                gen = "real"                                # real photos keep one folder whatever their source
            else:
                gen = str(ex.get(a.generator_column) if a.generator_column else a.fake_name)
                gen = gen.strip().lower().replace(" ", "_").replace("/", "_") or a.fake_name
            if a.max_per_generator and counts[gen] >= a.max_per_generator:
                continue
            data = image_bytes(ex.get(a.image_column))
            if not data:
                continue
            try:
                with PILImage.open(io.BytesIO(data)) as im:
                    fmt = (im.format or "PNG").upper()
            except Exception:                           # noqa: BLE001 - unreadable image: skip
                continue
            ext = {"JPEG": ".jpg", "JPG": ".jpg", "WEBP": ".webp", "PNG": ".png"}.get(fmt, ".png")
            d = out / gen
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{i:06d}{ext}").write_bytes(data)     # original bytes: harmonisation happens later
            counts[gen] += 1
        print("  so far:", dict(counts), flush=True)
    print("saved:", dict(counts), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
