"""FastAPI server for the AEGIS web demo (docs/WEB_API_CONTRACT.md).

Privacy: uploads are parsed straight from the request stream into memory (no spooled temp file), processed and dropped.
Logs hold only status code, latency and an anonymous counter - never file names, image data or per-image results.

Run: python -m src.serving.app --host 127.0.0.1 --port 8000
Env: AEGIS_ALLOWED_ORIGINS (comma-separated CORS origins).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import threading
from contextlib import asynccontextmanager
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.logger import get_logger
from src.serving.engine import DEFAULT_MODEL_VERSION, Engine, ImageError

log = get_logger(__name__)

MAX_BYTES = 10 * 1024 * 1024          # file size limit
BODY_SLACK = 64 * 1024                # multipart framing overhead allowed on top of MAX_BYTES
RATE_LIMIT = 20                       # requests ...
RATE_WINDOW = 60.0                    # ... per this many seconds, per client IP
MAX_WAITING = 8                       # requests allowed to wait for the GPU

MESSAGES = {
    "bad_image": "Không đọc được ảnh. Vui lòng tải lên tệp JPEG, PNG hoặc WebP hợp lệ.",
    "too_small": "Ảnh quá nhỏ. Cạnh ngắn của ảnh phải từ 224 px trở lên.",
    "too_large": "Tệp quá lớn. Dung lượng tối đa là 10 MB.",
    "rate_limited": "Bạn gửi quá nhiều yêu cầu. Vui lòng thử lại sau ít phút.",
    "not_ready": "Hệ thống đang khởi động hoặc quá tải. Vui lòng thử lại sau giây lát.",
    "internal": "Đã xảy ra lỗi khi phân tích ảnh. Vui lòng thử lại.",
}
STATUS = {"bad_image": 400, "too_small": 400, "too_large": 413, "rate_limited": 429, "not_ready": 503,
          "internal": 500}


class _TooLarge(Exception):
    pass


def err(code: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": MESSAGES[code]}, status_code=STATUS[code])


class _FileExtractor:
    """Streaming multipart parser that keeps only the bytes of the form field `file` (in memory)."""

    def __init__(self, content_type: str) -> None:
        ctype, opts = parse_options_header(content_type)
        boundary = opts.get(b"boundary")
        if ctype != b"multipart/form-data" or not boundary:
            raise ValueError("not multipart/form-data")
        self.data = bytearray()
        self.found = False
        self._hdr: Dict[bytes, bytes] = {}
        self._field = b""
        self._value = b""
        self._is_file = False
        self._parser = MultipartParser(boundary, {
            "on_part_begin": self._part_begin, "on_header_field": self._hfield, "on_header_value": self._hvalue,
            "on_header_end": self._hend, "on_headers_finished": self._hfinished, "on_part_data": self._pdata})

    def _part_begin(self) -> None:
        self._hdr, self._field, self._value, self._is_file = {}, b"", b"", False

    def _hfield(self, d: bytes, s: int, e: int) -> None:
        self._field += d[s:e]

    def _hvalue(self, d: bytes, s: int, e: int) -> None:
        self._value += d[s:e]

    def _hend(self) -> None:
        self._hdr[self._field.lower()] = self._value
        self._field, self._value = b"", b""

    def _hfinished(self) -> None:
        _, opts = parse_options_header(self._hdr.get(b"content-disposition", b""))
        self._is_file = opts.get(b"name") == b"file" and not self.found
        self.found = self.found or self._is_file

    def _pdata(self, d: bytes, s: int, e: int) -> None:
        if self._is_file:
            self.data += d[s:e]
            if len(self.data) > MAX_BYTES:
                raise _TooLarge()

    def feed(self, chunk: bytes) -> None:
        self._parser.write(chunk)


def _client_ip(request: Request) -> str:
    return (request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")).strip()


def create_app(engine: Optional[Engine] = None, engine_factory: Optional[Callable[[], Engine]] = None,
               rate_limit: int = RATE_LIMIT, max_waiting: int = MAX_WAITING) -> FastAPI:
    """`engine` = ready engine (tests); `engine_factory` = loaded in a background thread at startup."""
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if st.engine is None and engine_factory is not None:
            threading.Thread(target=_load, daemon=True, name="engine-loader").start()
        yield

    app = FastAPI(title="AEGIS web demo API", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    st = app.state
    st.engine, st.load_error = engine, None
    st.pending = 0                                   # requests waiting for / holding the GPU
    st.counter = itertools.count(1)                  # anonymous request counter
    st.hits = defaultdict(deque)                     # ip -> timestamps (in-memory only)
    st.gpu_lock = asyncio.Lock()

    origins = [o.strip() for o in os.environ.get("AEGIS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "OPTIONS"],
                           allow_headers=["*"], max_age=600)

    def _load() -> None:
        try:
            st.engine = engine_factory()
        except Exception as e:  # noqa: BLE001
            st.load_error = type(e).__name__
            log.error("engine load failed: %s: %s", type(e).__name__, e)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"error": "not_found" if exc.status_code == 404 else "http_error",
                             "message": "Không tìm thấy." if exc.status_code == 404 else "Yêu cầu không hợp lệ."},
                            status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _val_exc(_: Request, __: RequestValidationError) -> JSONResponse:
        return err("bad_image")

    @app.get("/api/health")
    async def health() -> JSONResponse:
        e = st.engine
        body = {"status": "ok" if e is not None else ("error" if st.load_error else "loading"),
                "model_version": e.model_version if e else DEFAULT_MODEL_VERSION,
                "device": e.device if e else "unknown", "queue": st.pending}
        return JSONResponse(body, status_code=200 if e is not None else 503)

    def _rate_limited(ip: str) -> bool:
        now = time.monotonic()
        q: Deque[float] = st.hits[ip]
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= rate_limit:
            return True
        q.append(now)
        if len(st.hits) > 5000:                      # opportunistic cleanup of idle IPs
            for k in [k for k, v in st.hits.items() if not v or now - v[-1] > RATE_WINDOW]:
                del st.hits[k]
        return False

    async def _read_file(request: Request) -> Optional[bytes]:
        """Returns file bytes, None if the `file` field is missing/invalid. Raises _TooLarge."""
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES + BODY_SLACK:
            raise _TooLarge()
        try:
            ex = _FileExtractor(request.headers.get("content-type", ""))
        except ValueError:
            return None
        total = 0
        try:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_BYTES + BODY_SLACK:
                    raise _TooLarge()
                ex.feed(chunk)
        except _TooLarge:
            raise
        except Exception:  # noqa: BLE001 - malformed multipart
            return None
        return bytes(ex.data) if ex.found else None

    @app.post("/api/predict")
    async def predict(request: Request) -> JSONResponse:
        t0 = time.perf_counter()
        n = next(st.counter)
        resp = await _predict(request)
        log.info("predict #%d status=%d latency_ms=%d", n, resp.status_code, int((time.perf_counter() - t0) * 1000))
        return resp

    async def _predict(request: Request) -> JSONResponse:
        if _rate_limited(_client_ip(request)):
            return err("rate_limited")
        eng: Optional[Engine] = st.engine
        if eng is None or st.pending >= max_waiting + 1:
            return err("not_ready")
        try:
            data = await _read_file(request)
        except _TooLarge:
            return err("too_large")
        if not data:
            return err("bad_image")
        st.pending += 1
        try:
            async with st.gpu_lock:
                result = await asyncio.get_running_loop().run_in_executor(None, eng.analyze, data)
            return JSONResponse(result)
        except ImageError as e:
            return err(e.code)
        except Exception as e:  # noqa: BLE001
            log.error("analysis failed: %s", type(e).__name__)
            return err("internal")
        finally:
            st.pending -= 1

    return app


def build_default_app() -> FastAPI:
    return create_app(engine_factory=Engine.load)


def main(argv: Optional[List[str]] = None) -> int:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    # A whole class behind one school NAT shares a single public IP, so the per-IP limit must be raisable.
    ap.add_argument("--rate-limit", type=int, default=RATE_LIMIT, help="requests per minute per client IP")
    ap.add_argument("--max-waiting", type=int, default=MAX_WAITING, help="requests allowed to queue for the GPU")
    a = ap.parse_args(argv)
    app = create_app(engine_factory=Engine.load, rate_limit=a.rate_limit, max_waiting=a.max_waiting)
    log.info("serving with rate_limit=%d/min per IP, max_waiting=%d", a.rate_limit, a.max_waiting)
    uvicorn.run(app, host=a.host, port=a.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
