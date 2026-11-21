# AEGIS Web Demo — API Contract (v1)

Shared contract between the inference server (`src/serving/`, runs on the team laptop) and the web frontend (`web/`, deployed on Vercel). Both sides must follow it exactly.

## Decisions (user, 2026-11-21)
- **No storage.** Uploaded images are processed in memory and discarded. The server never writes images, filenames, or per-image results to disk or logs. Logs contain only status code, latency and an anonymous request counter.
- **Inference host:** the team laptop (RTX 4060) exposed through a Cloudflare Tunnel. Vercel hosts only the frontend.
- **Audience:** ~150 users, deadline 1 week. Users may upload any real or AI image, edited with any tool.
- **Model:** head variant `aug`, Expert 3 = SRM, fusion `prob`, cache version `v2` (the v3 configuration in `docs/PROJECT_BRIEF.md` §10).

## Endpoints

### `GET /api/health`
```json
{ "status": "ok", "model_version": "aegis-v3-aug-srm", "device": "cuda", "queue": 0 }
```
`status` is `"loading"` while models load (HTTP 503 for `/api/predict` meanwhile).

### `POST /api/predict`
Request: `multipart/form-data` with one field `file` (JPEG, PNG or WebP; ≤ 10 MB; shorter side ≥ 224 px).

Response 200:
```json
{
  "label": "AI",
  "p_ai": 0.87,
  "confidence": 0.87,
  "warning_flag": false,
  "flag_reasons": [],
  "experts": {
    "clip":   { "p_ai": 0.91, "weight": 0.55 },
    "dinov2": { "p_ai": 0.72, "weight": 0.25 },
    "freq":   { "p_ai": 0.95, "weight": 0.20 }
  },
  "quality": { "jpeg_quality_est": 96.1, "log_laplacian_var": 7.2, "high_freq_energy_ratio": 0.031 },
  "ood_score": 0.21,
  "ood_threshold": 0.34,
  "model_version": "aegis-v3-aug-srm",
  "latency_ms": 640
}
```
- `label` = `"AI"` if `p_ai >= 0.5` else `"Real"`; `confidence = max(p_ai, 1 - p_ai)`.
- `flag_reasons` ⊆ `["conflict", "uncertain", "ood"]` (contract constants `FLAG_CONFLICT`, `FLAG_UNCERTAIN`, `FLAG_OOD`).
- Weights sum to 1; each in [0.05, 0.90].

Errors (JSON `{ "error": "<code>", "message": "<Vietnamese, user-facing>" }`):

| HTTP | `error` | When |
|---|---|---|
| 400 | `bad_image` | Not decodable, unsupported format (e.g. HEIC), or animated without a first frame |
| 400 | `too_small` | Shorter side < 224 px |
| 413 | `too_large` | File > 10 MB |
| 429 | `rate_limited` | More than 20 requests / minute from one client IP |
| 503 | `not_ready` | Models still loading or queue full |

## Preprocessing (server side, must match training)
EXIF-orientation fix → RGB (first frame if animated) → `resize_crop` exactly as `src/data/normalize.py` (shorter side → 256 with LANCZOS, center crop 224, offsets rounded to multiples of 8). **No `jpeg_harmonize`** (that step only equalises the training dataset). The same 224×224 array feeds all three experts and the quality features.

## CORS
Allowed origins from env `AEGIS_ALLOWED_ORIGINS` (comma-separated; e.g. the Vercel URL and `http://localhost:3000`).
