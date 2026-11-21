# AEGIS: Adaptive Multi-Expert AI-Generated Image Forensics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11%20(CUDA%2012.8)-ee4c2c.svg)](https://pytorch.org/)
[![Tests: 128 passing](https://img.shields.io/badge/tests-128%20passing-success.svg)](#running-the-tests)
[![Status: Research prototype](https://img.shields.io/badge/Status-Research%20prototype-orange.svg)](#limitations)

> *"An intelligent detection system should not only recognize what it knows, but also understand what it does not know, decide when it needs to adapt, and verify whether its adaptation actually improves reliability."*

**AEGIS** is a student research project on detecting AI-generated images. Instead of training a bigger detector, it builds a **decision layer** on top of three frozen experts with different blind spots, and routes trust between them image by image:

- a **profiler** measures how degraded an image is (JPEG, blur, sharpness) and how uncertain each expert is;
- a small **gating network** turns that profile into per-image expert weights;
- a **few-shot adaptation** step adapts to a new generator only when enough labelled samples exist;
- every update must beat the current model on a held-out split (**Champion–Challenger**) or is rolled back.

The full write-up, including every experiment round, what failed and how it was fixed, is in **[`reports/AEGIS_BaoCao_DoAn.docx`](reports/AEGIS_BaoCao_DoAn.docx)** (Vietnamese, 18 pages, 9 figures).

---

## Table of Contents

- [Key Results](#key-results)
- [How It Works](#how-it-works)
- [Data and Shortcut Control](#data-and-shortcut-control)
- [Experiment History](#experiment-history)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Web Demo](#web-demo)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [References](#references)
- [License](#license)

---

## Key Results

Main experiment (v3): 16,000 GenImage images, balanced 50/50 Real/AI, test split run **once**. ΔAUROC is AEGIS minus the baseline with a 95% paired bootstrap confidence interval (B = 2000, resampled by source image); `*` means the interval excludes 0.

| Scenario | Expert heads trained on clean images | Expert heads trained with blur/JPEG augmentation |
|---|---|---|
| S1 clean | −0.002\* vs Static LR | tie |
| S2 JPEG (Q 85/65/40) | **−0.009\*** vs Static LR | tie |
| S3 blur / resize | **+0.013\*** vs Static LR | tie |
| S4 Midjourney, N = 20 few-shot | **+0.014\*** vs Static LR refit on the same support set | **+0.014\*** |
| S4 Midjourney, N = 100 | **+0.009\*** | **+0.010\*** |
| S4 VQDM | ≈ 0 (every method ≈ 0.998 AUROC) | ≈ 0 |

What the results support:

1. **The experts really have complementary blind spots.** The frequency expert drops from 0.98 to 0.58 AUROC under JPEG Q40; CLIP drops from 0.99 to 0.66 under heavy blur, while DINOv2 stays at 0.78.
2. **The gating network routes in the right direction**: under blur it moves weight from CLIP to DINOv2; under JPEG it lowers the frequency expert's weight. This pattern repeats in every experiment round.
3. **Two gains replicate** across a different dataset and split (v1 → v3): blur/resize with clean heads, and few-shot adaptation on Midjourney at N = 20.
4. **Routing is already close to its ceiling.** An oracle that knows the degradation level and picks the best fixed weights for it beats AEGIS by at most ~0.01 AUROC, so the remaining limit is expert quality, not routing.

What they do not support: AEGIS does **not** beat a static learned weighting on clean or JPEG images, and a single CLIP expert is very strong on Midjourney.

---

## How It Works

```text
                          Input image
                               │  resize shorter side to 256, centre-crop 224×224
               ┌───────────────┴────────────────┐
               ▼                                ▼
     Profiler                           Expert pool (frozen backbones)
     • blind JPEG estimate              • CLIP ViT-L/14   – semantics
     • sharpness, high-freq energy      • DINOv2 ViT-B/14 – local structure
     • per-expert entropy & OOD         • SRM high-pass CNN, 1.17 M params – frequency
               └───────────────┬────────────────┘
                               ▼
              Gating MLP 12 → 32 → 3,  w = 0.05 + 0.85·softmax(z)
              P = Σ wᵢ·Pᵢ  (temperature-calibrated)
                               │
            Few-shot: N<10 keep │ 10–50 anchored blend │ >50 refit
            Champion–Challenger on val_gating_verify, rollback on regression
                               ▼
          Label + confidence + yellow flag (expert conflict / borderline / OOD)
```

Only the expert heads, the 1.17 M-parameter frequency CNN and the gating MLP are trained; all embeddings are cached, so calibration and gating run in seconds on a laptop (RTX 4060, 8 GB).

---

## Data and Shortcut Control

- **Source:** a seeded subset of the GenImage test split ([arXiv:2306.08571](https://arxiv.org/abs/2306.08571)): 2,000 AI + 2,000 Real (ImageNet) images per generator. **Seen:** ADM, Stable Diffusion v1.4. **Unseen (S4 only):** Midjourney, VQDM. AEGIS re-splits the images itself; it does not use GenImage's own train/val split.
- **GenImage leaks labels:** every AI image is PNG and every Real image is JPEG, and each generator has a fixed resolution. A logistic regression on three image-quality features alone (the *shortcut audit*) reached **0.706 AUROC** with plain cropping. JPEG-history harmonisation plus resize-then-crop brought it down to **0.63**; the remaining leak comes from resampling history (ADM is 256 px native). Instead of hiding it, every results table includes this `quality_only` model as a **floor**.
- **No leakage:** SHA-256 deduplication, group-aware splits (all degraded variants of an image share its split), 50/50 balance per split, and `test_benchmark` is never used for selection.

---

## Experiment History

| Round | Finding | Fix / decision |
|---|---|---|
| Audit | Format and size leak labels (AUROC 0.706) | JPEG harmonisation + `resize_crop` (0.63); `quality_only` floor in every table |
| Balance | Real outnumbered AI 1.8 : 1 | Real allocated per split to match AI |
| SRM bug | Frequency expert scored 0.51 on its own training data | Stale BatchNorm statistics; re-estimated after training (0.988 on val) |
| v1 | First real results; GLIDE separable by image quality alone (0.979) | Replace GLIDE with VQDM |
| v2 | Bootstrap CIs, weight plots, oracle ceiling ≤ 0.011 | Two of our own hypotheses rejected; no further routing tweaks |
| Fusion | Logit-space fusion tried | Chosen on validation only: no gain, keep probability fusion |
| v3 | Main results above | Reported with limitations |

Every round's data and results are kept (`reports/v1`, `reports/v2`, `reports/v3`) and each is tied to a git commit.

---

## Project Structure

```text
aegis-ai-forensics/
├── CLAUDE.md                # project rules, architecture, multi-agent workflow
├── configs/default.yaml     # every hyperparameter and path
├── docs/
│   ├── PROJECT_BRIEF.md     # research brief + decision log (§5–§10)
│   ├── WEB_API_CONTRACT.md  # contract between server and web frontend
│   └── research/            # per-paper analyses and METHOD_SPECS.md
├── src/
│   ├── core/                # data contracts, config, logger
│   ├── data/                # ingest, normalisation, quality features, splits, shortcut audit
│   ├── models/              # experts, feature cache, heads, bundles
│   ├── ood/                 # entropy and cosine-distance profiler
│   ├── learning/            # calibration, gating, few-shot, flags, export
│   ├── evaluation/          # metrics, ablation, significance, weight analysis
│   └── serving/             # FastAPI inference server for the web demo
├── web/                     # Next.js frontend (Vercel)
├── scripts/                 # data download, toy dataset, PDF→Markdown, server launcher
├── reports/                 # experiment results, figures, the Word report
├── tests/                   # 128 pytest tests
└── data/                    # images, caches, models (not in git)
```

---

## Getting Started

### 1. Environment

```bash
python -m venv .venv
.venv/Scripts/activate                       # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Developed on Windows 11, Python 3.14, PyTorch 2.11 + CUDA 12.8, RTX 4060 Laptop (8 GB).

### 2. Running the tests

```bash
python -m pytest tests/ -q        # 128 tests on a synthetic toy dataset, no downloads
```

### 3. Data

```bash
python scripts/fetch_genimage.py --out data/raw/genimage --per-class 2000   # ~5 GB, resumable
python -m src.data.ingest --mode resize_crop --out data/normalized_v2       # normalise, split, audit
```

The archive is read with HTTP range requests, so only the selected images are downloaded. GenImage is licensed CC BY-NC-SA 4.0 (non-commercial use only).

### 4. Pipeline

```bash
M=data/normalized_v2/manifest.json
python -m src.models.extract --manifest $M --expert clip          # GPU, ~25 min
python -m src.models.extract --manifest $M --expert dinov2
python -m src.models.heads   --manifest $M --expert clip   --variant aug   # also: dinov2, freq; clean/aug
python -m src.models.heads   --manifest $M --expert freq   --variant aug --select-freq
python -m src.models.bundle  --manifest $M --split val_gating_fit --head-variant aug   # each split
python -m src.evaluation.select_fusion                             # validation only
python -m src.evaluation.ablation --s4-seeds 0 1 2 3 4 --out reports/v3 --save-predictions
python -m src.evaluation.significance --pred reports/v3/predictions --out reports/v3 --manifest $M
python -m src.evaluation.weights_analysis --pred reports/v3/predictions --out reports/v3 --manifest $M
```

The report figures are regenerated from these outputs with `python reports/report_build/make_figures.py` and `node reports/report_build/build_docx.js`.

---

## Web Demo

A small demo lets people upload an image and see AEGIS's verdict, the expert weights and the yellow flag.

```text
Browser ──► Vercel (web/, Next.js) ──► Cloudflare Tunnel ──► laptop: FastAPI + GPU (src/serving/)
```

- **No storage:** images are processed in memory and discarded; logs contain only status and latency.
- **Run the server:** `python -m src.learning.export --head-variant aug` once, then
  `.\scripts\serve.ps1 -Origins "https://<your-app>.vercel.app" -RateLimit 300 -MaxWaiting 64`
- **Expose it:** `cloudflared tunnel --url http://localhost:8000`
- **Frontend:** import the repo on Vercel with Root Directory `web` and set `NEXT_PUBLIC_AEGIS_API_URL` to the tunnel URL. See [`web/README.md`](web/README.md).

API details: [`docs/WEB_API_CONTRACT.md`](docs/WEB_API_CONTRACT.md).

---

## Limitations

- **Real-world images are out of distribution.** Through the web API, 40 raw Midjourney PNGs were classified correctly only **40%** of the time (Real ImageNet JPEGs: 97%; raw SD v1.4 PNGs: 95%). One likely cause: during training every AI image carried one JPEG round (harmonisation), while raw PNGs from new tools never did. Treat the demo's verdicts as indicative only.
- Design decisions in later rounds were made after seeing earlier test results; v3 uses a fresh split but is not fully blind.
- A residual quality shortcut remains for ADM (0.75 audit AUROC).
- Champion–Challenger and rollback are unit-tested but not yet demonstrated on real data.
- Scale: 4 generators, 2,000 images per class each.

---

## Documentation

- [`reports/AEGIS_BaoCao_DoAn.docx`](reports/AEGIS_BaoCao_DoAn.docx): full project report (Vietnamese).
- [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md): research brief and dated decision log.
- [`docs/research/METHOD_SPECS.md`](docs/research/METHOD_SPECS.md): literature review mapped onto AEGIS.
- [`CLAUDE.md`](CLAUDE.md): project rules and multi-agent development workflow.

## References

1. GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image. [arXiv:2306.08571](https://arxiv.org/abs/2306.08571)
2. Towards Universal Fake Image Detectors that Generalize Across Generative Models (UnivFD). [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)
3. DINOv2: Learning Robust Visual Features without Supervision. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
4. Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues (F3Net). [arXiv:2007.09355](https://arxiv.org/abs/2007.09355)
5. Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning (FreqNet). [arXiv:2403.07240](https://arxiv.org/abs/2403.07240)
6. On Calibration of Modern Neural Networks. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
7. Energy-based Out-of-distribution Detection. [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)
8. Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. [arXiv:2403.17608](https://arxiv.org/abs/2403.17608)

## License

Code: MIT, see [LICENSE](LICENSE). The GenImage images, ImageNet images and pretrained CLIP/DINOv2 weights are **not** included and keep their own licences.
