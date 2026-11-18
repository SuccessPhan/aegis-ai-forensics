# AEGIS: Adaptive Multi-Expert AI-Generated Image Forensics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status: Early Development](https://img.shields.io/badge/Status-Early%20Development-orange.svg)](#roadmap)

> *"An intelligent detection system should not only recognize what it knows, but also understand what it does not know, decide when it needs to adapt, and verify whether its adaptation actually improves reliability."*

**AEGIS** is a modular, research-driven framework for detecting AI-generated images.

Single static classifiers tend to break down on images that were compressed, blurred or resized (for example by social media), and on images from generators they never saw in training. AEGIS addresses this by combining several complementary forensic experts and deciding, image by image, how much to trust each one. It does this through:

- **Out-of-distribution (OOD) and degradation profiling**: measuring how unusual or damaged an input is.
- **Probability calibration**: making each expert's confidence scores mean what they say.
- **Dynamic weight routing**: shifting trust between experts based on that evidence.
- **Safe few-shot adaptation**: adapting to new data only when there is enough of it, and only if it measurably helps.

> [!NOTE]
> **Current scope:** detection of AI-generated *static images* only. Video, manipulation localization and web crawling are out of scope.
>
> **Status:** the project is in early development. The repository structure and development workflow are in place; the modules described below are being implemented. See the [Roadmap](#roadmap).

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Research Principles](#research-principles)
- [Roadmap](#roadmap)
- [License](#license)

---

## Key Features

### 1. Heterogeneous Expert Pool

Three experts look at an image from complementary angles, so the blind spots of one are covered by the others. Each expert uses a frozen pretrained backbone with a lightweight trainable head and its own preprocessing.

| Expert | Backbone | What it looks for |
|---|---|---|
| Semantic | CLIP (UnivFD-style) | High-level semantic and structural inconsistencies |
| Fine-grained structure | DINOv2 | Texture, lighting and local patch artifacts |
| Frequency domain | FFT / high-pass CNN | Upsampling and checkerboard traces that are invisible in RGB space |

### 2. Data & OOD Profiler

Estimates how degraded an input is (JPEG compression, Gaussian blur, resolution) and how uncertain the models are about it (predictive entropy, energy score, cosine distance to known data).

### 3. Adaptive Gating Controller

Assigns calibrated weights $(w_1, w_2, w_3)$ to the experts per sample or batch, moving trust away from experts that are known to be fragile under the detected conditions. Weight bounds and entropy regularization prevent any expert from being switched off without OOD evidence.

### 4. Data-Aware Few-Shot Adaptation

How AEGIS adapts depends on how many new labeled samples $N$ are available:

| Samples | Behavior |
|---|---|
| $N < 10$ | **Guardrail.** System weights are not changed; a data-scarcity warning is logged. |
| $10 \le N \le 50$ | **Regularized update.** Weights are blended toward the new estimate: $w_{final} = (1-\alpha)\,w_{base} + \alpha\,w_{new}$ |
| $N > 50$ | **Full recalibration** of the gating controller. |

### 5. Champion-Challenger Verification & Rollback

Every update is evaluated against the current production model (the *champion*) on a held-out benchmark. It is promoted only if it performs better; otherwise the system automatically rolls back.

### 6. Built for Local Hardware

Backbones are frozen and their embeddings are cached to disk (`.pt` / `.parquet`), so calibration and gating experiments run in seconds on a consumer CPU or GPU.

---

## System Architecture

```text
                        [ Input Image / Batch ]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        [ Data & OOD Profiler ]        [ Heterogeneous Expert Pool ]
        • JPEG / blur estimation       • Expert 1: Semantic (CLIP)
        • OOD distance & entropy       • Expert 2: Structure (DINOv2)
                    │                  • Expert 3: Frequency (FFT)
                    └──────────────┬──────────────┘
                                   ▼
                   [ Adaptive Gating Controller ]
                   • Probability calibration
                   • Dynamic weighting (w1, w2, w3)
                   • Few-shot anchor adaptation
                                   │
                                   ▼
                 [ Champion-Challenger & Rollback ]
                                   │
                                   ▼
          [ Calibrated Prediction + Uncertainty Warning ]
```

---

## Project Structure

```text
aegis-ai-forensics/
├── .claude/agents/      # Specialized AI subagents used for development
├── CLAUDE.md            # Project rules: architecture, hardware and scientific constraints
├── configs/             # Centralized hyperparameters (default.yaml)
├── docs/
│   ├── pdfs/            # Source papers (PDF, not tracked by git)
│   ├── raw_papers/      # Papers converted to Markdown
│   └── research/        # Paper analyses and method specifications
├── data/                # Not tracked by git
│   ├── toy/             # 100-sample dataset for fast tests
│   └── cache/           # Cached feature embeddings
├── src/
│   ├── core/            # Shared data contracts, logging, config loading
│   ├── data/            # Validation, SHA-256 deduplication, leakage-free splits
│   ├── models/          # Frozen backbones and expert heads
│   ├── ood/             # Degradation and uncertainty estimators
│   ├── learning/        # Calibration, gating controller, few-shot adaptation
│   └── evaluation/      # Metrics, Champion-Challenger, rollback, ablation studies
├── scripts/             # Data preparation and experiment runners
└── tests/               # Pytest suite
```

---

## Getting Started

> [!IMPORTANT]
> The modules are still being implemented. The commands below describe the intended workflow; some files they reference (such as `requirements.txt` and the scripts) are not in the repository yet.

### 1. Set up the environment

```bash
git clone https://github.com/SuccessPhan/aegis-ai-forensics.git
cd aegis-ai-forensics

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the sanity check

Generate the toy dataset, then run the test suite. Every module is required to pass on the toy dataset in under 15 seconds.

```bash
python scripts/create_toy_dataset.py
pytest tests/
```

### 3. Run benchmarks and ablation studies

```bash
python scripts/run_ablation.py --config configs/default.yaml
```

---

## Research Principles

AEGIS follows a hypothesis-driven research workflow with these rules:

- **Leakage-free evaluation.** Data is deduplicated by SHA-256 and split three ways: `train_experts` trains the expert heads, `val_gating` is used for calibration and gating, and `test_benchmark` is held out for evaluation only.
- **No self-labeling loops.** Unverified model predictions are never treated as ground truth for adaptation.
- **Traceable methods.** Every method used is documented in `docs/research/` from the source papers, without invented citations or numbers.
- **Single source of truth.** All modules share the data contracts in `src/core/contracts.py` and read hyperparameters from `configs/default.yaml`.

---

## Roadmap

- [x] **Milestone 1:** Project structure, development rules (`CLAUDE.md`) and multi-agent workflow.
- [ ] **Milestone 2:** Data contracts, leakage-free dataset pipeline and toy dataset.
- [ ] **Milestone 3:** Expert pool (CLIP, DINOv2, FFT) with frozen feature caching.
- [ ] **Milestone 4:** Probability calibration, OOD/degradation profiling and adaptive gating.
- [ ] **Milestone 5:** Few-shot adaptation, Champion-Challenger rollback loop and ablation studies.
- [ ] **Milestone 6:** Streamlit diagnostic dashboard.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
