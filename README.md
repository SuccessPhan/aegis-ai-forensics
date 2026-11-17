cat << 'EOF' > README.md
# AEGIS: Adaptive Multi-Expert AI-Generated Image Forensics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Modular](https://img.shields.io/badge/Architecture-Adaptive%20MoE-success.svg)]()

> *"An intelligent detection system should not only recognize what it knows, but also understand what it does not know, decide when it needs to adapt, and verify whether its adaptation actually improves reliability."*

**AEGIS** (`aegis-ai-forensics`) is a modular, research-driven framework for **AI-Generated Image Detection**. Instead of relying on a single static classifier that degrades under post-processing distortions (e.g., social media compression) or unseen generative architectures, AEGIS orchestrates a pool of heterogeneous forensic experts via **Out-of-Distribution (OOD) profiling, dynamic weight routing, and safe few-shot calibration**.

---

## Key Architectural Highlights

1. **Heterogeneous Multi-Expert Pool:** Combines complementary forensic perspectives to eliminate single-model blind spots:
   * **Semantic Expert (`CLIP`):** Captures high-level structural and semantic inconsistencies.
   * **Fine-Grained Structural Expert (`DINOv2`):** Analyzes surface textures, lighting, and local patch artifacts.
   * **Frequency-Domain Expert (`FFT / High-Pass CNN`):** Detects high-frequency upsampling and checkerboard traces invisible in RGB space.
2. **Data & OOD Profiler:** Evaluates input image degradation (JPEG compression artifacts, Gaussian blur, resolution shifts) alongside model-intrinsic uncertainty (Predictive Entropy & Energy/Cosine OOD scores).
3. **Adaptive Gating Controller:** Dynamically assigns confidence-calibrated weights $(w_1, w_2, w_3)$ to experts per sample/batch, shifting reliance away from fragile experts when distortions are detected.
4. **Data-Aware Few-Shot Calibration & Safety Rollback:**
   * *Low-Sample Guardrail ($N < 10$):* Defaults to conservative routing without mutating system weights.
   * *Few-Shot Regime ($10 \le N \le 50$):* Performs regularized prototype/weight calibration in feature space without retraining heavy backbones.
   * *Champion-Challenger Verification:* Validates every weight update against a benchmark suite with automatic rollback on performance regression.
5. **Local-First Engineering:** Designed with frozen backbone feature caching (`.pt`/`.parquet`) for rapid experimentation on consumer hardware.

---

## System Architecture

```text
                        [ Input Image / Batch ]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        [ Data & OOD Profiler ]        [ Heterogeneous Expert Pool ]
        • JPEG / Blur Estimation       • Expert 1: Semantic (CLIP)
        • OOD Distance & Entropy       • Expert 2: Structure (DINOv2)
                    │                  • Expert 3: Frequency (FFT)
                    └──────────────┬──────────────┘
                                   ▼
                   [ Adaptive Gating Controller ]
                   • Probability Calibration
                   • Dynamic Weighting (w1, w2, w3)
                   • Few-Shot Anchor Adaptation
                                   │
                                   ▼
                 [ Champion-Challenger & Rollback ]
                                   │
                                   ▼
          [ Calibrated Prediction + Uncertainty Warning ]
```
## Project Structure
```
aegis-ai-forensics/
├── .claude/agents/            # Specialized AI subagents for modular development
├── CLAUDE.md                  # Project constitution, hardware & scientific rules
├── configs/
│   └── default.yaml           # Centralized hyperparameters and routing thresholds
├── docs/
│   ├── raw_papers/            # Extracted markdown texts of reference papers
│   └── research/              # Literature review, research gaps, and method specs
├── data/
│   ├── toy/                   # Lightweight sanity-check dataset (100 samples)
│   └── cache/                 # Cached feature embeddings for fast local iteration
├── src/
│   ├── core/                  # Data contracts (contracts.py), logging, config loader
│   ├── data/                  # Validation, SHA-256 deduplication, leakage-free splits
│   ├── models/                # Frozen backbone wrappers & expert classifiers
│   ├── ood/                   # Image quality profilers & OOD/uncertainty estimators
│   ├── learning/              # Adaptive gating network & few-shot calibrator
│   └── evaluation/            # Metrics (AUROC, F1, ECE), Champion-Challenger, Rollback
├── scripts/                   # Experiment runners & ablation study pipelines
└── tests/                     # Pytest suite enforcing interface contract
```
## Getting Started
### 1. Environment Setup
```
git clone https://github.com/SuccessPhan/aegis-ai-forensics.git
cd aegis-ai-forensics

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Sanity Check on Toy Dataset
Verify the end-to-end data contracts, expert inference, and gating pipeline on the lightweight toy dataset:
```
python scripts/create_toy_dataset.py
pytest tests/
```
### 3. Running Ablation & Routing Benchmarks
```
python scripts/run_ablation.py --config configs/default.yaml
```
## Scientific Foundation & Reproducibility
### AEGIS adheres strictly to a hypothesis-driven research workflow:
image_2ba581.png
-  No Self-Labeling Loops: Unverified model predictions are never used as absolute ground truth for adaptation.
-  Leakage-Free Evaluation: Strict separation between train_experts, val_gating (calibration), and test_benchmark splits.
-  Traceable Baselines: Every integrated detector and OOD metric is documented in docs/research/LITERATURE_REVIEW.md with verified publication metadata and official repositories.
## Roadmap
- [x] Phase 1: Core architecture, strict data contracts (contracts.py), and multi-agent workflow setup.
- [ ] Phase 2: Leakage-free dataset pipeline and frozen feature caching (CLIP, DINOv2, FFT).
- [ ] Phase 3: Probability calibration, OOD/degradation profiling, and adaptive gating controller.
- [ ] Phase 4: Few-shot calibration, Champion-Challenger rollback loop, and Streamlit diagnostic dashboard.
## License
### Distributed under the MIT License. See LICENSE for more information.
### EOF
