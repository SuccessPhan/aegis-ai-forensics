# AEGIS (aegis-ai-forensics) - Project Constitution & Architecture

> **Required reading:** [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) — the research context, core mechanisms, and the mandatory Ablation Study (4 scenarios: clean / JPEG / blur-resize / OOD-few-shot; AEGIS vs. each single expert vs. Naive Average Ensemble vs. Static Learned Weights). Every agent and every module must serve that final experiment table.

## 1. Core Vision & Chosen Direction (Adaptive Multi-Expert)
- **Project:** AEGIS — Adaptive Multi-Expert AI-Generated Image Forensics System.
- **Core Philosophy:** "An intelligent detection system should not only recognize what it knows, but also understand what it does not know, decide when it needs to adapt, and verify whether its adaptation actually improves reliability."
- **Chosen Strategy (Direction A):** Instead of heavy online backpropagation of giant neural networks, AEGIS orchestrates a pool of heterogeneous forensic experts via **OOD/degradation profiling, probability calibration, dynamic weight routing $(w_1, w_2, w_3)$, and safe few-shot calibration**.
- **Phase 1 Scope:** Static AI-generated image detection only. No video, no localization, no web crawlers.

## 2. System Architecture & Pipeline
Detailed technical specification (v2.0): [`docs/PROJECT_BRIEF.md` §5](docs/PROJECT_BRIEF.md). If this file and the brief disagree, update both — never silently pick one.

1. **Data Layer (`src/data/`):** Image validation, SHA-256 deduplication, metadata provenance, shortcut removal, and strict 3-way leakage-free split:
   - **Dataset:** GenImage subset, ~12k–16k images, 50% Real (ImageNet) / 50% AI. **Seen generators:** ADM, Stable Diffusion v1.4. **Unseen (S4 only):** Midjourney, GLIDE. BigGAN is excluded (128×128). CIFAKE is allowed only as dummy data for pipeline tests — never in a benchmark.
   - **Shortcut removal (MANDATORY, before splitting):** every image (Real and AI) is Center-Cropped to **224×224** and saved as **lossless PNG**. A JPEG-history harmonization step is configurable (`data.normalize.jpeg_harmonize`) because Real ImageNet images carry JPEG traces in their pixels even after PNG re-saving.
   - **Shortcut audit (MANDATORY):** a Logistic Regression on only the 3 image-quality features (`jpeg_quality_est`, `log_laplacian_var`, `high_freq_energy_ratio`) on clean S1 data must score Accuracy and AUROC within **0.50 ± 0.05** (`test_no_dataset_shortcut`).
   - **Image-quality features** live in `src/data/quality.py` (owned by `data-architect`, signal-level only, numpy). `jpeg_quality_est` must be a **blind** estimate from pixels (blockiness) — reading quantization tables from file headers is forbidden.
   - `train_experts`: Train individual expert heads (clean images only).
   - `val_gating`: Unseen data used for probability calibration (Temperature/Platt scaling) and training the gating controller. Internally divided into `val_gating_fit` (calibration + gating training) and `val_gating_verify` (Champion-Challenger decisions). Contains degraded variants with **continuous random** parameters: JPEG $Q \in [30, 95]$, blur $\sigma \in [0.5, 3.0]$.
   - `test_benchmark`: Strictly held-out evaluation set, used **once** for the final Ablation table — never for model selection, tuning, or rollback decisions. Uses **fixed** degradation levels: S2 JPEG $Q \in \{85, 65, 40\}$; S3 resize $0.5\times$ bilinear down-up, Gaussian blur ($k=11, \sigma=1.5$) and ($k=17, \sigma=2.5$).
   - **Group-aware splitting:** all degraded variants (JPEG/blur/resize) of one source image share that source's split.
   - **OOD protocol (scenario S4):** Midjourney and GLIDE are held out entirely from `train_experts` and `val_gating`. Few-shot support samples and evaluation images are disjoint. Support set = $N/2$ new AI images + $N/2$ Real images from a separate Real pool. Run at N ∈ {5, 20, 100}.
2. **Heterogeneous Expert Pool (`src/models/`):** Must implement independent `preprocess()` per expert (never resize images before the frequency expert):
   - Expert 1: Semantic — CLIP `ViT-L/14` (fallback `ViT-B/16`), frozen backbone + linear/MLP head trained on clean `train_experts`.
   - Expert 2: Fine-grained Structure — DINOv2 `ViT-B/14`, frozen backbone + linear/MLP head trained on clean `train_experts`.
   - Expert 3: Frequency-Domain — **4-block Frequency CNN (~1.2M params)** on log-magnitude FFT / SRM high-pass input, **trained from scratch** on clean `train_experts` (the only from-scratch exception).
   - Frozen backbones always run in `eval()` + `torch.no_grad()`.
3. **OOD Profiler (`src/ood/`):** Model-intrinsic uncertainty per expert — Predictive Entropy $H(P_i)$ and Cosine Distance to the `train_experts` centroid. Reuses the image-quality features from `src/data/quality.py`.
4. **Adaptive Gating Controller (`src/learning/`):**
   - **Input (12-d):** 3 quality features (`jpeg_quality_est`, `log_laplacian_var`, `high_freq_energy_ratio`) + 3 entropies + 3 cosine distances + 3 calibrated probabilities $P_1, P_2, P_3$.
   - **Model:** 2-layer MLP `Linear(12,32) → ReLU → Dropout(0.2) → Linear(32,3)`, with entropy regularization.
   - **Weights:** $w = 0.05 + 0.85 \cdot \mathrm{softmax}(z)$, so $\sum w_i = 1$ and $w_i \in [0.05, 0.90]$. Output $P_{final} = \sum_i w_i P_i$.
   - **Data-Aware Few-Shot Adaptation (S4):**
     - $N < 10$: Conservative routing, $\alpha = 0$; keep weights unchanged; log scarcity warning.
     - $10 \le N \le 50$: Regularized anchor update $w_{adapted} = (1-\alpha)\,w_{gating} + \alpha\,w_{few\_shot}$ with $\alpha = \min(0.6, N/50)$; $w_{few\_shot}$ from regularized logistic regression / grid search on the support set; per-expert prototypes updated for OOD distances. Global `val_gating` weights are not modified.
     - $N > 50$: Full gating recalibration (refit the gating MLP on `val_gating_fit` + support set).
   - **Yellow flag** (`warning_flag = True`) if either: (a) some $P_i > 0.80$ and some $P_j < 0.20$ and $\mathrm{Std}(P_1,P_2,P_3) > 0.30$; or (b) $P_{final} \in [0.40, 0.60]$ or mean OOD score > 95th percentile of `val_gating`.
5. **Evaluation & Safety (`src/evaluation/`):** Metrics: Accuracy, Precision, Recall, F1, AUROC, ECE, plus **Selective Accuracy** (accuracy on unflagged samples) and **Flagged Rate**. Champion-Challenger verification on `val_gating_verify`, automatic Rollback on regression. Ablation baselines: each single expert, Naive Average Ensemble, Static Learned Weights (logistic-regression stacking on `val_gating_fit`), and AEGIS Adaptive Gating.

## 3. Local Hardware Constraints (STRICT)
- **Target hardware:** NVIDIA RTX 4060 Laptop (8 GB VRAM).
- **No Training From Scratch:** Freeze heavy backbones. Sole exception: the ~1.2M-parameter Frequency CNN of Expert 3.
- **Feature Caching:** Extract and cache embeddings to `data/cache/` (`.pt` float16 or `.parquet`) so gating and calibration run in seconds on local CPU/GPU.
- **Tiny-First Rule:** Every module must pass `pytest` on `data/toy/` (100 samples) in under 15 seconds.

## 4. Multi-Agent Workflow (Orchestration)
The user assigns tasks to the main session. The main session acts as **orchestrator**: it breaks the task down, delegates each part to the subagent that owns the relevant folders, verifies the results, and reports back.

| Subagent | Owns |
|---|---|
| `research-analyst` | `docs/research/` (reads `docs/raw_papers/`) |
| `data-architect` | `configs/`, `src/core/`, `src/data/` (incl. `quality.py`), `scripts/pdf_to_md.py`, `scripts/create_toy_dataset.py` |
| `expert-detector-eng` | `src/models/`, `src/ood/`, `data/cache/` |
| `gating-eval-eng` | `src/learning/`, `src/evaluation/` |
| `quick-edit` (Haiku) | Small, precisely specified edits only — no algorithm code, no contract changes |

- **Delegate large, self-contained work packages**, each with: task, inputs, constraints, and a checkable **"Done when"** criterion (usually `pytest tests/test_<area>_*` passing in < 15 s).
- **Route by cost:** light, precisely specified edits (docs wording, renames, a single config value, quick repo lookups) go to the `quick-edit` subagent (Haiku). Tiny edits that are cheaper to do than to describe, and Q&A, stay in the main session. Algorithm/design work goes to the Sonnet owner agents.
- **Order:** `data-architect` freezes `contracts.py` first → then `expert-detector-eng` and `gating-eval-eng` may run in parallel. Any contract/config change goes back through `data-architect`.
- **Verify before reporting:** the orchestrator re-runs the relevant tests itself and does not trust a subagent's "done" without evidence.
- Work spanning several owners is split per owner; no subagent edits outside its folders.

## 5. Scientific Integrity Rules
- **Zero Hallucination:** Never invent paper titles, authors, DOIs, or numbers. Analyze only markdown files present in `docs/raw_papers/`.
- **No Self-Labeling Loops:** Never treat unverified detector predictions as absolute ground truth.
- **Single Source of Truth:** All inter-module data structures must inherit from `src/core/contracts.py`. Read hyperparameters from `configs/default.yaml`. Use centralized logger (no `print()`).
