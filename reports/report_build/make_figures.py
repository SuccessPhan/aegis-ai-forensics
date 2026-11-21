"""Build the report figures and a JSON of every number the report quotes.

All numbers are read from experiment outputs (reports/v2, reports/v3, data/normalized_*/shortcut_audit.json,
data/cache/freq_selection.json). Nothing is typed in by hand except the SRM BatchNorm bug measurements, which
come from the orchestrator's diagnostic run (documented in docs/PROJECT_BRIEF.md and commit a580571).

Run from the repo root:  .venv/Scripts/python reports/report_build/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "report_build" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (dataviz reference palette, light mode, slots 1-5).
C = {"clip": "#2a78d6", "dinov2": "#eb6834", "freq": "#1baf7a", "static_lr": "#eda100", "aegis": "#e87ba4"}
SURFACE, INK, INK2, GRID, FLOOR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
NAMES = {"clip": "CLIP", "dinov2": "DINOv2", "freq": "Freq (SRM)", "static_lr": "Static LR",
         "aegis": "AEGIS", "naive_avg": "Naive avg", "static_learned_support": "Static LR + support",
         "aegis_noadapt": "AEGIS (không thích nghi)", "quality_only": "Sàn quality_only"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 9,
})


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


numbers = {}

# ---------------------------------------------------------------- 1. shortcut audit
audits = {}
for tag, d in [("crop", "normalized_crop"), ("resize_crop", "normalized_resize_crop"),
               ("v1", "normalized_v1"), ("v2", "normalized_v2")]:
    a = json.load(open(ROOT / "data" / d / "shortcut_audit.json", encoding="utf-8"))["shortcut_audit"]
    audits[tag] = {"pooled": a["auroc"], **{g: r["auroc"] for g, r in a["per_generator"].items()}}
numbers["audit"] = audits

gens = ["adm", "sd_v1_4", "midjourney", "glide", "vqdm"]
series = [("crop", "Lần 0: chỉ crop 224"), ("resize_crop", "Lần 0: resize_crop"),
          ("v1", "v1: resize_crop + cân bằng"), ("v2", "v3: dữ liệu có VQDM")]
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.2
for i, ((tag, lab), col) in enumerate(zip(series, cols)):
    xs, ys = [], []
    for j, g in enumerate(gens):
        if g in audits[tag]:
            xs.append(j + (i - 1.5) * w)
            ys.append(audits[tag][g])
    ax.bar(xs, ys, width=w * 0.9, color=col, label=lab, zorder=3)
ax.axhspan(0.45, 0.55, color=FLOOR, alpha=0.18, zorder=1)
ax.axhline(0.5, color=FLOOR, lw=1, ls="--", zorder=2)
ax.text(len(gens) - 0.55, 0.555, "vùng 'không lộ nhãn' 0.50 ± 0.05", ha="right", va="bottom", color=INK2, fontsize=8)
ax.set_xticks(range(len(gens)), ["ADM\n(seen)", "SD v1.4\n(seen)", "Midjourney\n(unseen)", "GLIDE\n(unseen)", "VQDM\n(unseen)"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("AUROC chỉ từ 3 đặc trưng chất lượng")
ax.set_title("Audit đường tắt: mức lộ nhãn qua chất lượng ảnh")
ax.legend(loc="upper left", ncol=2)
save(fig, "fig1_audit.png")

# ---------------------------------------------------------------- v3 ablation
v3 = [r for r in read_csv(ROOT / "reports/v3/ablation.csv") if r["variant"] == "full"]


def auroc(rows, hv, sc, lv, m, gen="", N=""):
    for r in rows:
        if (r["head_variant"], r["scenario"], r["level"], r["method"], r["generator"], r["N"]) == (hv, sc, lv, m, gen, N):
            return float(r["auroc"])
    return None


def s4(rows, hv, m, gen, N):
    for r in rows:
        if r["head_variant"] == hv and r["scenario"] == "S4" and r["generator"] == gen and r["method"] == m and r["N"] == N:
            return float(r["auroc"])
    return None


# ---------------------------------------------------------------- 2. expert robustness (v3, head clean)
jpeg_lv = [("clean", "S1", "sạch"), ("jpeg_q85", "S2", "Q85"), ("jpeg_q65", "S2", "Q65"), ("jpeg_q40", "S2", "Q40")]
blur_lv = [("clean", "S1", "sạch"), ("blur_k11_s1.5", "S3", "σ=1.5"), ("blur_k17_s2.5", "S3", "σ=2.5")]
rob = {}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, gridspec_kw={"width_ratios": [4, 3]})
for ax, levels, title in [(axs[0], jpeg_lv, "Nén JPEG"), (axs[1], blur_lv, "Làm mờ Gaussian")]:
    for m in ["clip", "dinov2", "freq", "aegis"]:
        ys = [auroc(v3, "clean", sc, lv, m) for lv, sc, _ in levels]
        rob.setdefault(title, {})[m] = ys
        ax.plot(range(len(levels)), ys, color=C[m], lw=2, marker="o", ms=6, label=NAMES[m],
                ls="--" if m == "aegis" else "-", zorder=4 if m == "aegis" else 3)
    ax.set_xticks(range(len(levels)), [t for _, _, t in levels])
    ax.set_title(title, fontweight="normal")
axs[0].set_ylabel("AUROC (test v3, head sạch)")
axs[0].set_ylim(0.45, 1.01)
fig.suptitle("Mỗi expert hỏng ở một kiểu suy giảm khác nhau", fontweight="bold", fontsize=11)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
save(fig, "fig2_robustness.png")
numbers["robustness_v3_clean"] = rob

# ---------------------------------------------------------------- 3. main v3 dot plot
cells = [("S1", "clean", "", "", "S1 sạch"), ("S2", "all", "", "", "S2 JPEG"),
         ("S3", "all", "", "", "S3 mờ/resize"), ("S4", None, "midjourney", "20", "S4 Midjourney N=20")]
methods = ["clip", "dinov2", "freq", "static_lr", "aegis"]
main = {}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for ax, hv in zip(axs, ["clean", "aug"]):
    for i, (sc, lv, gen, N, lab) in enumerate(cells):
        y = len(cells) - 1 - i
        vals = {}
        for k, m in enumerate(methods):
            if sc == "S4":
                v = s4(v3, hv, m if m != "static_lr" else "static_learned_support", gen, N) if m in ("aegis", "static_lr") \
                    else s4(v3, hv, m, gen, "")
            else:
                v = auroc(v3, hv, sc, lv, m)
            vals[m] = v
            ax.scatter(v, y + (k - 2) * 0.1, s=42, color=C[m], zorder=3, edgecolor=SURFACE, linewidth=1.2,
                       label=NAMES[m] if i == 0 else None)
        q = auroc(v3, hv, sc, lv, "quality_only") if sc != "S4" else s4(v3, hv, "quality_only", gen, "")
        ax.plot([q, q], [y - 0.3, y + 0.3], color=FLOOR, lw=2, zorder=2, label="Sàn quality_only" if i == 0 else None)
        vals["quality_only"] = q
        main[f"{hv}|{lab}"] = vals
    ax.set_title(f"Head {'sạch' if hv == 'clean' else 'augmentation'}", fontweight="normal")
    ax.set_xlim(0.45, 1.005)
    ax.set_xlabel("AUROC")
axs[0].set_yticks(range(len(cells)), [c[-1] for c in reversed(cells)])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07), fontsize=8)
fig.suptitle("Kết quả chính v3 (ở S4, 'Static LR' là bản refit trên cùng support set)", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig3_main_v3.png")
numbers["main_v3"] = main

# ---------------------------------------------------------------- 4. forest plot of significance (v3)
sig = read_csv(ROOT / "reports/v3/significance.csv")


def sig_row(rows, hv, sc, lv, gen, N, b):
    for r in rows:
        if (r["head_variant"], r["scenario"], r["level"], r["generator"], r["N"], r["baseline"], r["metric"]) == \
                (hv, sc, lv, gen, N, b, "auroc"):
            return {k: float(r[k]) for k in ("delta", "ci_lo", "ci_hi", "p_value")} | {"sig": r["significant"] == "True"}
    return None


forest_cells = [("S1", "clean", "", "", "static_lr", "S1 vs Static LR"),
                ("S2", "all", "", "", "static_lr", "S2 vs Static LR"),
                ("S3", "all", "", "", "static_lr", "S3 vs Static LR"),
                ("S4", "all", "midjourney", "20", "static_learned_support", "S4 MJ N=20 vs Static LR+support"),
                ("S4", "all", "midjourney", "100", "static_learned_support", "S4 MJ N=100 vs Static LR+support"),
                ("S4", "all", "vqdm", "20", "static_learned_support", "S4 VQDM N=20 vs Static LR+support")]
forest = {}
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (sc, lv, gen, N, b, lab) in enumerate(forest_cells):
    y = len(forest_cells) - 1 - i
    for k, (hv, col) in enumerate([("clean", "#2a78d6"), ("aug", "#eb6834")]):
        r = sig_row(sig, hv, sc, lv, gen, N, b)
        forest[f"{hv}|{lab}"] = r
        yy = y + (0.13 if k == 0 else -0.13)
        ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=col, lw=2, zorder=3)
        ax.scatter(r["delta"], yy, s=40, color=col, zorder=4, edgecolor=SURFACE, linewidth=1.2,
                   label=(f"Head {'sạch' if hv == 'clean' else 'augmentation'}") if i == 0 else None)
ax.axvline(0, color=FLOOR, lw=1.2, zorder=2)
ax.set_yticks(range(len(forest_cells)), [c[-1] for c in reversed(forest_cells)])
ax.set_xlabel("ΔAUROC = AEGIS − baseline   (← AEGIS kém hơn | AEGIS tốt hơn →)\n"
              "CI 95%, bootstrap theo cụm ảnh gốc, B=2000; đoạn không cắt 0 = có ý nghĩa thống kê")
ax.set_title("Mức ý nghĩa thống kê của chênh lệch (v3)")
ax.legend(loc="lower right")
save(fig, "fig4_forest_v3.png")
numbers["forest_v3"] = forest

# ---------------------------------------------------------------- 5. few-shot vs N (Midjourney, v3)
Ns = ["5", "20", "100"]
fs = {}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, hv in zip(axs, ["clean", "aug"]):
    lines = {"aegis": [s4(v3, hv, "aegis", "midjourney", n) for n in Ns],
             "static_learned_support": [s4(v3, hv, "static_learned_support", "midjourney", n) for n in Ns]}
    noad = s4(v3, hv, "aegis_noadapt", "midjourney", "0")
    clip = s4(v3, hv, "clip", "midjourney", "")
    fs[hv] = {**lines, "aegis_noadapt": noad, "clip": clip}
    ax.plot(range(3), lines["aegis"], color=C["aegis"], lw=2, marker="o", ms=7, label="AEGIS", zorder=3)
    ax.plot(range(3), lines["static_learned_support"], color=C["static_lr"], lw=2, marker="s", ms=6,
            label="Static LR + support", zorder=3)
    ax.axhline(noad, color=FLOOR, lw=1.5, ls="--", label="AEGIS không thích nghi")
    ax.axhline(clip, color=C["clip"], lw=1.2, ls=":", label="CLIP đơn lẻ")
    ax.set_xticks(range(3), [f"N={n}\n({r})" for n, r in zip(Ns, ["giữ nguyên", "neo α", "refit"])])
    ax.set_title(f"Head {'sạch' if hv == 'clean' else 'augmentation'}", fontweight="normal")
axs[0].set_ylabel("AUROC trên Midjourney (unseen)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=8)
fig.suptitle("Thích nghi few-shot trên generator chưa thấy (trung bình 5 seed)", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig5_fewshot_v3.png")
numbers["fewshot_mj_v3"] = fs

# ---------------------------------------------------------------- 6. replication v1(v2) vs v3
sig_v1 = read_csv(ROOT / "reports/v2/significance.csv")   # v2 predictions are byte-identical to v1
rep_cells = [("S1", "clean", "", "", "static_lr", "S1"), ("S2", "all", "", "", "static_lr", "S2"),
             ("S3", "all", "", "", "static_lr", "S3"),
             ("S4", "all", "midjourney", "20", "static_learned_support", "S4 MJ N=20")]
rep = {}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, hv in zip(axs, ["clean", "aug"]):
    for k, (rows, lab, col) in enumerate([(sig_v1, "v1 (GLIDE, split cũ)", "#2a78d6"), (sig, "v3 (VQDM, split mới)", "#eb6834")]):
        for j, (sc, lv, gen, N, b, name) in enumerate(rep_cells):
            r = sig_row(rows, hv, sc, lv, gen, N, b)
            rep[f"{hv}|{name}|{lab[:2]}"] = r
            x = j + (k - 0.5) * 0.36
            ax.bar(x, r["delta"], width=0.32, color=col, zorder=3, label=lab if j == 0 else None)
            ax.plot([x, x], [r["ci_lo"], r["ci_hi"]], color=INK2, lw=1, zorder=4)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(rep_cells)), [c[-1] for c in rep_cells])
    ax.set_title(f"Head {'sạch' if hv == 'clean' else 'augmentation'}", fontweight="normal")
axs[0].set_ylabel("ΔAUROC AEGIS − Static LR")
axs[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Kết quả có lặp lại khi đổi dữ liệu? (thanh dọc = CI 95%)", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig6_replication.png")
numbers["replication"] = rep

# ---------------------------------------------------------------- 7. oracle headroom (v2 predictions, diagnostic only)
grid = [(a, b, 1 - a - b) for a in np.arange(0, 1.0001, 0.05) for b in np.arange(0, 1.0001 - a, 0.05)]
slices = [("S1", "clean", "sạch"), ("S2", "jpeg_q85", "Q85"), ("S2", "jpeg_q65", "Q65"), ("S2", "jpeg_q40", "Q40"),
          ("S3", "blur_k11_s1.5", "σ1.5"), ("S3", "blur_k17_s2.5", "σ2.5"), ("S3", "resize_0.5", "resize")]
orc = {}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
for ax, hv in zip(axs, ["clean", "aug"]):
    z = np.load(ROOT / f"reports/v2/predictions/predictions_{hv}.npz", allow_pickle=True)
    for i, (sc, lv, lab) in enumerate(slices):
        pre = f"full::{sc}::{lv}::-::-::-##"
        y = z[pre + "y"]
        P = np.stack([z[pre + "p_clip"], z[pre + "p_dinov2"], z[pre + "p_freq"]], 1)
        o = max(roc_auc_score(y, P @ np.array(w)) for w in grid)
        a = roc_auc_score(y, z[pre + "p_aegis"])
        s = roc_auc_score(y, z[pre + "p_static_lr"])
        orc[f"{hv}|{lab}"] = {"static_lr": s, "aegis": a, "oracle": o}
        ax.plot([i, i], [a, o], color=GRID, lw=3, zorder=1)
        ax.scatter(i, s, color=C["static_lr"], s=40, zorder=3, label="Static LR" if i == 0 else None, edgecolor=SURFACE)
        ax.scatter(i, a, color=C["aegis"], s=40, zorder=3, label="AEGIS" if i == 0 else None, edgecolor=SURFACE)
        ax.scatter(i, o, color=INK2, marker="_", s=160, zorder=3, linewidth=2, label="Oracle (cận trên)" if i == 0 else None)
    ax.set_xticks(range(len(slices)), [s[-1] for s in slices])
    ax.set_title(f"Head {'sạch' if hv == 'clean' else 'augmentation'}", fontweight="normal")
axs[0].set_ylabel("AUROC (test v2)")
axs[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Dư địa của việc chia trọng số: AEGIS so với cận trên oracle", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig7_oracle.png")
numbers["oracle_v2"] = orc

# ---------------------------------------------------------------- misc numbers
numbers["freq_selection_v2"] = json.load(open(ROOT / "data/cache/freq_selection.json", encoding="utf-8"))
numbers["splits_v2"] = json.load(open(ROOT / "data/normalized_v2/shortcut_audit.json", encoding="utf-8")).get("split_label_table")
numbers["srm_bug"] = {"before_train_auroc": 0.510, "before_val_auroc": 0.526, "after_val_auroc_300": 0.988,
                      "train_loss_final": 0.177}
fus = json.load(open(ROOT / "reports/fusion_selection_v3/fusion_selection.json", encoding="utf-8"))
numbers["fusion_selection"] = fus
(OUT.parent / "numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
