"""Group-aware, leakage-free splitting + S4 (OOD / few-shot) pools.

Splits (by source_group_id; every degraded variant follows its source):
  train_experts / val_gating_fit / val_gating_verify / test_benchmark   (seen generators + Real)
  s4_support / s4_eval   (unseen generators held out ENTIRELY from train/val; plus a dedicated Real store)
Unseen-generator AI images: per generator, a support pool and a disjoint eval pool.
Real images: a Real support store (s4_support) and a Real eval pool (s4_eval) are carved out first,
then the remaining Real images are split like the seen-generator fakes.
Stratified per (generator | real) so every split sees every class; deterministic given the seed.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from src.core.contracts import (ALL_SPLITS, SCENARIO_TRAIN_AUG, LABEL_FAKE, LABEL_REAL, SPLIT_S4_EVAL, SPLIT_S4_SUPPORT,
                                SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL_FIT, SPLIT_VAL_VERIFY, SPLITS, ImageRecord,
                                SplitManifest, SupportSet)
from src.core.logger import get_logger

log = get_logger(__name__)


def _cuts(n: int, fracs: List[float]) -> List[int]:
    """Sizes summing to n via largest remainder."""
    raw = [f * n for f in fracs]
    sizes = [int(math.floor(x)) for x in raw]
    rem = n - sum(sizes)
    order = sorted(range(len(fracs)), key=lambda i: raw[i] - sizes[i], reverse=True)
    for i in order[:rem]:
        sizes[i] += 1
    return sizes


def assign_splits(records: List[ImageRecord], cfg: dict, seed: Optional[int] = None, root: str = "") -> SplitManifest:
    """Assign `split` to every record (mutates records) and return a SplitManifest."""
    seed = cfg.get("seed", 0) if seed is None else seed
    gcfg = cfg["data"]["generators"]
    unseen = set(gcfg["unseen"])
    real_name = cfg["data"]["real_name"]
    sc = cfg["split"]
    ratios = [sc["ratios"]["train_experts"], sc["ratios"]["val_gating"], sc["ratios"]["test_benchmark"]]
    tot = float(sum(ratios))
    ratios = [r / tot for r in ratios]
    s4 = sc["s4"]
    rng = np.random.default_rng(seed)

    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        groups[r.source_group_id].append(r)
    strata: Dict[str, List[str]] = defaultdict(list)
    for gid in sorted(groups):
        gens = {r.generator for r in groups[gid]}
        if len(gens) != 1:
            raise ValueError(f"group {gid} mixes generators {gens}")
        strata[gens.pop()].append(gid)

    group_split: Dict[str, str] = {}

    def put(gids: List[str], split: str) -> None:
        for g in gids:
            group_split[g] = split

    def main_split(gids: List[str]) -> None:
        n_tr, n_val, n_te = _cuts(len(gids), ratios)
        n_fit, n_ver = _cuts(n_val, [sc["val_fit_frac"], 1 - sc["val_fit_frac"]])
        if n_val >= 2 and n_ver == 0:
            n_fit, n_ver = n_fit - 1, 1
        i = 0
        for split, n in ((SPLIT_TRAIN, n_tr), (SPLIT_VAL_FIT, n_fit), (SPLIT_VAL_VERIFY, n_ver), (SPLIT_TEST, n_te)):
            put(gids[i:i + n], split)
            i += n

    balance = bool(sc.get("balance_real", False))
    for gen in sorted(strata):
        gids = list(strata[gen])
        rng.shuffle(gids)
        n = len(gids)
        if gen == real_name and balance:
            continue                      # allocated below, once the AI split sizes are known
        if gen in unseen:
            n_sup = min(n - 1, max(math.ceil(s4["support_pool_frac"] * n), int(s4["support_pool_min"]))) if n > 1 else 0
            put(gids[:n_sup], SPLIT_S4_SUPPORT)
            put(gids[n_sup:], SPLIT_S4_EVAL)
        elif gen == real_name:
            n_sup = max(math.ceil(s4["real_support_frac"] * n), int(s4["real_min"]))
            n_ev = max(math.ceil(s4["real_eval_frac"] * n), int(s4["real_min"]))
            if n_sup + n_ev >= n:  # tiny datasets: never starve the main splits completely
                n_sup = n_ev = max(1, n // 4)
            put(gids[:n_sup], SPLIT_S4_SUPPORT)
            put(gids[n_sup:n_sup + n_ev], SPLIT_S4_EVAL)
            main_split(gids[n_sup + n_ev:])
        else:
            main_split(gids)

    dropped = 0
    if balance and real_name in strata:
        # Real groups are allocated so that clean Real per split == clean AI per split. Order matters only when
        # Real is short: s4_eval is filled last and absorbs the whole shortfall.
        ai_n = {s: 0 for s in ALL_SPLITS}
        for gid, g in group_split.items():
            ai_n[g] += 1
        real_gids = list(strata[real_name])
        rng.shuffle(real_gids)
        i = 0
        got = {}
        order = [SPLIT_S4_SUPPORT, SPLIT_TRAIN, SPLIT_VAL_FIT, SPLIT_VAL_VERIFY, SPLIT_TEST, SPLIT_S4_EVAL]
        for s in order:
            k = min(ai_n[s], len(real_gids) - i)
            put(real_gids[i:i + k], s)
            got[s] = k
            i += k
        dropped = len(real_gids) - i
        for gid in real_gids[i:]:
            group_split[gid] = ""         # surplus Real: not needed for 50/50 -> removed below
        short = {s: ai_n[s] - got[s] for s in order if ai_n[s] != got[s]}
        if short:
            log.warning("balance_real: Real is short; shortfall per split (AI - Real) = %s", short)
        if dropped:
            log.warning("balance_real: %d surplus Real source groups dropped (more Real than AI)", dropped)
        log.info("balance_real: clean Real per split = %s vs AI = %s", got, ai_n)
    if dropped:
        records[:] = [r for r in records if group_split[r.source_group_id] != ""]
    for r in records:
        r.split = group_split[r.source_group_id]
        r.is_unseen_generator = r.generator in unseen
    man = SplitManifest(records=records, seed=seed, root=root,
                        meta={"unseen": sorted(unseen), "splits": list(ALL_SPLITS)})
    problems = check_no_leakage(man, cfg)
    if problems:
        raise AssertionError("split leakage: " + "; ".join(problems))
    log.info("split done: %s", {s: len(man.by_split(s)) for s in ALL_SPLITS})
    log.info("split x label (clean) [Real/AI]: %s", split_label_table(man))
    return man


def split_label_table(man: SplitManifest) -> Dict[str, Dict[str, int]]:
    """Clean-image counts per split x label: {split: {"real": n, "ai": n}}."""
    out: Dict[str, Dict[str, int]] = {}
    for s in ALL_SPLITS:
        rs = [r for r in man.by_split(s) if r.is_clean]
        out[s] = {"real": sum(r.label == LABEL_REAL for r in rs), "ai": sum(r.label == LABEL_FAKE for r in rs)}
    return out


def check_no_leakage(man: SplitManifest, cfg: dict) -> List[str]:
    """Return a list of violations (empty == clean)."""
    problems: List[str] = []
    unseen = set(cfg["data"]["generators"]["unseen"])
    grp: Dict[str, set] = defaultdict(set)
    sha: Dict[str, set] = defaultdict(set)
    for r in man.records:
        grp[r.source_group_id].add(r.split)
        sha[r.sha256].add(r.split)
        if r.split is None:
            problems.append(f"{r.sha256[:8]} has no split")
        if r.generator in unseen and r.split in SPLITS:
            problems.append(f"unseen generator {r.generator} in {r.split}")
        if r.generator not in unseen and r.label == LABEL_FAKE and r.split in (SPLIT_S4_SUPPORT, SPLIT_S4_EVAL):
            problems.append(f"seen fake {r.generator} in S4 pool")
    by_sha = {r.sha256: r for r in man.records}
    for r in man.records:
        if r.degradation is not None and r.degradation.scenario == SCENARIO_TRAIN_AUG:
            if r.split != SPLIT_TRAIN:
                problems.append(f"train_aug variant {r.sha256[:8]} in {r.split}")
            parent = by_sha.get(r.parent_sha256 or "")
            if parent is not None and parent.split != r.split:
                problems.append(f"train_aug variant {r.sha256[:8]} split differs from its parent")
    problems += [f"group {g} in splits {sorted(map(str, s))}" for g, s in grp.items() if len(s) > 1]
    problems += [f"sha {h[:8]} in splits {sorted(map(str, s))}" for h, s in sha.items() if len(s) > 1]
    return problems


def get_support_set(man: SplitManifest, n: int, seed: int, generator: Optional[str] = None) -> SupportSet:
    """Seeded support set: ceil(n/2) unseen-generator AI + floor(n/2) Real, drawn from the s4_support pool
    (clean images only; the Real part comes from the dedicated Real store). For a fixed seed the sets are
    nested across N (N=5 is a subset of N=20, ...). Odd N gives the extra sample to AI."""
    pool = man.by_split(SPLIT_S4_SUPPORT, clean_only=True)
    ai = sorted((r for r in pool if r.label == LABEL_FAKE and (generator is None or r.generator == generator)),
                key=lambda r: r.sha256)
    real = sorted((r for r in pool if r.label == LABEL_REAL), key=lambda r: r.sha256)
    n_ai, n_real = (n + 1) // 2, n // 2
    if len(ai) < n_ai or len(real) < n_real:
        raise ValueError(f"support pool too small for N={n}: AI {len(ai)}/{n_ai}, Real {len(real)}/{n_real}")
    rng = np.random.default_rng(seed)
    ai_perm = [ai[i] for i in rng.permutation(len(ai))]
    real_perm = [real[i] for i in rng.permutation(len(real))]
    return SupportSet(n=n, generator=generator, seed=seed, ai_records=ai_perm[:n_ai], real_records=real_perm[:n_real])


def get_s4_eval_set(man: SplitManifest, generator: Optional[str] = None) -> List[ImageRecord]:
    """S4 evaluation images (clean): unseen-generator AI eval pool + Real eval pool; disjoint from support."""
    return [r for r in man.by_split(SPLIT_S4_EVAL, clean_only=True)
            if r.label == LABEL_REAL or generator is None or r.generator == generator]
