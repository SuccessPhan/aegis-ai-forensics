import copy
from collections import Counter

import pytest

from src.core.contracts import (S4_SPLITS, SPLIT_S4_EVAL, SPLIT_S4_SUPPORT, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL_FIT,
                                SPLIT_VAL_VERIFY, SPLITS, SplitManifest)
from src.data.split import assign_splits, check_no_leakage, get_s4_eval_set, get_support_set

UNSEEN = {"midjourney", "glide"}


def test_toy_has_100_balanced_clean_sources(toy_manifest):
    clean = [r for r in toy_manifest.records if r.is_clean]
    assert len(clean) == 100
    assert Counter(r.label for r in clean) == {0: 50, 1: 50}
    assert UNSEEN <= {r.generator for r in clean}
    assert len({r.sha256 for r in toy_manifest.records}) == len(toy_manifest.records)   # dedup'd


def test_no_sha_or_group_across_splits(toy_manifest, cfg):
    sha_split, grp_split = {}, {}
    for r in toy_manifest.records:
        assert sha_split.setdefault(r.sha256, r.split) == r.split
        assert grp_split.setdefault(r.source_group_id, r.split) == r.split
    assert check_no_leakage(toy_manifest, cfg) == []


def test_all_four_splits_plus_s4_populated(toy_manifest):
    for s in SPLITS + S4_SPLITS:
        assert toy_manifest.by_split(s), s


def test_unseen_generators_only_in_s4(toy_manifest):
    for r in toy_manifest.records:
        if r.generator in UNSEEN:
            assert r.split in S4_SPLITS
        if r.split in SPLITS:
            assert r.generator not in UNSEEN


def test_variants_share_parent_split_and_val_contains_degraded(toy_manifest):
    by_sha = {r.sha256: r for r in toy_manifest.records}
    variants = [r for r in toy_manifest.records if not r.is_clean]
    assert variants
    for v in variants:
        p = by_sha[v.parent_sha256]
        assert p.split == v.split and p.source_group_id == v.source_group_id
    for s in (SPLIT_VAL_FIT, SPLIT_VAL_VERIFY):
        assert any(not r.is_clean for r in toy_manifest.by_split(s)), s
    assert not any(not r.is_clean for r in toy_manifest.by_split(SPLIT_TRAIN))
    test_names = {r.degradation.name for r in toy_manifest.by_split(SPLIT_TEST) if not r.is_clean}
    assert {"jpeg_q85", "jpeg_q65", "jpeg_q40", "resize_0.5", "blur_k11_s1.5", "blur_k17_s2.5"} == test_names


@pytest.mark.parametrize("n", [5, 6, 20])
def test_support_set_balanced_and_disjoint_from_eval(toy_manifest, n):
    sup = get_support_set(toy_manifest, n, seed=0)
    assert len(sup.ai_records) == (n + 1) // 2 and len(sup.real_records) == n // 2
    assert all(r.label == 1 and r.generator in UNSEEN for r in sup.ai_records)
    assert all(r.label == 0 for r in sup.real_records)
    ev = get_s4_eval_set(toy_manifest)
    assert ev and {r.split for r in sup.records} == {SPLIT_S4_SUPPORT} and {r.split for r in ev} == {SPLIT_S4_EVAL}
    assert not {r.sha256 for r in sup.records} & {r.sha256 for r in ev}
    assert not {r.source_group_id for r in sup.records} & {r.source_group_id for r in ev}
    ai_ev = [r for r in ev if r.label == 1]
    assert ai_ev and all(r.generator in UNSEEN for r in ai_ev)


def test_support_is_seeded_and_nested(toy_manifest):
    a, b = get_support_set(toy_manifest, 20, 1), get_support_set(toy_manifest, 20, 1)
    assert [r.sha256 for r in a.records] == [r.sha256 for r in b.records]
    c = get_support_set(toy_manifest, 20, 2)
    assert [r.sha256 for r in a.records] != [r.sha256 for r in c.records]
    small = get_support_set(toy_manifest, 6, 1)
    assert {r.sha256 for r in small.records} <= {r.sha256 for r in a.records}
    with pytest.raises(ValueError):
        get_support_set(toy_manifest, 100, 1)       # toy pool too small for N=100


def test_leakage_checker_has_power(toy_manifest, cfg):
    bad = SplitManifest(copy.deepcopy(toy_manifest.records), toy_manifest.seed, toy_manifest.root)
    tr = next(r for r in bad.records if r.split == SPLIT_TRAIN and r.is_clean)
    tr.split = SPLIT_TEST                      # same group now spans two splits (variants none for train)
    g = next(r for r in bad.records if r.generator == "glide")
    g.split = SPLIT_TRAIN                      # unseen generator in train
    assert len(check_no_leakage(bad, cfg)) >= 1
    ok = assign_splits(copy.deepcopy([r for r in toy_manifest.records if r.is_clean]), cfg)
    assert any("s4" in str(r.split) for r in ok.records)


def test_split_deterministic(toy_manifest, cfg):
    clean = [r for r in toy_manifest.records if r.is_clean]
    toy_cfg = copy.deepcopy(cfg)
    toy_cfg["split"] = {**cfg["split"], **{"s4": {**cfg["split"]["s4"], **cfg["toy"]["split_overrides"]["s4"]}}}
    a = assign_splits(copy.deepcopy(clean), toy_cfg, seed=cfg["seed"])
    assert {r.sha256: r.split for r in a.records} == {r.sha256: r.split for r in clean}
