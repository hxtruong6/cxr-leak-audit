"""Tests for the audit pipeline: pairwise MI, CI, permutation, verdict."""
from __future__ import annotations

import numpy as np
import pytest

from cxr_leak_audit import (
    AuditReport,
    audit_split,
    bootstrap_ci,
    pairwise_mi,
    permutation_pvalue,
    threshold_sweep,
)


def _clean_split(n=4000, seed=0):
    """Independent labels -> no leakage."""
    rng = np.random.default_rng(seed)
    return {
        "k1": rng.integers(0, 2, n),
        "k2": rng.integers(0, 2, n),
        "u1": rng.integers(0, 2, n),
        "u2": rng.integers(0, 2, n),
    }


def _leaky_split(n=4000, seed=0):
    """u1 is a noisy copy of k1 -> strong co-occurrence leakage."""
    rng = np.random.default_rng(seed)
    k1 = rng.integers(0, 2, n)
    flip = rng.random(n) < 0.1
    return {
        "k1": k1,
        "k2": rng.integers(0, 2, n),
        "u1": np.where(flip, 1 - k1, k1),  # 90% agreement with k1
        "u2": rng.integers(0, 2, n),
    }


def test_pairwise_mi_count_and_sort():
    pairs = pairwise_mi(_clean_split(), known=["k1", "k2"], unknown=["u1", "u2"])
    assert len(pairs) == 4  # 2 unknown x 2 known
    mis = [p["mi"] for p in pairs]
    assert mis == sorted(mis, reverse=True)


def test_clean_split_passes():
    rep = audit_split(_clean_split(), known=["k1", "k2"], unknown=["u1", "u2"],
                      n_boot=200, n_perm=200, seed=0)
    assert isinstance(rep, AuditReport)
    assert rep.passed
    assert rep.max_mi < 0.05
    assert rep.n_violations == 0


def test_leaky_split_fails_and_flags_pair():
    rep = audit_split(_leaky_split(), known=["k1", "k2"], unknown=["u1", "u2"],
                      n_boot=200, n_perm=200, seed=0)
    assert not rep.passed
    assert rep.max_mi > 0.05
    assert (rep.worst_unknown, rep.worst_known) == ("u1", "k1")
    assert rep.pvalue < 0.05  # leakage is significant vs the permutation null


def test_ci_brackets_point_estimate():
    s = _leaky_split()
    lo, hi = bootstrap_ci(s["u1"], s["k1"], n_boot=300, seed=1)
    from cxr_leak_audit import miller_madow_mi
    point = miller_madow_mi(s["u1"], s["k1"])
    assert lo <= point <= hi


def test_permutation_pvalue_bounds():
    s = _clean_split()
    p = permutation_pvalue(s["u1"], s["k1"], n_perm=500, seed=2)
    assert 0.0 < p <= 1.0


def test_threshold_sweep_monotone():
    pairs = pairwise_mi(_leaky_split(), known=["k1", "k2"], unknown=["u1", "u2"])
    sweep = threshold_sweep(pairs, [0.03, 0.05, 0.07])
    # higher threshold -> fewer or equal violations
    counts = [sweep[0.03], sweep[0.05], sweep[0.07]]
    assert counts == sorted(counts, reverse=True)


def test_reproducible_with_seed():
    s = _leaky_split()
    a = audit_split(s, known=["k1", "k2"], unknown=["u1", "u2"],
                    n_boot=200, n_perm=200, seed=7)
    b = audit_split(s, known=["k1", "k2"], unknown=["u1", "u2"],
                    n_boot=200, n_perm=200, seed=7)
    assert a.to_dict() == b.to_dict()


def test_missing_label_raises():
    with pytest.raises(ValueError):
        audit_split(_clean_split(), known=["k1", "nope"], unknown=["u1"])


def test_known_unknown_overlap_raises():
    # A label in both lists would yield MI(x, x) = H(x) -> a spurious leak.
    with pytest.raises(ValueError, match="both"):
        audit_split(_clean_split(), known=["k1", "u1"], unknown=["u1", "u2"])


def test_degenerate_label_warns():
    s = _clean_split()
    s["k1"] = np.zeros_like(s["k1"])  # no positives
    with pytest.warns(UserWarning, match="single value"):
        audit_split(s, known=["k1", "k2"], unknown=["u1"],
                    n_boot=50, n_perm=50)


def test_report_serialization_roundtrip():
    rep = audit_split(_clean_split(), known=["k1"], unknown=["u1"],
                      n_boot=100, n_perm=100, seed=0)
    d = rep.to_dict()
    assert d["passed"] is True
    assert "summary" not in d  # summary() is a method, not a field
    assert rep.summary().startswith("[PASS]")
    assert "# Split leakage audit" in rep.to_markdown()
