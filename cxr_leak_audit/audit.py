"""Split-leakage audit: per-pair MI, bootstrap CI, permutation null, verdict.

The audit screens a multi-label open-set split for known<->unknown
co-occurrence leakage. For every (unknown, known) label pair it computes the
Miller--Madow mutual information (:func:`cxr_leak_audit.mi.miller_madow_mi`);
the worst pair gets a paired bootstrap confidence interval and a permutation
p-value. A split **passes** when no pair exceeds the leakage threshold
(default ``0.05`` nats).

All randomness is seeded and reproducible. Bootstrap resampling is **paired**
(the same row indices select both vectors) so the dependence structure is
preserved; the permutation null shuffles one vector to break it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence, cast

import numpy as np

from .mi import miller_madow_mi


def pairwise_mi(
    labels: dict[str, np.ndarray],
    known: Sequence[str],
    unknown: Sequence[str],
) -> list[dict]:
    """Miller--Madow MI for every (unknown, known) label pair.

    Args:
        labels: Mapping from label name to its per-sample binary vector
            ``[N]``. All vectors must be aligned (same N, same row order).
        known: Known-class label names (exclude any "no finding" head).
        unknown: Held-out (unknown) class label names.

    Returns:
        One dict per pair with keys ``unknown``, ``known``, ``mi`` (nats),
        sorted by descending ``mi``.
    """
    rows = []
    for u in unknown:
        for k in known:
            rows.append(
                {"unknown": u, "known": k,
                 "mi": miller_madow_mi(labels[u], labels[k])}
            )
    rows.sort(key=lambda r: cast(float, r["mi"]), reverse=True)
    return rows


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired bootstrap confidence interval for the Miller--Madow MI.

    Args:
        a: Label vector ``[N]``.
        b: Label vector ``[N]``, aligned with ``a``.
        n_boot: Number of bootstrap resamples.
        ci: Central confidence level (e.g. ``0.95`` for a 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``(low, high)`` percentile confidence bounds, in nats.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = miller_madow_mi(a[idx], b[idx])
    lo_q = 100 * (1 - ci) / 2
    hi_q = 100 * (1 + ci) / 2
    return float(np.percentile(boot, lo_q)), float(np.percentile(boot, hi_q))


def permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> float:
    """Permutation-null p-value for the observed Miller--Madow MI.

    The null hypothesis is independence between ``a`` and ``b``; the null
    distribution is built by shuffling ``b``. Uses the add-one estimator so
    the p-value is never exactly zero.

    Args:
        a: Label vector ``[N]``.
        b: Label vector ``[N]``, aligned with ``a``.
        n_perm: Number of permutations.
        seed: RNG seed for reproducibility.

    Returns:
        One-sided p-value ``P(MI_null >= MI_observed)``.
    """
    a = np.asarray(a)
    b = np.asarray(b).copy()
    observed = miller_madow_mi(a, b)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(b)
        if miller_madow_mi(a, b) >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def threshold_sweep(
    pairs: list[dict],
    thresholds: Sequence[float] = (0.03, 0.05, 0.07),
) -> dict[float, int]:
    """Count violating pairs at each leakage threshold.

    Args:
        pairs: Output of :func:`pairwise_mi`.
        thresholds: Leakage cutoffs (nats) to evaluate.

    Returns:
        Mapping ``threshold -> number of pairs with MI > threshold``.
    """
    mi = np.array([p["mi"] for p in pairs])
    return {float(t): int((mi > t).sum()) for t in thresholds}


@dataclass
class AuditReport:
    """Verdict and statistics from :func:`audit_split`.

    Attributes:
        passed: ``True`` if no pair exceeds ``threshold``.
        threshold: Leakage threshold used, in nats.
        n_pairs: Number of (unknown, known) pairs audited.
        max_mi: Largest Miller--Madow MI across pairs, in nats.
        worst_unknown: Unknown label of the worst pair.
        worst_known: Known label of the worst pair.
        ci_low: Lower bootstrap CI bound on the worst pair (nats).
        ci_high: Upper bootstrap CI bound on the worst pair (nats).
        pvalue: Permutation p-value of the worst pair.
        n_violations: Pairs with MI > ``threshold``.
        sweep: Violation counts at a few reference thresholds.
        pairs: Full per-pair table (descending MI).
    """

    passed: bool
    threshold: float
    n_pairs: int
    max_mi: float
    worst_unknown: str
    worst_known: str
    ci_low: float
    ci_high: float
    pvalue: float
    n_violations: int
    sweep: dict[float, int] = field(default_factory=dict)
    pairs: list[dict] = field(default_factory=list)
    units: str = "nats"
    ci_level: float = 0.95

    @property
    def _ci_pct(self) -> str:
        pct = self.ci_level * 100
        return f"{pct:.0f}" if pct == round(pct) else f"{pct:g}"

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the report."""
        return {
            "passed": self.passed,
            "threshold": self.threshold,
            "units": self.units,
            "n_pairs": self.n_pairs,
            "max_mi": self.max_mi,
            "worst_pair": {"unknown": self.worst_unknown, "known": self.worst_known},
            "worst_pair_ci": [self.ci_low, self.ci_high],
            "worst_pair_ci_level": self.ci_level,
            "worst_pair_pvalue": self.pvalue,
            "n_violations": self.n_violations,
            "threshold_sweep": self.sweep,
            "pairs": self.pairs,
        }

    def summary(self) -> str:
        """One-line human-readable verdict."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] max I_MM = {self.max_mi:.4f} {self.units} "
            f"({self.worst_unknown} <-> {self.worst_known}, "
            f"{self._ci_pct}% CI [{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"per-pair p={self.pvalue:.3f}); "
            f"{self.n_violations}/{self.n_pairs} pairs exceed {self.threshold} {self.units}"
        )

    def to_markdown(self, top: int = 5) -> str:
        """Markdown report with the verdict and the top-``top`` pairs."""
        lines = [
            f"# Split leakage audit --- {'PASS' if self.passed else 'FAIL'}",
            "",
            f"- Threshold: {self.threshold} {self.units}",
            f"- Pairs audited: {self.n_pairs}",
            f"- Max MI: **{self.max_mi:.4f} {self.units}** "
            f"({self.worst_unknown} <-> {self.worst_known})",
            f"- Worst-pair {self._ci_pct}% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}]; "
            f"per-pair permutation p = {self.pvalue:.3f}",
            f"- Violations (MI > {self.threshold}): {self.n_violations}",
            "",
            "| Threshold | Violating pairs |",
            "|---|---|",
        ]
        for t, c in self.sweep.items():
            lines.append(f"| {t} | {c} |")
        lines += ["", f"## Top {top} pairs by MI", "",
                  "| Unknown | Known | MI (nats) |", "|---|---|---|"]
        for p in self.pairs[:top]:
            lines.append(f"| {p['unknown']} | {p['known']} | {p['mi']:.4f} |")
        return "\n".join(lines)


def audit_split(
    labels: dict[str, np.ndarray],
    known: Sequence[str],
    unknown: Sequence[str],
    *,
    threshold: float = 0.05,
    n_boot: int = 1000,
    n_perm: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    sweep_thresholds: Sequence[float] = (0.03, 0.05, 0.07),
) -> AuditReport:
    """Audit a multi-label open-set split for known<->unknown leakage.

    Computes the Miller--Madow MI for every (unknown, known) pair, attaches a
    paired bootstrap CI and a permutation p-value to the worst pair, and
    returns a PASS/FAIL verdict against ``threshold``.

    Args:
        labels: Mapping from label name to aligned per-sample binary vector.
        known: Known-class label names (exclude any "no finding" head).
        unknown: Held-out (unknown) class label names.
        threshold: Leakage cutoff in nats; the split passes iff no pair
            exceeds it.
        n_boot: Bootstrap resamples for the worst-pair CI.
        n_perm: Permutations for the worst-pair p-value.
        ci: Central confidence level for the worst-pair bootstrap CI.
        seed: RNG seed for reproducibility.
        sweep_thresholds: Extra thresholds for the violation-count sweep;
            ``threshold`` is always included so the report is self-consistent.

    Returns:
        An :class:`AuditReport`.

    Raises:
        ValueError: If ``known`` or ``unknown`` is empty, overlap, or a name
            is missing from ``labels``.

    Note:
        The CI and p-value describe the **single worst pair**, selected as the
        maximum over all pairs. They are *not* family-wise corrected: the
        per-pair p-value answers "is this pair's co-occurrence significant on
        its own", not "is the largest of all pairs surprising under global
        independence". With many pairs the worst per-pair p is optimistic;
        read it as a diagnostic for the named pair, and rely on the
        ``threshold`` verdict (which scans every pair) for the overall call.
    """
    if not known or not unknown:
        raise ValueError("both `known` and `unknown` must be non-empty")
    missing = [c for c in list(known) + list(unknown) if c not in labels]
    if missing:
        raise ValueError(f"labels missing for: {missing}")
    overlap = sorted(set(known) & set(unknown))
    if overlap:
        raise ValueError(
            f"label(s) appear in both `known` and `unknown`: {overlap}; a class "
            "cannot be simultaneously known and held-out (this would report a "
            "spurious self-information leak)"
        )
    degenerate = [c for c in list(known) + list(unknown)
                  if len(np.unique(labels[c])) < 2]
    if degenerate:
        warnings.warn(
            f"label(s) take a single value (no positives or no negatives): "
            f"{degenerate}. Their mutual information is 0 by construction and "
            "they cannot leak or be leaked --- check these are the right columns.",
            stacklevel=2,
        )

    pairs = pairwise_mi(labels, known, unknown)
    worst = pairs[0]
    a, b = labels[worst["unknown"]], labels[worst["known"]]
    ci_low, ci_high = bootstrap_ci(a, b, n_boot=n_boot, ci=ci, seed=seed)
    pval = permutation_pvalue(a, b, n_perm=n_perm, seed=seed)
    sweep = threshold_sweep(pairs, sorted(set(sweep_thresholds) | {threshold}))
    n_viol = int(sum(1 for p in pairs if p["mi"] > threshold))

    return AuditReport(
        passed=n_viol == 0,
        threshold=threshold,
        n_pairs=len(pairs),
        max_mi=float(worst["mi"]),
        worst_unknown=str(worst["unknown"]),
        worst_known=str(worst["known"]),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci,
        pvalue=pval,
        n_violations=n_viol,
        sweep=sweep,
        pairs=pairs,
    )
