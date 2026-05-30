"""cxr-leak-audit: information-theoretic leakage audit for open-set splits.

Screens a multi-label open-set recognition split for known<->unknown
co-occurrence leakage using the Miller--Madow mutual-information audit:
per-pair bias-corrected MI (in nats), a paired bootstrap CI and a permutation
null on the worst pair, and a PASS/FAIL verdict against a leakage threshold
(default 0.05 nats).

A passing split guarantees that, before any model is trained, the harvestable
co-occurrence signal between every known and unknown class is bounded --- so a
reported open-set ranking reflects detection rather than contamination.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .mi import miller_madow_mi
from .audit import (
    AuditReport,
    audit_split,
    bootstrap_ci,
    pairwise_mi,
    permutation_pvalue,
    threshold_sweep,
)

try:
    __version__ = _pkg_version("cxr-leak-audit")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [
    # Core statistic
    "miller_madow_mi",
    # Audit building blocks
    "pairwise_mi",
    "bootstrap_ci",
    "permutation_pvalue",
    "threshold_sweep",
    # One-call audit
    "audit_split",
    "AuditReport",
]
