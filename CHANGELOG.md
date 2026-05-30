# Changelog

All notable changes to `cxr-leak-audit` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] — 2026-05-31

### Fixed

- `audit_split` now raises `ValueError` when a label appears in both `known`
  and `unknown`. Previously the self-pair `MI(x, x) = H(x)` produced a
  spurious leakage FAIL from an overlapping argument list.
- Corrected the numeric value in the `miller_madow_mi` docstring example
  (`0.5683` → `0.5681`; `ln 2 − 1/(2·4)`), and added a test that locks it.

### Added

- `audit_split` warns when any known/unknown label is single-valued (no
  positives or no negatives): its MI is 0 by construction and it cannot leak
  or be leaked, almost always signalling a wrong column.

## [0.1.0] — 2026-05-31

Initial release: the Miller–Madow co-occurrence leakage audit for multi-label
open-set splits, as a library and a CLI.

### Added

- `miller_madow_mi(a, b)` — Miller–Madow bias-corrected mutual information
  between two binary label vectors, in nats.
- `pairwise_mi`, `bootstrap_ci` (paired), `permutation_pvalue`,
  `threshold_sweep` — the audit building blocks.
- `audit_split(...)` — one-call audit returning an `AuditReport`
  (PASS/FAIL verdict, worst pair, bootstrap CI, permutation p-value,
  threshold sweep, full per-pair table) with `to_dict` / `to_markdown` /
  `summary` renderers.
- `cxr-leak-audit` console script — audits a CSV of binary label columns;
  exits `0` on PASS and `1` on FAIL for use in CI / release gating.
- Test suite verifying every statistic from first principles; verified to
  reproduce the accompanying MLOSR-CXR benchmark's audit numbers exactly.
