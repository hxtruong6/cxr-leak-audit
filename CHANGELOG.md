# Changelog

All notable changes to `cxr-leak-audit` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
