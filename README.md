# cxr-leak-audit

[![CI](https://github.com/hxtruong6/cxr-leak-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/hxtruong6/cxr-leak-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hxtruong6/cxr-leak-audit/blob/main/LICENSE)
[![Python versions](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue.svg)](https://github.com/hxtruong6/cxr-leak-audit)

**Information-theoretic co-occurrence leakage audit for multi-label open-set splits — one number, one verdict, before you train anything.**

## Why cxr-leak-audit?

In multi-label open-set recognition (MLOSR) — chest X-rays being the canonical
case — a held-out "unknown" pathology often **co-occurs** with a "known" one
(e.g. *pleural thickening* with *pleural effusion*). A model can then flag the
unknown by memorising that correlation instead of detecting novelty, and the
reported open-set ranking measures *contamination*, not *detection*. Most
benchmarks never check for this.

`cxr-leak-audit` makes the check a one-liner. For every (unknown, known) label
pair it computes the **Miller–Madow bias-corrected mutual information** (in
nats), attaches a **paired bootstrap 95% CI** and a **permutation p-value** to
the worst pair, and returns a **PASS/FAIL** verdict against a leakage threshold
(default `0.05` nats).

- **Framework-agnostic** — `numpy`/`pandas` in, a verdict out. No PyTorch, no datasets, no model.
- **A-priori guarantee** — a passing split bounds the harvestable co-occurrence signal *before any model is trained*.
- **Reproducible** — every estimate is seeded; bootstrap is paired, permutation null breaks the pairing.
- **CI-friendly** — the CLI exits `1` on FAIL, so it gates a data-release script or a benchmark PR.

> Not chest-X-ray specific despite the name: it audits any multi-label
> open-set split given binary label columns.

## Install

```bash
pip install cxr-leak-audit
```

Requires Python 3.10+, `numpy`, `pandas`, `scikit-learn`.

### Development install

```bash
git clone https://github.com/hxtruong6/cxr-leak-audit.git
cd cxr-leak-audit
pip install -e .[dev]
```

## Quick start (library)

```python
import numpy as np
from cxr_leak_audit import audit_split

# Binary label vectors, all aligned to the same N rows (your evaluation split).
labels = {
    "effusion":  effusion_vec,    # known
    "atelectasis": atel_vec,      # known
    "pneumothorax": ptx_vec,      # unknown (held out)
    "fracture":  fracture_vec,    # unknown
}

report = audit_split(
    labels,
    known=["effusion", "atelectasis"],
    unknown=["pneumothorax", "fracture"],
    threshold=0.05,     # nats
)

print(report.summary())
# [PASS] max I_MM = 0.0123 nats (fracture <-> effusion, 95% CI [...], p=0.31); 0/4 pairs exceed 0.05 nats

report.passed        # bool
report.max_mi        # worst-pair Miller–Madow MI, nats
report.to_dict()     # JSON-serializable full report (per-pair table included)
report.to_markdown() # human-readable report
```

## Quick start (CLI)

```bash
cxr-leak-audit labels.csv \
    --known atelectasis,edema,effusion \
    --unknown fracture,pneumothorax \
    --threshold 0.05 \
    --id-col image_id \
    --json report.json
```

`labels.csv` has one binary (`0/1`) column per label (plus an optional id
column). The command prints a one-line verdict and **exits `0` on PASS, `1`
on FAIL** — drop it into CI to block a leaky split from being merged:

```yaml
- run: cxr-leak-audit splits/test.csv --known $KNOWN --unknown $UNKNOWN
```

## What it computes

For known/unknown indicator vectors, the audit statistic is the Miller–Madow
bias-corrected mutual information in nats:

```
I_MM(a, b) = I_plugin(a, b) − (k_a − 1)(k_b − 1) / (2n)
```

| Function | Purpose |
|---|---|
| `miller_madow_mi(a, b)` | Bias-corrected MI between two binary label vectors (nats). |
| `pairwise_mi(labels, known, unknown)` | MI for every (unknown, known) pair, sorted. |
| `bootstrap_ci(a, b)` | Paired bootstrap CI on the MI of one pair. |
| `permutation_pvalue(a, b)` | Permutation-null p-value (independence). |
| `threshold_sweep(pairs, thresholds)` | Violating-pair counts at several thresholds. |
| `audit_split(...)` | One call → `AuditReport` (verdict, worst pair, CI, p, sweep, table). |

## Why these choices

- **Miller–Madow**, not plug-in: the naive MI estimate is biased upward at
  finite N, which would manufacture phantom leakage. The correction removes the
  leading bias term.
- **Nats**, not bits: matches the information-theoretic convention; `0.05` nats
  is roughly 1/14 of the `ln 2` uncertainty of one balanced binary label.
- **Pairwise**: a tractable, interpretable operationalization of leakage. It
  does **not** detect higher-order (multi-label coalition) synergy — a known
  limitation; treat a pass as necessary, not sufficient.
- **Worst-pair CI/p are per-pair, not family-wise.** The bootstrap CI and
  permutation p-value describe the single maximum pair, selected post hoc;
  they are not corrected for having scanned every pair. Use them as a
  diagnostic for the named pair, and rely on the `threshold` verdict (which
  scans all pairs) for the overall PASS/FAIL call.

## Reproducibility

The estimator reproduces the audit numbers reported in the accompanying MLOSR-CXR
benchmark exactly (e.g. CheXpert worst pair `0.002297` nats, VinBigData worst
pair `0.041245` nats), given the same locked splits.

## Testing

```bash
pytest tests/ -v
```

Each statistic is verified from first principles: Miller–Madow MI against a
hand-built contingency table, independence → MI ≈ 0, identical labels →
`H − 1/(2n)`, leaky split → FAIL with the planted pair flagged.

## Citation

```bibtex
@software{cxr_leak_audit,
  author = {Hoang Xuan Truong},
  title  = {cxr-leak-audit: Information-Theoretic Leakage Audit for Open-Set Splits},
  url    = {https://github.com/hxtruong6/cxr-leak-audit},
  year   = {2026}
}
```

## License

MIT.
