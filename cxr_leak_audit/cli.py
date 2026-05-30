"""Command-line interface: ``cxr-leak-audit``.

Reads a CSV of binary label columns and audits a known/unknown split for
co-occurrence leakage. Exit code is ``0`` on PASS and ``1`` on FAIL, so the
command can gate a CI pipeline or a data-release script.

Example:
    cxr-leak-audit labels.csv --known atelectasis,edema,effusion \\
        --unknown fracture,pneumothorax --threshold 0.05 --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import __version__
from .audit import audit_split


def _split_csv(arg: str) -> list[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cxr-leak-audit",
        description="Information-theoretic leakage audit for open-set splits.",
    )
    p.add_argument("labels", help="CSV with one binary column per label.")
    p.add_argument("--known", required=True, type=_split_csv,
                   help="Comma-separated known-class column names.")
    p.add_argument("--unknown", required=True, type=_split_csv,
                   help="Comma-separated unknown-class column names.")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Leakage threshold in nats (default: 0.05).")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples for the worst-pair CI (default: 1000).")
    p.add_argument("--permutations", type=int, default=1000,
                   help="Permutations for the worst-pair p-value (default: 1000).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    p.add_argument("--id-col", default=None,
                   help="Name of a non-label id column to ignore (e.g. image_id).")
    p.add_argument("--json", default=None, metavar="PATH",
                   help="Write the full report as JSON to PATH.")
    p.add_argument("--markdown", default=None, metavar="PATH",
                   help="Write a Markdown report to PATH.")
    p.add_argument("--version", action="version",
                   version=f"cxr-leak-audit {__version__}")
    return p


def _load_labels(path: str, columns: list[str], id_col: str | None) -> dict:
    df = pd.read_csv(path)
    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"error: column(s) not found in {path}: {missing}\n"
            f"available: {list(df.columns)}"
        )
    labels = {}
    for c in columns:
        v = df[c].to_numpy()
        uniq = set(np.unique(v).tolist())
        if not uniq <= {0, 1}:
            raise SystemExit(
                f"error: column {c!r} is not binary 0/1 (saw {sorted(uniq)[:5]}...)"
            )
        labels[c] = v.astype(int)
    return labels


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = _load_labels(args.labels, args.known + args.unknown, args.id_col)
    report = audit_split(
        labels, known=args.known, unknown=args.unknown,
        threshold=args.threshold, n_boot=args.bootstrap,
        n_perm=args.permutations, seed=args.seed,
    )
    print(report.summary())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"wrote JSON report -> {args.json}")
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(report.to_markdown())
        print(f"wrote Markdown report -> {args.markdown}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
