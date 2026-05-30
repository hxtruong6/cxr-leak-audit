"""End-to-end CLI tests (exit codes, JSON output)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cxr_leak_audit.cli import main


def _write_csv(path, leaky: bool, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    k1 = rng.integers(0, 2, n)
    u1 = (np.where(rng.random(n) < 0.1, 1 - k1, k1) if leaky
          else rng.integers(0, 2, n))
    df = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)],
        "k1": k1,
        "k2": rng.integers(0, 2, n),
        "u1": u1,
    })
    df.to_csv(path, index=False)


def test_cli_pass_returns_zero(tmp_path):
    csv = tmp_path / "clean.csv"
    _write_csv(csv, leaky=False)
    code = main([str(csv), "--known", "k1,k2", "--unknown", "u1",
                 "--id-col", "image_id", "--bootstrap", "100",
                 "--permutations", "100"])
    assert code == 0


def test_cli_fail_returns_one_and_writes_json(tmp_path):
    csv = tmp_path / "leaky.csv"
    _write_csv(csv, leaky=True)
    out = tmp_path / "report.json"
    code = main([str(csv), "--known", "k1,k2", "--unknown", "u1",
                 "--id-col", "image_id", "--bootstrap", "100",
                 "--permutations", "100", "--json", str(out)])
    assert code == 1
    report = json.loads(out.read_text())
    assert report["passed"] is False
    assert report["worst_pair"] == {"unknown": "u1", "known": "k1"}


def test_cli_nonbinary_column_errors(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"k1": [0, 1, 2], "u1": [0, 1, 0]}).to_csv(csv, index=False)
    import pytest
    with pytest.raises(SystemExit):
        main([str(csv), "--known", "k1", "--unknown", "u1"])
