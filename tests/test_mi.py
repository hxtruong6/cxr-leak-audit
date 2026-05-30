"""First-principles verification of the Miller--Madow MI estimator."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cxr_leak_audit import miller_madow_mi


def test_identical_vectors_equal_entropy_minus_correction():
    # For identical balanced binary vectors, plug-in MI = H = ln 2; the
    # Miller--Madow correction is (2-1)(2-1)/(2n) = 1/(2n).
    n = 100
    a = np.array([0, 1] * (n // 2))
    expected = math.log(2) - 1 / (2 * n)
    assert miller_madow_mi(a, a) == pytest.approx(expected, abs=1e-9)


def test_independent_vectors_near_zero():
    # Independent labels: plug-in MI is a small positive bias; Miller--Madow
    # subtracts it, leaving a value close to zero (possibly slightly negative).
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 5000)
    b = rng.integers(0, 2, 5000)
    assert abs(miller_madow_mi(a, b)) < 0.01


def test_constant_vector_zero_mi():
    # A degenerate (constant) vector carries no information.
    a = np.array([1, 0, 1, 1, 0, 1])
    const = np.ones(6, dtype=int)
    assert miller_madow_mi(a, const) == pytest.approx(0.0, abs=1e-12)


def test_known_contingency_value():
    # Hand-built 2x2 contingency: a,b agree on 40+40, disagree on 10+10 (n=100).
    # plug-in MI computed from the joint/marginals, then minus 1/(2n).
    a = np.array([0] * 50 + [1] * 50)
    b = np.array([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)
    n = 100
    # marginals: a -> 50/50; b -> 50/50; joint 00=40,01=10,10=10,11=40
    def term(j, pa, pb):
        return (j / n) * math.log((j / n) / (pa * pb)) if j else 0.0
    plugin = (term(40, .5, .5) + term(10, .5, .5)
              + term(10, .5, .5) + term(40, .5, .5))
    expected = plugin - 1 / (2 * n)
    assert miller_madow_mi(a, b) == pytest.approx(expected, abs=1e-9)


def test_documented_example_value():
    # Locks the value shown in the miller_madow_mi docstring (n=4, identical
    # balanced binary -> ln2 - 1/(2*4) = 0.5681).
    a = np.array([0, 0, 1, 1])
    assert round(float(miller_madow_mi(a, a)), 4) == 0.5681


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        miller_madow_mi(np.array([0, 1]), np.array([0, 1, 0]))


def test_empty_raises():
    with pytest.raises(ValueError):
        miller_madow_mi(np.array([]), np.array([]))
