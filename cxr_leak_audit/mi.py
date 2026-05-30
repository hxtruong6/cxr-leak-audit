"""Miller--Madow bias-corrected mutual information for binary label vectors.

The audit statistic is the mutual information between a known-class indicator
and an unknown-class indicator, in **nats**, with the Miller--Madow correction
subtracted to remove the upward (finite-sample) bias of the plug-in estimator:

    I_MM(a, b) = I_plugin(a, b) - (k_a - 1)(k_b - 1) / (2n)

where ``k_a``/``k_b`` are the number of distinct values each vector takes
(2 for a non-degenerate binary label) and ``n`` is the sample size. The plug-in
term is :func:`sklearn.metrics.mutual_info_score`, which returns nats.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mutual_info_score


def miller_madow_mi(a: np.ndarray, b: np.ndarray) -> float:
    """Miller--Madow bias-corrected mutual information, in nats.

    Args:
        a: Per-sample label vector, shape ``[N]`` (typically binary 0/1).
        b: Per-sample label vector, shape ``[N]``, aligned with ``a``.

    Returns:
        The bias-corrected mutual information in nats. May be slightly
        negative when the two labels are independent (the correction can
        exceed the small positive plug-in estimate); this is expected.

    Raises:
        ValueError: If ``a`` and ``b`` have different lengths or are empty.

    Example:
        >>> import numpy as np
        >>> a = np.array([0, 0, 1, 1])
        >>> float(round(miller_madow_mi(a, a), 4))  # identical -> ln2 - 1/(2n)
        0.5681
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"a and b must have the same length; got {a.shape[0]} and {b.shape[0]}"
        )
    n = a.shape[0]
    if n == 0:
        raise ValueError("a and b must be non-empty")
    mi = float(mutual_info_score(a, b))
    k_a = len(np.unique(a))
    k_b = len(np.unique(b))
    return float(mi - (k_a - 1) * (k_b - 1) / (2 * n))
