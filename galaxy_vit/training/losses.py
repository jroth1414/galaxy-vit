"""Class-balanced loss reweighting (Cui et al. 2019).

Used by T1.5 to upweight Galaxy10's small classes (class 4 has only ~234
train samples vs class 2's ~1851). Pure-stdlib weight computation; the
trainer applies the result through ``torch.nn.functional.cross_entropy``'s
built-in ``weight`` parameter — no custom autograd required.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_BETA = 0.9999


def class_balanced_weights(
    class_counts: Sequence[int],
    *,
    beta: float = DEFAULT_BETA,
) -> list[float]:
    """Effective-number-of-samples reweighting (Cui+19 Eq. 2).

    For each class ``c`` with ``n_c`` training samples::

        effective_n_c = (1 - beta**n_c) / (1 - beta)
        weight_c      = 1 / effective_n_c

    The resulting weights are normalised so they sum to ``len(class_counts)``,
    keeping the loss magnitude in roughly the same range as unweighted CE
    (which makes LR transfer between weighted / unweighted runs reasonable).

    Parameters
    ----------
    class_counts:
        Per-class sample counts from the training set, in label order.
    beta:
        Effective-number hyperparameter in (0, 1). Cui+19 recommends
        0.9, 0.99, 0.999, 0.9999 with the latter giving the strongest
        rebalancing. DEVPLAN T1.5 prescribes beta=0.9999.

    Returns
    -------
    Plain-Python floats, length ``len(class_counts)``, summing to
    ``len(class_counts)``.

    Raises
    ------
    ValueError
        If beta is not strictly in (0, 1) or any count is non-positive.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1); got {beta}")
    if any(n <= 0 for n in class_counts):
        raise ValueError(f"all class_counts must be positive; got {list(class_counts)}")

    effective = [(1.0 - beta**n) / (1.0 - beta) for n in class_counts]
    raw = [1.0 / e for e in effective]
    k = len(class_counts)
    s = sum(raw)
    return [w * k / s for w in raw]
