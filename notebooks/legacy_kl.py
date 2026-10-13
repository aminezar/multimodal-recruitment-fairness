"""New vectorized implementation of the thesis's historical score-bin formula.

This is a compatibility estimator, NOT a general KL divergence: it omits bins
where either group has zero mass and ignores scores outside [0, .98).
Keep these conventions for interpreting the archived results. No upstream
implementation is bundled. Written during publication preparation, not an
original thesis contribution.
"""
import numpy as np


def computeKL(x, y):
    """Male-to-female historical score-bin KL, in natural-log units."""
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if not x.size or not y.size:
        raise ValueError('Both score groups must be nonempty')
    lower = np.arange(49, dtype=float) / 50
    upper = lower + .02
    mass_x = ((x[:, None] >= lower) & (x[:, None] < upper)).sum(axis=0) / x.size
    mass_y = ((y[:, None] >= lower) & (y[:, None] < upper)).sum(axis=0) / y.size
    shared = (mass_x > 0) & (mass_y > 0)
    return float(np.sum(mass_x[shared] * np.log(mass_x[shared] / mass_y[shared])))
