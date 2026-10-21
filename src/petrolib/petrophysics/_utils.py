'''
Helpers shared by the petrophysics equations

'''
import numpy as np
import pandas as pd


def as_array(x) -> np.ndarray:

    '''
    Float numpy array of `x` (Series, array, list or scalar)
    '''

    return np.asarray(x, dtype=float)


def like(values, *refs):

    '''
    Returns `values` as a Series with the index of the first Series in `refs`.
    Otherwise returns an array, or a float for scalar inputs
    '''

    for r in refs:
        if isinstance(r, pd.Series):
            return pd.Series(values, index=r.index)
    return float(values) if np.ndim(values) == 0 else values


def solve_increasing(f, lo, hi=1., n_iter:int=60) -> np.ndarray:

    '''
    Element-wise root of `f`, a function increasing in x on [lo, hi], found by bisection.
    Where `f` has no root in the interval the nearest bound is returned. NaN where `f` is NaN
    '''

    lo = np.array(lo, dtype=float) * np.ones_like(np.asarray(f(np.asarray(hi, dtype=float)), dtype=float))
    hi = np.full_like(lo, hi)
    nan = np.isnan(f(.5 * (lo + hi)))
    for _ in range(n_iter):
        mid = .5 * (lo + hi)
        below = f(mid) < 0
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    x = .5 * (lo + hi)
    x[nan] = np.nan
    return x
