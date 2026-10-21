'''
Monte Carlo uncertainty of water saturation

'''
import numpy as np
import pandas as pd

from ._utils import as_array
from .saturation import archie


def _draw(value, size, rng):

    '''
    Samples of a parameter: a float is fixed, (low, mode, high) is a triangular distribution
    '''

    if np.ndim(value) == 0:
        return np.full(size, float(value))
    low, mode, high = value
    return rng.triangular(low, mode, high, size)


def monte_carlo_archie(phie, rt, rw=.03, a=1., m=2., n=2., n_iter:int=1000, seed:int=None) -> pd.DataFrame:

    r'''
    Spread of Archie water saturation from uncertain parameters

    Each parameter is a float (fixed) or a (low, most likely, high) tuple sampled from a
    triangular distribution. Each iteration draws one set of parameters for all samples

    Parameters
    ----------
    phie, rt : array-like
        Porosity and true resistivity

    rw, a, m, n : float or tuple
        Archie parameters, e.g. rw=(.02, .03, .045), m=(1.8, 2., 2.2)

    n_iter : int default 1000
        Number of iterations

    seed : int default None
        Random seed, for repeatable results

    Returns
    -------
    DataFrame with SW_P10, SW_P50 and SW_P90, the 10th, 50th and 90th percentiles of SW per sample

    Example
    -------
    >>> mc = monte_carlo_archie(df['PHIE'], df['RT'], rw=(.02, .03, .045), m=(1.8, 2, 2.2), seed=0)
    '''

    rng = np.random.default_rng(seed)
    p = {k: _draw(v, n_iter, rng) for k, v in dict(rw=rw, a=a, m=m, n=n).items()}
    phi, r = as_array(phie), as_array(rt)
    sw = np.stack([as_array(archie(phi, r, p['rw'][i], p['a'][i], p['m'][i], p['n'][i])) for i in range(n_iter)])
    q = np.nanpercentile(sw, [10, 50, 90], axis=0)
    index = phie.index if isinstance(phie, pd.Series) else None
    return pd.DataFrame({'SW_P10': q[0], 'SW_P50': q[1], 'SW_P90': q[2]}, index=index)
