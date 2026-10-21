'''
Lithology: M-N and matrix (MID) parameters, and multi-mineral volumes

'''
import os
import numpy as np
import pandas as pd

from ._utils import as_array, like

#expected log errors, used to weight the logs in `multimineral`
DEFAULT_UNCERTAINTY = {'RHOB': .025, 'NPHI': .015, 'DT': 2., 'PEF': .15, 'GR': 5., 'U': .5}


def minerals() -> pd.DataFrame:

    r'''
    Typical log responses of common minerals and fresh water: RHOB (g/cc), NPHI (fraction,
    limestone units), DT (us/ft) and PEF (b/e). Tool and chart values vary; edit them for your
    tools before relying on them

    Example
    -------
    >>> minerals().loc['Quartz', 'RHOB']
    2.65
    '''

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'minerals.csv')
    return pd.read_csv(path, index_col='mineral')


def m_n(rhob, nphi, dt, rho_fluid:float=1., nphi_fluid:float=1., dt_fluid:float=189.):

    r'''
    M and N lithology parameters, independent of porosity

        M = 0.01 (dt_fluid - DT) / (RHOB - rho_fluid)
        N = (nphi_fluid - NPHI) / (RHOB - rho_fluid)

    Returns
    -------
    (M, N)

    Example
    -------
    >>> M, N = m_n(df['RHOB'], df['NPHI'], df['DT'])
    '''

    rho = as_array(rhob)
    with np.errstate(divide='ignore', invalid='ignore'):
        m = .01 * (dt_fluid - as_array(dt)) / (rho - rho_fluid)
        n = (nphi_fluid - as_array(nphi)) / (rho - rho_fluid)
    return like(m, rhob, nphi, dt), like(n, rhob, nphi, dt)


def rhomaa(rhob, phit, rho_fluid:float=1.):

    r'''
    Apparent matrix density, RHOMAA = (RHOB - PHIT rho_fluid) / (1 - PHIT)
    '''

    phi = as_array(phit)
    return like((as_array(rhob) - phi * rho_fluid) / (1 - phi), rhob, phit)


def dtmaa(dt, phit, dt_fluid:float=189.):

    r'''
    Apparent matrix slowness, DTMAA = (DT - PHIT dt_fluid) / (1 - PHIT)
    '''

    phi = as_array(phit)
    return like((as_array(dt) - phi * dt_fluid) / (1 - phi), dt, phit)


def umaa(pef, rhob, phit, u_fluid:float=.398):

    r'''
    Apparent matrix volumetric photoelectric factor (barns/cc)

        U = PEF rho_e,  rho_e = (RHOB + 0.1883) / 1.0704
        UMAA = (U - PHIT u_fluid) / (1 - PHIT)

    Parameters
    ----------
    pef, rhob, phit : array-like
        Photoelectric factor (b/e), bulk density (g/cc) and total porosity

    u_fluid : float default 0.398
        U of the pore fluid (fresh water 0.398)
    '''

    phi = as_array(phit)
    u = as_array(pef) * (as_array(rhob) + .1883) / 1.0704
    return like((u - phi * u_fluid) / (1 - phi), pef, rhob, phit)


def multimineral(df:pd.DataFrame, components:dict, logs:list=None, uncertainties:dict=None,
                 unity_weight:float=10.) -> pd.DataFrame:

    r'''
    Volumes of minerals and fluids from several logs, by constrained least squares on each sample

        log = sum(V_i * response_i),  sum(V_i) = 1,  0 <= V_i <= 1

    Parameters
    ----------
    df : pd.DataFrame
        Log data

    components : dict
        Response of each mineral or fluid to each log, e.g.::

            {'Quartz': {'RHOB': 2.65, 'NPHI': -.02, 'GR': 20},
             'Illite': {'RHOB': 2.52, 'NPHI': .30, 'GR': 150},
             'Water':  {'RHOB': 1.00, 'NPHI': 1.00, 'GR': 0}}

        Use at most one more component than logs. `minerals()` has typical mineral values

    logs : list default None
        Logs to use. Default: the columns of `df` that every component has a response for

    uncertainties : dict default None
        Expected error of each log, to weight them. Defaults from DEFAULT_UNCERTAINTY, else 1

    unity_weight : float default 10.
        Weight of the sum(V) = 1 constraint

    Returns
    -------
    DataFrame with one volume column per component and MM_ERROR, the weighted RMS misfit of the
    logs (around 1 or less when the model fits within the log errors). NaN where there are too few logs

    Example
    -------
    >>> mm = minerals()
    >>> comps = {m: mm.loc[m, ['RHOB', 'NPHI', 'DT']].to_dict() for m in ['Quartz', 'Calcite', 'Dolomite', 'Water']}
    >>> volumes = multimineral(df, comps)
    '''

    from scipy.optimize import lsq_linear

    names = list(components)
    if logs is None:
        logs = [c for c in df.columns if all(c in r for r in components.values())]
    if not logs:
        raise ValueError('No log of `df` has a response for every component')
    errors = {**DEFAULT_UNCERTAINTY, **(uncertainties or {})}
    w = np.array([1 / errors.get(l, 1.) for l in logs])
    A = np.array([[components[c][l] for c in names] for l in logs], dtype=float)

    X = df[logs].to_numpy(dtype=float)
    volumes = np.full((len(df), len(names)), np.nan)
    misfit = np.full(len(df), np.nan)
    for i, row in enumerate(X):
        ok = np.isfinite(row)
        if ok.sum() + 1 < len(names):
            continue
        Aw = np.vstack([A[ok] * w[ok, None], np.full(len(names), unity_weight)])
        bw = np.concatenate([row[ok] * w[ok], [unity_weight]])
        x = lsq_linear(Aw, bw, bounds=(0, 1)).x
        volumes[i] = x
        misfit[i] = np.sqrt(np.mean(((A[ok] @ x - row[ok]) * w[ok])**2))

    out = pd.DataFrame(volumes, columns=names, index=df.index)
    out['MM_ERROR'] = misfit
    return out
