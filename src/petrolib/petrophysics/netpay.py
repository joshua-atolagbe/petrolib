'''
Net pay flags, summation and cutoff sensitivity

'''
import numpy as np
import pandas as pd

from ._utils import as_array


def net_flags(vsh, phie, sw, vsh_cutoff:float, por_cutoff:float, sw_cutoff:float) -> pd.DataFrame:

    r'''
    ROCK, RES and PAY flags (1 net, 0 not net). Each flag is a subset of the one before:

        ROCK : Vsh < vsh_cutoff
        RES  : ROCK and PHIE > por_cutoff
        PAY  : RES  and SW < sw_cutoff

    Example
    -------
    >>> flags = net_flags(df['VShale'], df['PHIE'], df['SW'], .5, .1, .6)
    '''

    v, p, s = as_array(vsh), as_array(phie), as_array(sw)
    rock = v < vsh_cutoff
    res = rock & (p > por_cutoff)
    pay = res & (s < sw_cutoff)
    index = vsh.index if isinstance(vsh, pd.Series) else None
    return pd.DataFrame({'ROCK_NET_FLAG': rock.astype(int), 'RES_NET_FLAG': res.astype(int),
                         'PAY_NET_FLAG': pay.astype(int)}, index=index)


def sample_thickness(depth:pd.Series, top:float, bot:float) -> pd.Series:

    r'''
    Thickness each sample represents: its depth +/- half the sampling step, clipped to the zone
    [top, bot]. Gaps in the data carry no thickness, so they count as not net
    '''

    depth = pd.Series(depth)
    if len(depth) < 2:
        return pd.Series(0., index=depth.index)
    step = np.median(np.diff(np.sort(depth.to_numpy())))
    return ((depth + step/2).clip(upper=bot) - (depth - step/2).clip(lower=top)).clip(lower=0)


def weighted_mean(values, weights) -> float:

    r'''
    Weighted mean that skips NaN values. NaN if there is no weight
    '''

    values, weights = pd.Series(values), pd.Series(weights, index=pd.Series(values).index)
    ok = values.notna()
    w = weights[ok].sum()
    return (values[ok] * weights[ok]).sum() / w if w > 0 else np.nan


def summarize(depth, vsh, phie, sw, flag, top:float, bot:float, perm=None) -> dict:

    r'''
    Summation of one flag over a zone, as in Techlog Cutoff and Summation

    Returns
    -------
    dict with Gross, Net, NTG, average VShale and PHIE (thickness weighted), average SW
    (pore volume weighted), BVW, permeability (thickness weighted), PHIH (PHIE h) and
    HCPV (PHIE (1 - SW) h)
    '''

    h = sample_thickness(depth, top, bot)
    on = pd.Series(as_array(flag) == 1, index=h.index)
    hf = h[on]
    s = lambda x: pd.Series(as_array(x), index=h.index)[on]
    phi, swf = s(phie), s(sw)
    gross = bot - top
    net = hf.sum()
    out = {'Gross': gross, 'Net': net, 'Not Net': gross - net, 'NTG': net / gross if gross > 0 else 0.,
           'BVW': weighted_mean(swf * phi, hf),
           'Average VShale': weighted_mean(s(vsh), hf),
           'Average Porosity': weighted_mean(phi, hf),
           'Average Water Saturation': weighted_mean(swf, phi * hf),
           'Average Permeability': weighted_mean(s(perm), hf) if perm is not None else np.nan,
           'PHIH': (phi * hf).sum(),
           'HCPV': (phi * (1 - swf) * hf).sum()}
    return out


def cutoff_sensitivity(df:pd.DataFrame, cutoff:str='por', values=None, depth:str='DEPTH', vsh:str='VShale',
                       phie:str='PHIE', sw:str='SW', vsh_cutoff:float=.5, por_cutoff:float=.06,
                       sw_cutoff:float=.8, top:float=None, bot:float=None) -> pd.DataFrame:

    r'''
    Net pay summation as one cutoff varies and the other two stay fixed, to choose cutoffs

    Parameters
    ----------
    df : pd.DataFrame
        Data of one zone with depth, Vsh, PHIE and SW columns

    cutoff : str default 'por'
        Cutoff to vary: 'vsh', 'por' or 'sw'

    values : array-like default None
        Values of that cutoff. Default: 21 values over the range of the curve

    vsh_cutoff, por_cutoff, sw_cutoff : float
        Cutoffs kept fixed

    top, bot : float default None
        Zone top and bottom. Default: first and last depth of `df`

    Returns
    -------
    DataFrame with the cutoff value and the PAY summation for each value

    Example
    -------
    >>> sens = cutoff_sensitivity(zone_df, cutoff='por', values=np.arange(0, .2, .01), sw_cutoff=.6)
    >>> petrolib.plots.cutoffPlot(sens)
    '''

    cols = {'vsh': vsh, 'por': phie, 'sw': sw}
    if cutoff not in cols:
        raise ValueError("cutoff must be 'vsh', 'por' or 'sw'")
    if values is None:
        values = np.linspace(np.nanmin(df[cols[cutoff]]), np.nanmax(df[cols[cutoff]]), 21)
    top = df[depth].min() if top is None else top
    bot = df[depth].max() if bot is None else bot

    rows = []
    for v in values:
        c = {'vsh': vsh_cutoff, 'por': por_cutoff, 'sw': sw_cutoff, cutoff: v}
        flags = net_flags(df[vsh], df[phie], df[sw], c['vsh'], c['por'], c['sw'])
        rows.append({'cutoff': cutoff, 'value': v,
                     **summarize(df[depth], df[vsh], df[phie], df[sw], flags['PAY_NET_FLAG'].to_numpy(), top, bot)})
    return pd.DataFrame(rows)
