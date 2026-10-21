'''
Rock typing: rock quality index, flow zone indicator, Winland R35 and Lorenz plots

'''
import numpy as np
import pandas as pd

from ._utils import as_array, like

#pore throat classes of Martin et al. (1997), lower bound of R35 in microns
PORE_TYPES = (('megaport', 10.), ('macroport', 2.), ('mesoport', .5), ('microport', .1), ('nanoport', 0.))


def rqi(k, phie):

    r'''
    Rock quality index (microns), K in mD and PHIE as a fraction

        RQI = 0.0314 sqrt(K / PHIE)
    '''

    with np.errstate(divide='ignore', invalid='ignore'):
        return like(.0314 * np.sqrt(as_array(k) / as_array(phie)), k, phie)


def normalized_porosity(phie):

    r'''
    Normalized porosity (pore to grain volume ratio), PHIZ = PHIE / (1 - PHIE)
    '''

    phi = as_array(phie)
    return like(phi / (1 - phi), phie)


def fzi(k, phie):

    r'''
    Flow zone indicator (microns), Amaefule et al. (1993)

        FZI = RQI / PHIZ

    Samples with the same FZI belong to the same hydraulic flow unit

    Example
    -------
    >>> df['FZI'] = fzi(df['Perm'], df['PHIE'])
    '''

    with np.errstate(divide='ignore', invalid='ignore'):
        return like(as_array(rqi(k, phie)) / as_array(normalized_porosity(phie)), k, phie)


def drt(fzi_values):

    r'''
    Discrete rock type from the flow zone indicator (Guo et al., 2005)

        DRT = round(2 ln(FZI) + 10.6)
    '''

    f = as_array(fzi_values)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.round(2 * np.log(f) + 10.6)
    return like(np.where(np.isfinite(d), d, np.nan), fzi_values)


def winland_r35(k, phie):

    r'''
    Winland R35 pore throat radius (microns), K in mD and PHIE as a fraction (Kolodzie, 1980)

        log10(R35) = 0.732 + 0.588 log10(K) - 0.864 log10(100 PHIE)

    Example
    -------
    >>> df['R35'] = winland_r35(df['Perm'], df['PHIE'])
    '''

    with np.errstate(divide='ignore', invalid='ignore'):
        r35 = 10**(.732 + .588*np.log10(as_array(k)) - .864*np.log10(100*as_array(phie)))
    return like(r35, k, phie)


def pore_type(r35):

    r'''
    Pore throat class of an R35 radius: megaport (> 10 um), macroport (2-10), mesoport (0.5-2),
    microport (0.1-0.5) or nanoport (< 0.1). NaN where R35 is NaN
    '''

    r = as_array(r35)
    labels = np.full(r.shape, None, dtype=object)
    for name, low in reversed(PORE_TYPES):
        labels[r >= low] = name
    return like(labels, r35)


def lorenz(k, phi, h=None, stratigraphic:bool=False) -> tuple[pd.DataFrame, float]:

    r'''
    Lorenz curve: cumulative flow capacity (k h) against cumulative storage capacity (PHI h)

    Parameters
    ----------
    k, phi : array-like
        Permeability (mD) and porosity, in depth order

    h : array-like default None
        Thickness of each sample. Equal thicknesses if None

    stratigraphic : bool default False
        Keep depth order (stratigraphic modified Lorenz plot), which shows where the flow
        units are. Otherwise samples are sorted by decreasing k/PHI (classic Lorenz plot)

    Returns
    -------
    DataFrame with columns STORAGE and FLOW (0 to 1), and the Lorenz coefficient
    (0 homogeneous, 1 fully heterogeneous), from the classic curve

    Example
    -------
    >>> curve, lc = lorenz(df['Perm'], df['PHIE'], stratigraphic=True)
    '''

    kk, pp = as_array(k), as_array(phi)
    hh = np.ones_like(kk) if h is None else as_array(h)
    ok = np.isfinite(kk) & np.isfinite(pp) & np.isfinite(hh)
    kk, pp, hh = kk[ok], pp[ok], hh[ok]

    def curve(order):
        kh, ph = (kk * hh)[order], (pp * hh)[order]
        return (np.concatenate([[0], np.cumsum(ph) / ph.sum()]),
                np.concatenate([[0], np.cumsum(kh) / kh.sum()]))

    with np.errstate(divide='ignore', invalid='ignore'):
        classic = np.argsort(-(kk / pp))
    s, f = curve(classic)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz      #np.trapz is deprecated in numpy 2
    lc = 2 * (trapezoid(f, s) - .5)
    if stratigraphic:
        s, f = curve(np.arange(len(kk)))
    return pd.DataFrame({'STORAGE': s, 'FLOW': f}), float(lc)
