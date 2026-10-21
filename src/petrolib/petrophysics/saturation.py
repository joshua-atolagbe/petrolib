'''
Water saturation: Archie and shaly-sand models

All functions return SW in [0, 1], with SW = 1 where porosity is 0

'''
import numpy as np
import pandas as pd

from ._utils import as_array, like, solve_increasing


def _finish(sw, phi, *refs):

    '''
    SW = 1 where there is no porosity, values above 1 set to 1
    '''

    sw = np.where(phi == 0, 1., sw)
    sw = np.where(sw > 1, 1., sw)
    return like(sw, *refs)


def archie(phie, rt, rw:float=.03, a:float=1., m:float=2., n:float=2.):

    r'''
    Archie (1942) water saturation, for clean formations

        SW = (a * rw / (PHIE^m * RT))^(1/n)

    Parameters
    ----------
    phie : array-like
        Porosity (fraction)

    rt : array-like
        True resistivity (ohm-m)

    rw : float default 0.03
        Formation water resistivity at formation temperature (ohm-m)

    a, m, n : float default 1, 2, 2
        Tortuosity factor, cementation and saturation exponents

    Example
    -------
    >>> sw = archie(df['PHIE'], df['RT'], rw=.05, m=1.8)
    '''

    phi, r = as_array(phie), as_array(rt)
    with np.errstate(divide='ignore', invalid='ignore'):
        sw = ((a/(phi**m)) * (rw/r))**(1/n)
    return _finish(sw, phi, phie, rt)


def simandoux(phie, rt, vsh, rw:float=.03, rsh:float=2., a:float=1., m:float=2., n:float=2.):

    r'''
    Modified Simandoux water saturation, for shaly sands

        1/RT = PHIE^m SW^n / (a rw (1 - Vsh)) + Vsh SW / rsh

    Solved exactly for n = 2. For other n the n = 2 solution is raised to 2/n

    Parameters
    ----------
    phie, rt, vsh : array-like
        Effective porosity, true resistivity (ohm-m) and volume of shale

    rw : float default 0.03
        Formation water resistivity (ohm-m)

    rsh : float default 2.
        Shale resistivity (ohm-m), read in a nearby shale

    a, m, n : float default 1, 2, 2
        Archie parameters

    Example
    -------
    >>> sw = simandoux(df['PHIE'], df['RT'], df['VShale'], rw=.05, rsh=3.)
    '''

    phi, r, v = as_array(phie), as_array(rt), as_array(vsh)
    with np.errstate(divide='ignore', invalid='ignore'):
        A = (1 - v) * a * rw / (phi ** m)
        B = A * v / (2 * rsh)
        C = A / r
        sw = ((B **2 + C)**0.5 - B) **(2 / n)
    return _finish(sw, phi, phie, rt, vsh)


def indonesia(phie, rt, vsh, rw:float=.03, rsh:float=2., a:float=1., m:float=2., n:float=2.):

    r'''
    Indonesia (Poupon-Leveaux, 1971) water saturation, for shaly sands with fresh formation water

        1/sqrt(RT) = (Vsh^(1 - Vsh/2) / sqrt(rsh) + sqrt(PHIE^m / (a rw))) SW^(n/2)

    Parameters
    ----------
    phie, rt, vsh : array-like
        Effective porosity, true resistivity (ohm-m) and volume of shale

    rw : float default 0.03
        Formation water resistivity (ohm-m)

    rsh : float default 2.
        Shale resistivity (ohm-m)

    a, m, n : float default 1, 2, 2
        Archie parameters

    Example
    -------
    >>> sw = indonesia(df['PHIE'], df['RT'], df['VShale'], rw=.2, rsh=4.)
    '''

    phi, r, v = as_array(phie), as_array(rt), as_array(vsh)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = v**(1 - v/2) / np.sqrt(rsh) + np.sqrt(phi**m / (a * rw))
        sw = ((1 / np.sqrt(r)) / term)**(2 / n)
    return _finish(sw, phi, phie, rt, vsh)


def juhasz_b(rw:float, temp_c:float) -> float:

    r'''
    Equivalent conductance of clay exchange cations, B, from Juhasz (1981)

        B = (-1.28 + 0.225 T - 0.0004059 T^2) / (1 + rw^1.23 (0.045 T - 0.27))

    Parameters
    ----------
    rw : float
        Formation water resistivity at formation temperature (ohm-m)

    temp_c : float
        Formation temperature (degC)
    '''

    return (-1.28 + .225*temp_c - .0004059*temp_c**2) / (1 + rw**1.23 * (.045*temp_c - .27))


def qv_from_cec(cec, phit, rho_grain:float=2.65):

    r'''
    Cation exchange capacity per unit pore volume (meq/cc)

        Qv = CEC rho_grain (1 - PHIT) / (100 PHIT)

    Parameters
    ----------
    cec : array-like
        Cation exchange capacity (meq/100 g), from core

    phit : array-like
        Total porosity

    rho_grain : float default 2.65
        Grain density (g/cc)
    '''

    phi = as_array(phit)
    with np.errstate(divide='ignore', invalid='ignore'):
        return like(as_array(cec) * rho_grain * (1 - phi) / (100 * phi), cec, phit)


def waxman_smits(phit, rt, qv, rw:float=.03, b:float=None, temp_c:float=None,
                 a:float=1., m:float=2., n:float=2.):

    r'''
    Waxman-Smits (1968) water saturation, for shaly sands with known clay cation exchange

        1/RT = PHIT^m SW^n / a * (1/rw + B Qv / SW)

    Solved for SW by bisection

    Parameters
    ----------
    phit, rt : array-like
        Total porosity and true resistivity (ohm-m)

    qv : array-like or float
        Cation exchange capacity per pore volume (meq/cc), see `qv_from_cec`

    rw : float default 0.03
        Formation water resistivity (ohm-m)

    b : float default None
        Equivalent counterion conductance. If None, computed from `temp_c` with `juhasz_b`

    temp_c : float default None
        Formation temperature (degC), needed when `b` is None

    a, m, n : float default 1, 2, 2
        Archie parameters (m*, n* of Waxman-Smits)

    Example
    -------
    >>> sw = waxman_smits(df['PHIT'], df['RT'], qv=.3, rw=.05, temp_c=90)
    '''

    if b is None:
        assert temp_c is not None, 'Pass `b` or `temp_c`'
        b = juhasz_b(rw, temp_c)
    phi, r, q = as_array(phit), as_array(rt), as_array(qv)
    pm = phi**m

    def residual(sw):
        return pm * sw**n / (a * rw) + pm * b * q * sw**(n - 1) / a - 1 / r

    with np.errstate(divide='ignore', invalid='ignore'):
        sw = solve_increasing(residual, 0.)
    return _finish(sw, phi, phit, rt, qv)


def dual_water(phit, rt, swb, rw:float=.03, rwb:float=.1, a:float=1., m:float=2., n:float=2.,
               effective:bool=False):

    r'''
    Dual water (Clavier et al., 1984) water saturation

        1/RT = PHIT^m SWT^n / a * (1/rw + SWB/SWT (1/rwb - 1/rw))

    Solved for the total water saturation SWT by bisection on [SWB, 1]

    Parameters
    ----------
    phit, rt : array-like
        Total porosity and true resistivity (ohm-m)

    swb : array-like or float
        Bound water saturation, e.g. Vsh * PHIT_shale / PHIT

    rw : float default 0.03
        Free water resistivity (ohm-m)

    rwb : float default 0.1
        Bound water resistivity (ohm-m), from RT and PHIT in shale

    a, m, n : float default 1, 2, 2
        Archie parameters

    effective : bool default False
        Return the effective water saturation SWE = (SWT - SWB) / (1 - SWB) instead of SWT

    Example
    -------
    >>> swt = dual_water(df['PHIT'], df['RT'], swb=.15, rw=.05, rwb=.12)
    '''

    phi, r = as_array(phit), as_array(rt)
    sb = np.clip(as_array(swb) * np.ones_like(phi), 0, 1)
    pm = phi**m

    def residual(sw):
        return pm * sw**n / a / rw + pm * sw**(n - 1) / a * sb * (1/rwb - 1/rw) - 1 / r

    with np.errstate(divide='ignore', invalid='ignore'):
        swt = solve_increasing(residual, sb)
        swt = np.where(phi == 0, 1., np.minimum(swt, 1.))
        out = (swt - sb) / (1 - sb) if effective else swt
    return _finish(out, phi, phit, rt, swb)


def bulk_volume_water(phie, sw):

    r'''
    Bulk volume of water, BVW = PHIE * SW
    '''

    return like(as_array(phie) * as_array(sw), phie, sw)


def flushed_zone_saturation(phie, rxo, rmf:float=.1, a:float=1., m:float=2., n:float=2.):

    r'''
    Flushed zone water saturation, Archie with the mud filtrate: SXO = (a Rmf / (PHIE^m Rxo))^(1/n)

    Parameters
    ----------
    phie, rxo : array-like
        Porosity and flushed zone resistivity (micro-resistivity log, ohm-m)

    rmf : float default 0.1
        Mud filtrate resistivity at formation temperature (ohm-m), from the log header and `arps`

    a, m, n : float default 1, 2, 2
        Archie parameters

    Example
    -------
    >>> df['SXO'] = flushed_zone_saturation(df['PHIE'], df['RXO'], rmf=arps(.25, 75, 180))
    '''

    return archie(phie, rxo, rmf, a, m, n)


def movable_hydrocarbon_index(sw, sxo):

    r'''
    Movable hydrocarbon index, MHI = SW / SXO. Hydrocarbons moved when the mud filtrate invaded
    where MHI is below about 0.7 in sandstones (0.6 in carbonates); near 1 they did not
    '''

    with np.errstate(divide='ignore', invalid='ignore'):
        return like(as_array(sw) / as_array(sxo), sw, sxo)


def movable_hydrocarbons(phie, sw, sxo) -> pd.DataFrame:

    r'''
    Bulk volumes of the invaded and uninvaded zones

    Returns
    -------
    DataFrame with BVW (PHIE SW), BVXO (PHIE SXO), MOVABLE_HC (PHIE (SXO - SW), hydrocarbon moved by
    invasion) and RESIDUAL_HC (PHIE (1 - SXO))
    '''

    p, s, x = as_array(phie), as_array(sw), as_array(sxo)
    index = phie.index if isinstance(phie, pd.Series) else None
    return pd.DataFrame({'BVW': p * s, 'BVXO': p * x, 'MOVABLE_HC': p * (x - s), 'RESIDUAL_HC': p * (1 - x)}, index=index)
