'''
Capillary pressure, Leverett J-function and saturation-height functions

'''
import numpy as np

from ._utils import as_array, like

#psi per (g/cc of density difference x ft of height)
_PSI_PER_FT = .433
_FT_PER_M = 3.28084


def pc_from_height(height, delta_rho:float, unit:str='m'):

    r'''
    Reservoir capillary pressure (psi) at a height above the free water level

        Pc = 0.433 delta_rho h   (h in ft, delta_rho in g/cc)

    Parameters
    ----------
    height : array-like
        Height above the free water level

    delta_rho : float
        Water minus hydrocarbon density (g/cc)

    unit : str default 'm'
        Unit of `height`: 'm' or 'ft'

    Example
    -------
    >>> pc = pc_from_height(fwl - df['TVDSS'], delta_rho=.8)
    '''

    h = as_array(height) * (_FT_PER_M if unit == 'm' else 1.)
    return like(_PSI_PER_FT * delta_rho * h, height)


def height_from_pc(pc, delta_rho:float, unit:str='m'):

    r'''
    Height above the free water level of a reservoir capillary pressure (psi), inverse of `pc_from_height`
    '''

    h = as_array(pc) / (_PSI_PER_FT * delta_rho)
    return like(h / _FT_PER_M if unit == 'm' else h, pc)


def leverett_j(pc, k, phi, sigma:float=26., theta:float=30.):

    r'''
    Leverett J-function (dimensionless)

        J = 0.2166 Pc / (sigma cos(theta)) sqrt(K / PHI)

    Parameters
    ----------
    pc : array-like
        Capillary pressure (psi), at the conditions of `sigma` and `theta`

    k, phi : array-like
        Permeability (mD) and porosity (fraction)

    sigma : float default 26.
        Interfacial tension (dyne/cm). Reservoir oil-water about 26, gas-water about 50

    theta : float default 30.
        Contact angle (degrees)

    Example
    -------
    >>> j = leverett_j(pc_from_height(h, .8), df['Perm'], df['PHIE'])
    '''

    with np.errstate(divide='ignore', invalid='ignore'):
        j = .2166 * as_array(pc) / (sigma * np.cos(np.radians(theta))) * np.sqrt(as_array(k) / as_array(phi))
    return like(j, pc, k, phi)


def fit_saturation_height(j, sw) -> tuple[float, float]:

    r'''
    Fits SW = a J^b to core or log data, by least squares on log(SW) against log(J)

    Returns
    -------
    (a, b)

    Example
    -------
    >>> a, b = fit_saturation_height(core['J'], core['SW'])
    >>> sw_model = sw_from_j(j, a, b)
    '''

    jj, ss = as_array(j), as_array(sw)
    ok = np.isfinite(jj) & np.isfinite(ss) & (jj > 0) & (ss > 0)
    b, log_a = np.polyfit(np.log(jj[ok]), np.log(ss[ok]), 1)
    return float(np.exp(log_a)), float(b)


def sw_from_j(j, a:float, b:float, swirr:float=0.):

    r'''
    Water saturation from a fitted J-function, SW = a J^b, kept within [swirr, 1]
    '''

    with np.errstate(divide='ignore'):
        sw = a * as_array(j)**b
    return like(np.clip(sw, swirr, 1), j)
