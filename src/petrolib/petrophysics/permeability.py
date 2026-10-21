'''
Permeability from porosity and irreducible water saturation (mD)

The equations need the irreducible water saturation. Above the transition zone SW is close
to it; elsewhere estimate it, e.g. with `swirr_buckles`. Calibrate the constants to core

'''
import numpy as np

from ._utils import as_array, like

PERMEABILITY_METHODS = ('timur', 'coates', 'morris_biggs')


def _swirr(swirr, floor:float):
    return np.clip(as_array(swirr), floor, None)


def timur(phie, swirr, floor:float=.05):

    r'''
    Timur (1968) permeability, PHIE and SWIRR as fractions

        K = 8581 PHIE^4.4 / SWIRR^2    (0.136 PHI^4.4 / SWIRR^2 with both in percent)

    Parameters
    ----------
    phie : array-like
        Effective porosity

    swirr : array-like
        Irreducible water saturation

    floor : float default 0.05
        Lower limit of SWIRR, to avoid blow-up in very clean zones

    Example
    -------
    >>> k = timur(df['PHIE'], df['SW'])
    '''

    k = 8581. * pow(as_array(phie), 4.4) / pow(_swirr(swirr, floor), 2)
    return like(np.where(k < 0, 0., k), phie, swirr)


def coates(phie, swirr, c:float=100., floor:float=.05):

    r'''
    Coates permeability, PHIE and SWIRR as fractions

        sqrt(K) = c PHIE^2 (1 - SWIRR) / SWIRR

    c = 100 is the common form. Some references use K = 5000 PHIE^4 ((1 - SWIRR)/SWIRR)^2, i.e. c = 70.7

    Parameters
    ----------
    phie, swirr : array-like
        Effective porosity and irreducible water saturation

    c : float default 100.
        Constant, to calibrate to core

    floor : float default 0.05
        Lower limit of SWIRR

    Example
    -------
    >>> k = coates(df['PHIE'], swirr_buckles(df['PHIE'], .04))
    '''

    s = _swirr(swirr, floor)
    k = (c * as_array(phie)**2 * (1 - s) / s)**2
    return like(np.where(s >= 1, 0., k), phie, swirr)


def morris_biggs(phie, swirr, fluid:str='oil', floor:float=.05):

    r'''
    Morris and Biggs (1967) permeability, PHIE and SWIRR as fractions

        sqrt(K) = C PHIE^3 / SWIRR,  C = 250 (medium gravity oil) or 79 (dry gas)

    Parameters
    ----------
    phie, swirr : array-like
        Effective porosity and irreducible water saturation

    fluid : str default 'oil'
        'oil' or 'gas'

    floor : float default 0.05
        Lower limit of SWIRR

    Example
    -------
    >>> k = morris_biggs(df['PHIE'], df['SW'], fluid='gas')
    '''

    c = {'oil': 250., 'gas': 79.}[fluid]
    return like((c * as_array(phie)**3 / _swirr(swirr, floor))**2, phie, swirr)


def wyllie_rose(phie, swirr, c:float, x:float, y:float, floor:float=.05):

    r'''
    General Wyllie and Rose (1950) permeability, to fit to core

        sqrt(K) = c PHIE^x / SWIRR^y

    Example
    -------
    >>> k = wyllie_rose(df['PHIE'], df['SW'], c=93, x=2.2, y=1)     # close to Timur
    '''

    return like((c * as_array(phie)**x / _swirr(swirr, floor)**y)**2, phie, swirr)


def swirr_buckles(phie, buckles:float):

    r'''
    Irreducible water saturation from a Buckles number (constant bulk volume water at
    irreducible saturation, PHIE * SWIRR), clipped to [0, 1]

        SWIRR = buckles / PHIE

    Typical Buckles numbers: 0.02-0.04 for coarse to medium sandstone, 0.05-0.09 for fine and
    silty sandstone. Read it from a Buckles plot of the zone (`petrolib.plots.bucklesPlot`)

    Example
    -------
    >>> swirr = swirr_buckles(df['PHIE'], .04)
    '''

    phi = as_array(phie)
    with np.errstate(divide='ignore'):
        return like(np.clip(buckles / phi, 0, 1), phie)


def permeability(phie, swirr, method:str='timur', **kwargs):

    r'''
    Permeability with one of PERMEABILITY_METHODS: 'timur', 'coates' or 'morris_biggs'.
    Extra keyword arguments go to that function
    '''

    funcs = {'timur': timur, 'coates': coates, 'morris_biggs': morris_biggs}
    if method not in funcs:
        raise ValueError(f"Unknown permeability method '{method}'. Use one of {PERMEABILITY_METHODS}")
    return funcs[method](phie, swirr, **kwargs)
