'''
In-place hydrocarbon volumes

'''
from ._utils import as_array, like

#barrels per acre-ft, and cubic feet per acre-ft
_BBL_PER_ACRE_FT = 7758.
_FT3_PER_ACRE_FT = 43560.


def stoiip(area, net, phi, sw, bo:float=1.2, unit:str='field'):

    r'''
    Stock tank oil initially in place

        field  : 7758 A h PHI (1 - SW) / Bo   (A in acres, h in ft, result in STB)
        metric : A h PHI (1 - SW) / Bo        (A in m2, h in m, result in stock tank m3)

    Parameters
    ----------
    area : float or array-like
        Reservoir area

    net : float or array-like
        Net pay thickness

    phi, sw : float or array-like
        Average porosity and water saturation of the net pay (e.g. from `paySummary`)

    bo : float default 1.2
        Oil formation volume factor (rb/STB or rm3/sm3)

    unit : str default 'field'
        'field' or 'metric'

    Example
    -------
    >>> stoiip(area=640, net=35, phi=.18, sw=.3, bo=1.25)
    '''

    hcpv = as_array(area) * as_array(net) * as_array(phi) * (1 - as_array(sw))
    return like((_BBL_PER_ACRE_FT if unit == 'field' else 1.) * hcpv / bo, area, net, phi, sw)


def giip(area, net, phi, sw, bg:float=.005, unit:str='field'):

    r'''
    Gas initially in place

        field  : 43560 A h PHI (1 - SW) / Bg  (A in acres, h in ft, Bg in rcf/scf, result in scf)
        metric : A h PHI (1 - SW) / Bg        (A in m2, h in m, Bg in rm3/sm3, result in sm3)

    Example
    -------
    >>> giip(area=1000, net=50, phi=.15, sw=.25, bg=.004)
    '''

    hcpv = as_array(area) * as_array(net) * as_array(phi) * (1 - as_array(sw))
    return like((_FT3_PER_ACRE_FT if unit == 'field' else 1.) * hcpv / bg, area, net, phi, sw)
