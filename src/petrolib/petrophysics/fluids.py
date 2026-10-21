'''
Formation temperature and water resistivity

'''

from ._utils import as_array, like

#temperature offsets of Arps' formula
_ARPS = {'F': 6.77, 'C': 21.5}


def formation_temperature(depth, surface_temp:float, gradient:float=None, bht:float=None, td:float=None):

    r'''
    Formation temperature from a linear geothermal gradient

        T = surface_temp + gradient * depth,   gradient = (bht - surface_temp) / td if not given

    Parameters
    ----------
    depth : array-like
        Depth (TVD), in the length unit of `gradient` or `td`

    surface_temp : float
        Surface (or mean annual) temperature

    gradient : float default None
        Temperature gradient, e.g. degC/m. Pass this or `bht` and `td`

    bht, td : float default None
        Bottom hole temperature and the depth where it was measured

    Example
    -------
    >>> temp = formation_temperature(df['DEPTH'], surface_temp=10, bht=105, td=3200)
    '''

    if gradient is None:
        assert bht is not None and td is not None, 'Pass `gradient`, or `bht` and `td`'
        gradient = (bht - surface_temp) / td
    return like(surface_temp + gradient * as_array(depth), depth)


def arps(r1, t1, t2, unit:str='F'):

    r'''
    Arps' conversion of a water or mud resistivity to another temperature

        R2 = R1 (T1 + k) / (T2 + k),  k = 6.77 (degF) or 21.5 (degC)

    Parameters
    ----------
    r1 : array-like or float
        Resistivity at temperature `t1` (ohm-m)

    t1, t2 : array-like or float
        Temperature of the measurement and the target temperature

    unit : str default 'F'
        'F' or 'C'

    Example
    -------
    >>> rw = arps(.05, 75, 180)          # Rw measured at 75 degF, at 180 degF
    '''

    k = _ARPS[unit.upper()]
    return like(as_array(r1) * (as_array(t1) + k) / (as_array(t2) + k), r1, t1, t2)


def rw_from_salinity(ppm, temp:float=75., unit:str='F'):

    r'''
    Water resistivity of a NaCl solution (Bateman and Konen, 1977), converted to `temp` with Arps

        Rw(75 degF) = 0.0123 + 3647.5 / ppm^0.955

    Parameters
    ----------
    ppm : array-like or float
        NaCl equivalent salinity (ppm)

    temp : float default 75.
        Temperature to return Rw at

    unit : str default 'F'
        Unit of `temp`: 'F' or 'C'

    Example
    -------
    >>> rw = rw_from_salinity(150000, temp=95, unit='C')
    '''

    rw75 = .0123 + 3647.5 / as_array(ppm)**.955
    t75 = 75. if unit.upper() == 'F' else (75. - 32) / 1.8
    return like(arps(rw75, t75, temp, unit), ppm)


def rwa(rt, phi, a:float=1., m:float=2.):

    r'''
    Apparent water resistivity, Rwa = PHI^m RT / a. In water zones it equals Rw; in
    hydrocarbon zones it is higher, so the minimum Rwa of a clean zone estimates Rw

    Parameters
    ----------
    rt : array-like
        True resistivity (ohm-m)

    phi : array-like
        Porosity (fraction)

    a, m : float default 1, 2
        Tortuosity factor and cementation exponent

    Example
    -------
    >>> rw_estimate = np.nanpercentile(rwa(df['RT'], df['PHIE']), 5)
    '''

    return like(as_array(phi)**m * as_array(rt) / a, rt, phi)


def rw_from_sp(ssp, rmf, temp:float, unit:str='F'):

    r'''
    Formation water resistivity from the static SP of a clean, thick water sand

        K = 61 + 0.133 T (degF)  or  65 + 0.24 T (degC)
        Rwe = Rmfe 10^(SSP / K),  with Rmfe = 0.85 Rmf and Rw = Rwe

    The two simplifications hold for NaCl waters and mud filtrates with Rmf above about
    0.1 ohm-m at 75 degF; otherwise convert Rmf and Rwe with the SP-2 chart

    Parameters
    ----------
    ssp : array-like or float
        Static SP (mV), negative when the formation water is saltier than the filtrate

    rmf : float
        Mud filtrate resistivity at formation temperature (ohm-m), see `arps`

    temp : float
        Formation temperature

    unit : str default 'F'
        Unit of `temp`: 'F' or 'C'

    Example
    -------
    >>> rw = rw_from_sp(-90, rmf=arps(.4, 75, 160), temp=160)
    '''

    k = 61. + .133 * temp if unit.upper() == 'F' else 65. + .24 * temp
    return like(.85 * as_array(rmf) * 10**(as_array(ssp) / k), ssp)
