'''
Geomechanics and rock physics from logs: velocities, dynamic elastic moduli, brittleness, strength,
overburden, pore pressure (Eaton), fracture pressure and Gassmann fluid substitution

Units: density in g/cc, slowness in us/ft (or us/m), velocity in m/s, moduli in GPa,
depth in m (TVD) and pressure in MPa unless stated

'''
import numpy as np
import pandas as pd

from ._utils import as_array, like

_G = 9.80665                 #m/s2
_PSI_PER_MPA = 145.0377
_FT = .3048                  #m


def velocity(dt, unit:str='us/ft'):

    r'''
    Velocity (m/s) from slowness: V = 304800 / DT (us/ft) or 1e6 / DT (us/m)
    '''

    factor = 304800. if unit == 'us/ft' else 1e6
    with np.errstate(divide='ignore'):
        return like(factor / as_array(dt), dt)


def elastic_moduli(rhob, dtc, dts, unit:str='us/ft') -> pd.DataFrame:

    r'''
    Dynamic elastic properties from bulk density and compressional and shear slowness

        G = rho Vs^2 ;  K = rho (Vp^2 - 4/3 Vs^2) ;  PR = (Vp^2 - 2 Vs^2) / (2 (Vp^2 - Vs^2)) ;  E = 2 G (1 + PR)

    Parameters
    ----------
    rhob : array-like
        Bulk density (g/cc)

    dtc, dts : array-like
        Compressional and shear slowness

    unit : str default 'us/ft'
        Unit of the slowness: 'us/ft' or 'us/m'

    Returns
    -------
    DataFrame with VP, VS (m/s), VPVS, AI (acoustic impedance, m/s g/cc), G, K, E (GPa) and PR

    Example
    -------
    >>> mech = elastic_moduli(df['RHOB'], df['DT'], df['DTS'])
    '''

    rho = as_array(rhob) * 1000.
    vp, vs = as_array(velocity(dtc, unit)), as_array(velocity(dts, unit))
    with np.errstate(divide='ignore', invalid='ignore'):
        g = rho * vs**2
        k = rho * (vp**2 - 4 / 3 * vs**2)
        pr = (vp**2 - 2 * vs**2) / (2 * (vp**2 - vs**2))
        out = {'VP': vp, 'VS': vs, 'VPVS': vp / vs, 'AI': vp * rho / 1000., 'G': g / 1e9, 'K': k / 1e9,
               'E': 2 * g * (1 + pr) / 1e9, 'PR': pr}
    index = rhob.index if isinstance(rhob, pd.Series) else None
    return pd.DataFrame(out, index=index)


def brittleness(e, pr, e_range:tuple=None, pr_range:tuple=None):

    r'''
    Brittleness index (%) after Rickman et al. (2008): the mean of normalized Young's modulus and
    inverted normalized Poisson's ratio

        BI = 50 ((E - Emin) / (Emax - Emin) + (PRmax - PR) / (PRmax - PRmin)), clipped to [0, 100]

    Parameters
    ----------
    e, pr : array-like
        Young's modulus and Poisson's ratio (dynamic or static, in consistent units)

    e_range, pr_range : tuple default None
        (min, max) used to normalize. Default: the 1st and 99th percentiles of the data, so the
        index is relative to the interval. Fix them to compare wells

    Example
    -------
    >>> mech['BI'] = brittleness(mech['E'], mech['PR'])
    '''

    E, P = as_array(e), as_array(pr)
    e_lo, e_hi = e_range or np.nanpercentile(E, [1, 99])
    p_lo, p_hi = pr_range or np.nanpercentile(P, [1, 99])
    bi = 50 * ((E - e_lo) / (e_hi - e_lo) + (p_hi - P) / (p_hi - p_lo))
    return like(np.clip(bi, 0, 100), e, pr)


def ucs_mcnally(dtc):

    r'''
    Unconfined compressive strength (MPa) of sandstone from compressional slowness (us/ft),
    McNally (1987): UCS = 1200 exp(-0.036 DT). An empirical correlation; calibrate to core tests
    '''

    return like(1200. * np.exp(-.036 * as_array(dtc)), dtc)


def hydrostatic_pressure(depth, rho_water:float=1.03, unit:str='MPa', depth_unit:str='m'):

    r'''
    Hydrostatic pressure, rho_water g z
    '''

    z = as_array(depth) * (_FT if depth_unit == 'ft' else 1.)
    p = rho_water * 1000. * _G * z / 1e6
    return like(p * _PSI_PER_MPA if unit == 'psi' else p, depth)


def overburden(depth, rhob, rho_top:float=2.0, water_depth:float=0., rho_water:float=1.03, unit:str='MPa',
               depth_unit:str='m'):

    r'''
    Vertical (overburden) stress from integrating bulk density with depth

        Sv(z) = g (rho_water water_depth + integral of rho dz below the sea floor)

    Above the first logged depth, and where RHOB is missing, `rho_top` is used

    Parameters
    ----------
    depth : array-like
        True vertical depth below sea level or ground level, in increasing order

    rhob : array-like
        Bulk density (g/cc)

    rho_top : float default 2.0
        Density of the unlogged sediments (g/cc)

    water_depth : float default 0.
        Water depth offshore, in the unit of `depth`

    rho_water : float default 1.03
        Sea water density (g/cc)

    unit : str default 'MPa'
        'MPa' or 'psi'

    depth_unit : str default 'm'
        'm' or 'ft'

    Example
    -------
    >>> df['SV'] = overburden(df['TVD'], df['RHOB'], rho_top=1.95)
    '''

    scale = _FT if depth_unit == 'ft' else 1.
    z = as_array(depth) * scale
    rho = np.where(np.isfinite(as_array(rhob)), as_array(rhob), rho_top)
    wd = water_depth * scale
    first = rho_water * wd + rho_top * max(z[0] - wd, 0.)
    layers = rho * np.r_[0., np.diff(z)]
    sv = (first + np.cumsum(layers)) * 1000. * _G / 1e6
    return like(sv * _PSI_PER_MPA if unit == 'psi' else sv, depth)


def normal_compaction_trend(depth, values, fit=None):

    r'''
    Normal compaction trend: a straight line of log10(values) against depth, fitted on the
    `fit` samples (normally pressured shales), evaluated at every depth

    Parameters
    ----------
    depth, values : array-like
        Depth and shale sonic slowness (or resistivity)

    fit : array-like of bool default None
        Samples to fit. Default: all finite samples

    Example
    -------
    >>> shale = df['VShale'] > .7
    >>> df['DT_NORMAL'] = normal_compaction_trend(df['TVD'], df['DT'], fit=shale & (df['TVD'] < 2000))
    '''

    z, v = as_array(depth), as_array(values)
    ok = np.isfinite(z) & np.isfinite(v) & (v > 0)
    if fit is not None:
        ok &= as_array(fit).astype(bool)
    slope, intercept = np.polyfit(z[ok], np.log10(v[ok]), 1)
    return like(10**(intercept + slope * z), depth)


def pore_pressure_eaton(sv, ph, observed, normal, method:str='sonic', exponent:float=None):

    r'''
    Pore pressure in shales, Eaton (1975)

        sonic       : Pp = Sv - (Sv - Ph) (DT_normal / DT)^3
        resistivity : Pp = Sv - (Sv - Ph) (R / R_normal)^1.2

    Parameters
    ----------
    sv, ph : array-like
        Overburden and hydrostatic pressure

    observed, normal : array-like
        Shale sonic slowness (or resistivity), and its normal compaction trend

    method : str default 'sonic'
        'sonic' or 'resistivity'

    exponent : float default None
        Eaton exponent. Default 3 for sonic, 1.2 for resistivity; calibrate to pressure tests

    Example
    -------
    >>> pp = pore_pressure_eaton(sv, ph, df['DT'], df['DT_NORMAL'])
    '''

    S, P, o, nrm = as_array(sv), as_array(ph), as_array(observed), as_array(normal)
    if method == 'sonic':
        ratio, x = nrm / o, 3. if exponent is None else exponent
    elif method == 'resistivity':
        ratio, x = o / nrm, 1.2 if exponent is None else exponent
    else:
        raise ValueError("method must be 'sonic' or 'resistivity'")
    return like(S - (S - P) * ratio**x, sv, observed)


def fracture_pressure_eaton(sv, pp, pr):

    r'''
    Fracture pressure, Eaton (1969): Pf = PR / (1 - PR) (Sv - Pp) + Pp
    '''

    S, P, v = as_array(sv), as_array(pp), as_array(pr)
    return like(v / (1 - v) * (S - P) + P, sv, pp, pr)


def to_gradient(pressure, depth, unit:str='g/cc', depth_unit:str='m'):

    r'''
    Pressure (MPa) as a gradient: 'g/cc' (mud weight equivalent), 'MPa/km' or 'psi/ft'
    '''

    z = as_array(depth) * (_FT if depth_unit == 'ft' else 1.)
    p = as_array(pressure)
    with np.errstate(divide='ignore', invalid='ignore'):
        if unit == 'g/cc':
            g = p * 1e6 / (_G * z) / 1000.
        elif unit == 'MPa/km':
            g = p / z * 1000.
        elif unit == 'psi/ft':
            g = p * _PSI_PER_MPA / (z / _FT)
        else:
            raise ValueError("unit must be 'g/cc', 'MPa/km' or 'psi/ft'")
    return like(g, pressure, depth)


def wood_fluid_modulus(sw, k_water:float=2.8, k_hydrocarbon:float=.05):

    r'''
    Bulk modulus (GPa) of a water-hydrocarbon mix, Wood (1955): 1/K = SW / Kw + (1 - SW) / Khc.
    Typical Kw 2.2-3.0, oil 0.5-1.5, gas 0.02-0.15 GPa
    '''

    s = as_array(sw)
    return like(1 / (s / k_water + (1 - s) / k_hydrocarbon), sw)


def gassmann(k_sat, k_mineral:float, k_fluid, k_fluid_new, phi):

    r'''
    Gassmann (1951) fluid substitution of the saturated bulk modulus (GPa). The dry-rock modulus is
    found from the original saturation, then saturated with the new fluid

        K_dry = (K_sat (phi Km / Kf + 1 - phi) - Km) / (phi Km / Kf + K_sat / Km - 1 - phi)
        K_sat_new = K_dry + (1 - K_dry/Km)^2 / (phi / Kf_new + (1 - phi) / Km - K_dry / Km^2)

    Parameters
    ----------
    k_sat : array-like
        Saturated bulk modulus from logs, e.g. `elastic_moduli(...)['K']`

    k_mineral : float
        Mineral bulk modulus, e.g. quartz 36.6, calcite 76.8 GPa

    k_fluid, k_fluid_new : array-like or float
        Pore fluid modulus in situ and after substitution, see `wood_fluid_modulus`

    phi : array-like
        Porosity

    Example
    -------
    >>> k_brine = gassmann(mech['K'], 36.6, wood_fluid_modulus(df['SW'], 2.8, .05), 2.8, df['PHIE'])
    '''

    ks, km, p = as_array(k_sat), k_mineral, as_array(phi)
    kf, kf2 = as_array(k_fluid), as_array(k_fluid_new)
    with np.errstate(divide='ignore', invalid='ignore'):
        k_dry = (ks * (p * km / kf + 1 - p) - km) / (p * km / kf + ks / km - 1 - p)
        k_new = k_dry + (1 - k_dry / km)**2 / (p / kf2 + (1 - p) / km - k_dry / km**2)
    return like(k_new, k_sat, phi)


def vs_castagna(vp):

    r'''
    Shear velocity (m/s) from compressional velocity with the mudrock line of Castagna et al. (1985),
    for water-saturated clastics: Vs = 0.8621 Vp - 1172.4. Use when no shear log was run
    '''

    return like(.8621 * as_array(vp) - 1172.4, vp)
