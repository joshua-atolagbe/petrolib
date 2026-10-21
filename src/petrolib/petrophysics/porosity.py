'''
Total and effective porosity from density, neutron-density and sonic logs

'''
import numpy as np

from ._utils import as_array, like


def density_porosity(rhob, rhob_matrix:float=2.65, rhob_fluid:float=1.):

    r'''
    Density porosity (not clipped)

        PHID = (rhob_matrix - RHOB) / (rhob_matrix - rhob_fluid)

    Parameters
    ----------
    rhob : array-like
        Bulk density (g/cc)

    rhob_matrix : float default 2.65
        Grain density. Sandstone 2.65, limestone 2.71, dolomite 2.87

    rhob_fluid : float default 1.0
        Pore fluid density. 1.0 for fresh mud filtrate

    Example
    -------
    >>> phid = density_porosity(df['RHOB'], rhob_matrix=2.71)
    '''

    return like((rhob_matrix - as_array(rhob)) / (rhob_matrix - rhob_fluid), rhob)


def neutron_density_porosity(nphi, phid):

    r'''
    Neutron-density porosity

        sqrt((PHIN^2 + PHID^2) / 2)   where the logs cross over (PHIN < PHID, gas)
        (PHIN + PHID) / 2             elsewhere

    Parameters
    ----------
    nphi : array-like
        Neutron porosity (fraction), in the matrix units of `phid`

    phid : array-like
        Density porosity (fraction)

    Example
    -------
    >>> phit = neutron_density_porosity(df['NPHI'], density_porosity(df['RHOB']))
    '''

    n, d = as_array(nphi), as_array(phid)
    return like(np.where(n < d, np.sqrt((n**2 + d**2) / 2), (n + d) / 2), nphi, phid)


def shale_corrected_nd_porosity(nphi, phid, vsh, nphi_shale:float, phid_shale:float):

    r'''
    Effective porosity from neutron and density logs, corrected for shale before combining

        PHIN_c = PHIN - Vsh * nphi_shale ; PHID_c = PHID - Vsh * phid_shale  (floored at 0)
        PHIE   = neutron_density_porosity(PHIN_c, PHID_c)

    Parameters
    ----------
    nphi, phid : array-like
        Neutron and density porosity (fraction)

    vsh : array-like
        Volume of shale

    nphi_shale, phid_shale : float
        Neutron and density porosity of shale

    Example
    -------
    >>> phie = shale_corrected_nd_porosity(df['NPHI'], phid, df['VShale'], .33, .18)
    '''

    v = as_array(vsh)
    n = np.clip(np.clip(as_array(nphi), 0, None) - nphi_shale * v, 0, None)
    d = np.clip(np.clip(as_array(phid), 0, None) - phid_shale * v, 0, None)
    return like(neutron_density_porosity(n, d), nphi, phid, vsh)


def sonic_porosity(dt, dt_matrix:float=55.5, dt_fluid:float=189., cp:float=1.):

    r'''
    Wyllie time-average sonic porosity (not clipped)

        PHIS = (DT - dt_matrix) / (dt_fluid - dt_matrix) / cp

    Parameters
    ----------
    dt : array-like
        Compressional slowness (us/ft)

    dt_matrix : float default 55.5
        Matrix slowness. Sandstone 55.5, limestone 47.5, dolomite 43.5

    dt_fluid : float default 189.
        Fluid slowness

    cp : float default 1.
        Compaction correction for unconsolidated sands

    Example
    -------
    >>> phis = sonic_porosity(df['DT'], dt_matrix=47.5)
    '''

    return like((as_array(dt) - dt_matrix) / (dt_fluid - dt_matrix) / cp, dt)


def raymer_porosity(dt, dt_matrix:float=55.5, dt_fluid:float=189.):

    r'''
    Raymer-Hunt-Gardner (1980) sonic porosity, valid up to about 37% porosity

        1/DT = (1 - PHI)^2 / dt_matrix + PHI / dt_fluid,  solved for PHI

    Parameters
    ----------
    dt : array-like
        Compressional slowness (us/ft)

    dt_matrix : float default 55.5
        Matrix slowness. Sandstone 55.5, limestone 47.5, dolomite 43.5

    dt_fluid : float default 189.
        Fluid slowness

    Example
    -------
    >>> phis = raymer_porosity(df['DT'])
    '''

    t = as_array(dt)
    a, b = 1 / dt_matrix, 1 / dt_fluid - 2 / dt_matrix
    c = 1 / dt_matrix - 1 / t
    with np.errstate(invalid='ignore'):
        phi = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    return like(phi, dt)


def effective_porosity(phit, vsh, phi_shale:float):

    r'''
    Effective porosity, floored at 0

        PHIE = PHIT - Vsh * phi_shale

    Parameters
    ----------
    phit : array-like
        Total porosity

    vsh : array-like
        Volume of shale

    phi_shale : float
        Porosity read in shale by the same method as `phit`

    Example
    -------
    >>> phie = effective_porosity(phit, df['VShale'], .18)
    '''

    phie = as_array(phit) - phi_shale * as_array(vsh)
    return like(np.where(phie < 0, 0., phie), phit, vsh)
