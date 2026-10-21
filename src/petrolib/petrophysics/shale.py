'''
Volume of shale from the gamma ray index and from neutron-density separation

'''
import numpy as np

from ._utils import as_array, like

VSHALE_METHODS = ('linear', 'clavier', 'larionov_ter', 'larionov_older', 'stieber_1', 'stieber_2', 'stieber_m_pliocene')


def gamma_ray_index(gr, gr_sand:float, gr_shale:float):

    r'''
    Gamma ray index

        IGR = (GR - GR_sand) / (GR_shale - GR_sand), clipped to [0, 1]

    Parameters
    ----------
    gr : array-like
        Gamma ray log

    gr_sand, gr_shale : float
        Clean sand and shale baselines, e.g. the 5th and 95th GR percentiles of the zone.
        If gr_shale <= gr_sand, IGR is 0.5 everywhere

    Example
    -------
    >>> igr = gamma_ray_index(df['GR'], 20, 120)
    '''

    g = as_array(gr)
    if gr_shale > gr_sand:
        igr = np.clip((g - gr_sand) / (gr_shale - gr_sand), 0, 1)
    else:
        igr = np.full(g.shape, .5)
    return like(igr, gr)


def vshale(igr, method:str='linear'):

    r'''
    Volume of shale from the gamma ray index, clipped to [0, 1]

    Parameters
    ----------
    igr : array-like
        Gamma ray index, see `gamma_ray_index`

    method : str default 'linear'
        * 'linear'              Vsh = IGR
        * 'clavier'             Vsh = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)
        * 'larionov_ter'        Vsh = 0.083 (2^(3.7 IGR) - 1)   Tertiary rocks
        * 'larionov_older'      Vsh = 0.33 (2^(2 IGR) - 1)      older rocks
        * 'stieber_1'           Vsh = IGR / (2 - IGR)
        * 'stieber_2'           Vsh = IGR / (4 - 3 IGR)
        * 'stieber_m_pliocene'  Vsh = IGR / (3 - 2 IGR)         Miocene/Pliocene

    Example
    -------
    >>> vsh = vshale(gamma_ray_index(df['GR'], 20, 120), method='clavier')
    '''

    i = as_array(igr)
    if method == 'linear':
        v = i
    elif method == 'clavier':
        v = 1.7 - np.sqrt((3.38 - (i + .7)**2))
    elif method == 'larionov_ter':
        v = .083*((2 ** (3.7*i)) - 1)
    elif method == 'larionov_older':
        v = .33 *((2 ** (2*i)) - 1)
    elif method == 'stieber_1':
        v = i / (2 - i)
    elif method == 'stieber_2':
        v = i / (4 - (3 * i))
    elif method == 'stieber_m_pliocene':
        v = i / (3 - (2 * i))
    else:
        raise ValueError(f"Unknown vshale method '{method}'. Use one of {VSHALE_METHODS}")
    return like(np.clip(v, 0, 1), igr)


def vshale_nd(nphi, phid, nphi_shale:float, phid_shale:float):

    r'''
    Volume of shale from neutron-density separation, clipped to [0, 1]

        Vsh = (PHIN - PHID) / (PHIN_shale - PHID_shale)

    Gas closes the separation, so this reads low in gas zones. Use it as a second
    shale indicator next to the gamma ray, e.g. the minimum of both

    Parameters
    ----------
    nphi : array-like
        Neutron porosity (fraction), in the same matrix units as `phid`

    phid : array-like
        Density porosity (fraction), see `density_porosity`

    nphi_shale, phid_shale : float
        Neutron and density porosity of shale

    Example
    -------
    >>> vsh_nd = vshale_nd(df['NPHI'], density_porosity(df['RHOB']), .33, .18)
    '''

    n, d = as_array(nphi), as_array(phid)
    return like(np.clip((n - d) / (nphi_shale - phid_shale), 0, 1), nphi, phid)


def vshale_sp(sp, sp_clean:float, sp_shale:float):

    r'''
    Volume of shale from the SP log, clipped to [0, 1]

        Vsh = (SP - SP_clean) / (SP_shale - SP_clean)

    Parameters
    ----------
    sp : array-like
        Spontaneous potential (mV)

    sp_clean, sp_shale : float
        SP of a clean, thick water sand (static SP line) and of the shale base line

    Example
    -------
    >>> vsh_sp = vshale_sp(df['SP'], sp_clean=-80, sp_shale=0)
    '''

    return like(np.clip((as_array(sp) - sp_clean) / (sp_shale - sp_clean), 0, 1), sp)
