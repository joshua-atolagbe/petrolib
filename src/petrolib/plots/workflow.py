'''
Plots of the Quanti workflow steps: shale volume, porosity, saturation, permeability and net flags

Each step shows the curves it used and the result, with the zone track; `Quanti` draws these when a
step is run with show_plot=True
'''
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import style as st
from .style import themed
from .tracks import LogLayout, Curve, data_scale

STEPS = ('vshale', 'porosity', 'saturation', 'permeability', 'flags')


def _phi_scale(data:pd.DataFrame) -> tuple:
    #porosity tracks run from a round maximum on the left to 0 on the right
    cols = [c for c in ('PHIT', 'PHIE') if c in data]
    v = data[cols].to_numpy(dtype=float).ravel() if cols else np.array([np.nan])
    top = np.nanpercentile(v, 99.5) if np.isfinite(v).any() else .3
    return (max(.3, float(np.ceil(top * 10) / 10)), 0.)


@themed
def quantiPlot(data:pd.DataFrame, step:str, depth:str='DEPTH', gr:str='GR', rt:str='RT', nphi:str='NPHI',
               rhob:str='RHOB', tops:list=None, bases:list=None, zones:list=None, title:str=None, figsize:tuple=None,
               gr_base:float=None, palette:str=None) -> plt.Figure:

    r'''
    Tracks of one Quanti step

    Parameters
    ----------
    data : pd.DataFrame
        Quanti output with the computed curves (VShale, PHIE, PHIT, SW, Perm, *_NET_FLAG)

    step : str
        'vshale', 'porosity', 'saturation', 'permeability' or 'flags'

    depth, gr, rt, nphi, rhob : str
        Input log columns

    tops, bases, zones : list
        Zone tops, bases and names

    title : str
        Title of the plot; defaults to the step

    figsize : tuple default None
        Size of the figure; sized from the number of tracks when None

    gr_base : float default None
        Gamma ray value separating the sand and shale shading; default 75

    palette : str default None
        Matplotlib colour map for the shale volume shading; default a single shale colour

    Returns
    -------
    matplotlib Figure
    '''

    assert step in STEPS, f'step must be one of {STEPS}'
    tops, bases, zones = list(tops or []), list(bases or []), list(zones or [])
    z = data[depth]
    lo = min(tops + [z.min()]) if tops else z.min()
    hi = max(bases + [z.max()]) if bases else z.max()
    phi = _phi_scale(data)
    df = data.assign(BVW=data['PHIE'] * data['SW']) if {'PHIE', 'SW'} <= set(data.columns) else data

    def vsh_track(lay, i):
        c = Curve('VShale', label='VSH', unit='v/v', color=st.INK2, scale=(0., 1.), lw=.8)
        if palette is not None:
            lay.gradient(lay.tracks[i], df[depth].to_numpy(float), df['VShale'].to_numpy(float), 0.,
                         df['VShale'].to_numpy(float), palette, (0., 1.))
            lay.curves(i, df, depth, [c])
        else:
            lay.curves(i, df, depth, [c])
            lay.shade(i, df, depth, c, 'left', st.SHALE, 'shale', .8)
            lay.shade(i, df, depth, c, 'right', st.SAND, 'sand', .55)

    def phi_track(lay, i):
        pt = Curve('PHIT', unit='v/v', color=st.SERIES[0], scale=phi, lw=.8)
        pe = Curve('PHIE', unit='v/v', color=st.INK, scale=phi, lw=.9)
        lay.curves(i, df, depth, [pt, pe])
        lay.shade(i, df, depth, pe, 'right', st.WATER, 'effective', .45)
        lay.shade(i, df, depth, pt, pe, st.SHALE, 'clay bound', .6)

    def fluid_track(lay, i):
        pe = Curve('PHIE', unit='v/v', color=st.INK, scale=phi, lw=.9)
        bvw = Curve('BVW', unit='v/v', color=st.SERIES[0], scale=phi, lw=.7)
        lay.curves(i, df, depth, [pe, bvw])
        lay.shade(i, df, depth, bvw, 'right', st.WATER, 'water', .7)
        lay.shade(i, df, depth, pe, bvw, st.HYDROCARBON, 'hydrocarbon', .6)

    def sw_track(lay, i):
        c = Curve('SW', unit='v/v', color=st.SERIES[0], scale=(1., 0.), lw=.9)
        lay.curves(i, df, depth, [c])
        lay.shade(i, df, depth, c, 'right', st.WATER, 'water', .5)
        lay.shade(i, df, depth, c, 'left', st.HYDROCARBON, 'hydrocarbon', .45)

    def perm_track(lay, i):
        lo, hi = data_scale(df['Perm'], True)
        c = Curve('Perm', unit='mD', color=st.SERIES[6], scale=(max(lo, .01), max(hi, 1000.)), log=True, lw=.9)
        lay.curves(i, df, depth, [c])
        lay.shade(i, df, depth, c, 'left', st.SERIES[6], None, .15)

    def gr_track(lay, i):
        lay.curves(i, df, depth, [Curve(gr, color=st.CURVE_COLORS['GR'])], fill='sand', baseline=gr_base or 75.)

    def rt_track(lay, i):
        lay.curves(i, df, depth, [Curve(rt, color=st.CURVE_COLORS['RT'], log=True)])

    def nd_track(lay, i):
        lay.curves(i, df, depth, [Curve(nphi, color=st.CURVE_COLORS['NPHI'], scale=(.45, -.15), unit='v/v',
                                        ls=(0, (4, 1.5))),
                                  Curve(rhob, color=st.CURVE_COLORS['RHOB'], scale=(1.95, 2.95), unit='g/cm3')],
                   crossover='both')

    def rhob_track(lay, i):
        lay.curves(i, df, depth, [Curve(rhob, color=st.CURVE_COLORS['RHOB'])])

    plans = {
        'vshale': ('Shale volume', [(gr_track, 1), (vsh_track, 1)]),
        'porosity': ('Porosity', [(rhob_track, 1), (vsh_track, 1), (phi_track, 1.2)]),
        'saturation': ('Water saturation', [(rt_track, 1), (sw_track, 1), (fluid_track, 1.2)]),
        'permeability': ('Permeability', [(phi_track, 1.2), (perm_track, 1)]),
        'flags': ('Net flags', [(gr_track, 1), (rt_track, 1), (nd_track, 1.25), (vsh_track, 1), (fluid_track, 1.2),
                                (sw_track, 1), (perm_track, 1)]),
    }
    name, tracks = plans[step]
    flags = [f for f in ('ROCK_NET_FLAG', 'RES_NET_FLAG', 'PAY_NET_FLAG') if f in df] if step == 'flags' else []
    widths = [w for _, w in tracks] + ([.75] if zones else []) + ([.9] if flags else [])
    lay = LogLayout(lo, hi, widths, 4, title or name, f'{lo:g}–{hi:g}', figsize,
                    'Depth' if str(depth).upper() == 'DEPTH' else depth)
    for i, (draw, _) in enumerate(tracks):
        draw(lay, i)
    if zones:
        lay.zones(len(tracks), tops, bases, zones)
    if flags:
        lay.flags(len(widths) - 1, df, depth, flags)
    return lay.finish()
