'''
Geomechanics plots: mechanical property tracks, pressure against depth and rock physics cross plots

'''
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import style as st
from .style import themed
from .petrophysics import _axes, _scatter, _style
from .tracks import LogLayout, Curve, data_scale
from ..petrophysics.geomechanics import to_gradient


@themed
def mechanicalPlot(df:pd.DataFrame, depth:str='DEPTH', curves:dict=None, top:float=None, bottom:float=None,
                   title:str='Mechanical properties', figsize:tuple=None):

    r'''
    Tracks of mechanical properties against depth

    Parameters
    ----------
    df : pd.DataFrame
        Data, e.g. with the columns of `petrolib.petrophysics.elastic_moduli`

    depth : str default 'DEPTH'
        Depth column

    curves : list or dict default None
        Columns to show, or column -> track title. Default: those of E, PR, BI and UCS that are present.
        G and K tracks can be added by name

    top, bottom : float default None
        Depth range. Default: the whole data

    Returns
    -------
    Array of the track Axes

    Example
    -------
    >>> mech = df.join(elastic_moduli(df['RHOB'], df['DT'], df['DTS']))
    >>> mech['BI'] = brittleness(mech['E'], mech['PR'])
    >>> mechanicalPlot(mech)
    '''

    names = {'E': ("Young's modulus", 'GPa'), 'PR': ("Poisson's ratio", ''), 'BI': ('Brittleness', '%'),
             'UCS': ('UCS', 'MPa'), 'G': ('Shear modulus', 'GPa'), 'K': ('Bulk modulus', 'GPa')}
    if curves is None:
        curves = [c for c in ('E', 'PR', 'BI', 'UCS') if c in df.columns]
    if not isinstance(curves, dict):
        curves = {c: names.get(c, (c, ''))[0] for c in curves}
    lo = df[depth].min() if top is None else top
    hi = df[depth].max() if bottom is None else bottom
    data = df[df[depth].between(min(lo, hi), max(lo, hi))]
    layout = LogLayout(lo, hi, [1] * len(curves), 1, title, f'{min(lo, hi):g}–{max(lo, hi):g}', figsize,
                       'Depth' if str(depth).upper() == 'DEPTH' else depth)
    for i, ((col, label), color) in enumerate(zip(curves.items(), st.categorical(len(curves)))):
        layout.curves(i, data, depth, [Curve(col, label=label, unit=names.get(col, ('', ''))[1], color=color,
                                             scale=data_scale(data[col]), lw=.9)])
    layout.finish()
    return np.array(layout.tracks)


@themed
def pressurePlot(depth, hydrostatic=None, pore=None, overburden=None, fracture=None, mud_weight=None,
                 gradient:bool=False, depth_unit:str='m', title:str='Pressure', figsize:tuple=(5.5, 8), ax=None):

    r'''
    Hydrostatic, pore, overburden and fracture pressure (MPa) against depth, as pressures or as
    equivalent mud weight gradients (g/cc)

    Parameters
    ----------
    depth : array-like
        True vertical depth

    hydrostatic, pore, overburden, fracture : array-like default None
        Pressures in MPa, e.g. from `petrolib.petrophysics` hydrostatic_pressure, pore_pressure_eaton,
        overburden and fracture_pressure_eaton

    mud_weight : array-like or float default None
        Mud weight (g/cc), drawn as a gradient

    gradient : bool default False
        Show equivalent mud weight gradients instead of pressures

    Example
    -------
    >>> pressurePlot(df['TVD'], ph, pp, sv, pf, mud_weight=1.2, gradient=True)
    '''

    ax = _axes(ax, figsize)
    z = np.asarray(depth, dtype=float)
    lines = [(hydrostatic, 'Hydrostatic', st.SERIES[0], (0, (4, 2))), (pore, 'Pore pressure', st.SERIES[7], '-'),
             (fracture, 'Fracture pressure', st.SERIES[5], '-'), (overburden, 'Overburden', st.INK, '-')]
    if mud_weight is not None:
        mw = np.broadcast_to(np.asarray(mud_weight, dtype=float), z.shape)
        if not gradient:
            from ..petrophysics.geomechanics import hydrostatic_pressure
            mw = np.asarray(hydrostatic_pressure(z, 1., depth_unit=depth_unit)) * mw
        lines.append((mw, 'Mud weight', st.SERIES[1], '-'))
    for values, label, color, style in lines:
        if values is None:
            continue
        v = np.asarray(to_gradient(values, z, 'g/cc', depth_unit) if gradient and label != 'Mud weight' else values,
                       dtype=float)
        ax.plot(v, z, ls=style, color=color, lw=1.4, label=label, zorder=3)
    ax.set_ylim(np.nanmax(z), np.nanmin(z))
    st.legend(ax, loc='best')
    _style(ax, title, 'Equivalent gradient (g/cm³)' if gradient else 'Pressure (MPa)', f'Depth ({depth_unit})')
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.spines['top'].set_visible(True); ax.spines['bottom'].set_visible(False)
    return ax


@themed
def rockPhysicsPlot(df:pd.DataFrame, ai:str='AI', vpvs:str='VPVS', hue:str=None, cmap:str=None,
                    title:str='Rock physics cross plot', figsize:tuple=(7, 5.5), ax=None):

    r'''
    Vp/Vs against acoustic impedance. Gas sands plot at low Vp/Vs and low impedance, shales at high Vp/Vs

    Example
    -------
    >>> rockPhysicsPlot(df.join(elastic_moduli(df['RHOB'], df['DT'], df['DTS'])), hue='SW')
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, df[ai], df[vpvs], hue, cmap)
    _style(ax, title, 'Acoustic impedance (m/s · g/cm³)', 'Vp/Vs', f'{int(df[[ai, vpvs]].notna().all(axis=1).sum()):,} samples')
    return ax
