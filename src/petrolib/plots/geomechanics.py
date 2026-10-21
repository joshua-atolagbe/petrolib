'''
Geomechanics plots: mechanical property tracks, pressure against depth and rock physics cross plots

'''
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .petrophysics import _axes, _scatter, _style
from ..petrophysics.geomechanics import to_gradient


def mechanicalPlot(df:pd.DataFrame, depth:str='DEPTH', curves:dict=None, top:float=None, bottom:float=None,
                   title:str='Mechanical Properties', figsize:tuple=(10, 12)):

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
    Array of Axes

    Example
    -------
    >>> mech = df.join(elastic_moduli(df['RHOB'], df['DT'], df['DTS']))
    >>> mech['BI'] = brittleness(mech['E'], mech['PR'])
    >>> mechanicalPlot(mech)
    '''

    names = {'E': "Young's modulus (GPa)", 'PR': "Poisson's ratio", 'BI': 'Brittleness (%)', 'UCS': 'UCS (MPa)',
             'G': 'Shear modulus (GPa)', 'K': 'Bulk modulus (GPa)'}
    if curves is None:
        curves = [c for c in ('E', 'PR', 'BI', 'UCS') if c in df.columns]
    if not isinstance(curves, dict):
        curves = {c: names.get(c, c) for c in curves}
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, len(curves), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, label), color in zip(axes, curves.items(), colors):
        ax.plot(df[col], df[depth], color=color, linewidth=.8)
        ax.set_title(label, fontsize=11)
        ax.grid(which='major', linestyle=':', linewidth=.9, color='grey')
        ax.xaxis.set_ticks_position('top')
    lo = df[depth].min() if top is None else top
    hi = df[depth].max() if bottom is None else bottom
    axes[0].set_ylim(hi, lo)
    axes[0].set_ylabel('Depth')
    fig.suptitle(title, size=15)
    fig.tight_layout()
    return axes


def pressurePlot(depth, hydrostatic=None, pore=None, overburden=None, fracture=None, mud_weight=None,
                 gradient:bool=False, depth_unit:str='m', title:str='Pressure', figsize:tuple=(7, 10), ax=None):

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
    lines = [(hydrostatic, 'Hydrostatic', 'tab:blue', '--'), (pore, 'Pore pressure', 'tab:red', '-'),
             (fracture, 'Fracture pressure', 'tab:green', '-'), (overburden, 'Overburden', 'black', '-')]
    for values, label, color, style in lines:
        if values is None:
            continue
        v = to_gradient(values, z, 'g/cc', depth_unit) if gradient else np.asarray(values, dtype=float)
        ax.plot(v, z, style, color=color, linewidth=1.2, label=label)
    if mud_weight is not None:
        mw = np.broadcast_to(np.asarray(mud_weight, dtype=float), z.shape)
        if gradient:
            ax.plot(mw, z, '-', color='tab:orange', linewidth=1.5, label='Mud weight')
        else:
            from ..petrophysics.geomechanics import hydrostatic_pressure
            ax.plot(np.asarray(hydrostatic_pressure(z, 1., depth_unit=depth_unit)) * mw, z, '-', color='tab:orange',
                    linewidth=1.5, label='Mud weight')
    ax.set_ylim(np.nanmax(z), np.nanmin(z))
    ax.legend(loc='best', fontsize='small')
    _style(ax, title, 'Gradient (g/cc)' if gradient else 'Pressure (MPa)', f'Depth ({depth_unit})')
    return ax


def rockPhysicsPlot(df:pd.DataFrame, ai:str='AI', vpvs:str='VPVS', hue:str=None, cmap:str='viridis',
                    title:str='Rock Physics Cross Plot', figsize:tuple=(8, 6), ax=None):

    r'''
    Vp/Vs against acoustic impedance. Gas sands plot at low Vp/Vs and low impedance, shales at high Vp/Vs

    Example
    -------
    >>> rockPhysicsPlot(df.join(elastic_moduli(df['RHOB'], df['DT'], df['DTS'])), hue='SW')
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, df[ai], df[vpvs], hue, cmap)
    _style(ax, title, 'Acoustic impedance (m/s g/cc)', 'Vp/Vs')
    return ax
