'''
Neutron-density and Pickett cross plots

'''
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import style as st
from .style import themed
from .petrophysics import _axes, _scatter, _style, _ref


def _hue_type(df:pd.DataFrame, hue:str, color_code:str):
    #'num' or 'cat' as given, inferred from the column when only hue is given
    if hue is None:
        return None
    if color_code is not None:
        return color_code
    return 'num' if pd.api.types.is_numeric_dtype(df[hue]) else 'cat'


def _colored(ax, df, x, y, hue, code, cmap):
    if code == 'cat':
        _scatter(ax, df.assign(**{hue: df[hue].astype(str)}), x, y, hue)
    elif code == 'num':
        _scatter(ax, df, x, y, hue, cmap)
    else:
        _scatter(ax, df, x, y)


@themed
def crossPlot(df:pd.DataFrame, column_x:str, column_y:str, hue:str=None,
              color_code:str=None, figsize:tuple=(7, 6), rhob_fluid:float=1., res_name:str=None, cmap=None, ax=None):

    r'''
    Neutron-density cross plot with the sandstone, limestone and dolomite lines on limestone-compatible
    scales, labelled in porosity, for reservoir and fluid type identification. Gas moves points up
    and to the left of the lines, shale down and to the right

    The matrix lines are fits to the neutron-density chart for a CNL tool, after code by Yohanes Nuwara

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of well

    column_x : str
        Neutron porosity column (v/v, limestone units)

    column_y : str
        Bulk density column (g/cm3)

    hue : str
        Column to color code the points by

    color_code : str default None
        'num' for a continuous `hue`, 'cat' for a categorical one; inferred from the column when None

    figsize : tuple
        Size of plot

    rhob_fluid : float, default 1.0
        Fluid density

    res_name : str
        Reservoir name, shown under the title

    cmap : str
        Colour map for a continuous `hue`

    ax : matplotlib Axes default None
        Axes to draw on

    Returns
    -------
    matplotlib Axes

    Example
    -------
    >>> from petrolib.plots import crossPlot
    >>> crossPlot(df=df, column_x='NPHI', column_y='RHOB', res_name='RES_A', hue='GR')
    '''

    ax = _axes(ax, figsize)
    lsX = np.arange(0, 0.45001, 0.05)
    ssCnlX = np.array([np.roots([0.222, 1.021, 0.039 - i])[1] for i in lsX])
    dolCnlX = np.array([np.roots([1.40, 0.389, -0.01259 - i])[1] for i in lsX])
    matrix = {'Sandstone': (ssCnlX, 2.65), 'Limestone': (lsX, 2.71), 'Dolomite': (dolCnlX, 2.87)}
    lines = {name: (x, (rhob_fluid - rho) * lsX + rho) for name, (x, rho) in matrix.items()}

    #iso-porosity lines, then the matrix lines with a dot at every 0.05 porosity
    for k in range(len(lsX)):
        _ref(ax, [lines[n][0][k] for n in lines], [lines[n][1][k] for n in lines], color=st.GRID, lw=.8, zorder=1)
    colors = dict(zip(matrix, st.SERIES[:3]))
    for name, (x, y) in lines.items():
        ax.plot(x, y, color=colors[name], lw=1.6, zorder=4, solid_capstyle='round')
        ax.scatter(x, y, s=14, color=colors[name], edgecolor=st.SURFACE, linewidths=.8, zorder=5)
        st.line_label(ax, x[-2] + .015, y[-2] + .015, name, color=st.INK, ha='left', va='top', fontsize=8,
                      fontweight='semibold')
    for p, x, y in zip(lsX, *lines['Limestone']):
        if round(p * 100) % 10 == 0:
            st.line_label(ax, x - .008, y + .02, f'{p:.1f}', color=st.INK2, ha='right', va='top', fontsize=7)

    _colored(ax, df, df[column_x].to_numpy(float), df[column_y].to_numpy(float), hue,
             _hue_type(df, hue, color_code), cmap)
    y = df[column_y].to_numpy(float)
    ax.set_xlim(-.1, .6)
    ax.set_ylim(max(3., np.nanpercentile(y, 99.8) + .05), min(1.9, np.nanpercentile(y, .2) - .05))
    _style(ax, 'Neutron–density cross plot', f'{column_x} (v/v, limestone units)', f'{column_y} (g/cm³)',
           ' · '.join(s for s in (res_name, f'{int(np.isfinite(y).sum()):,} samples') if s))
    return ax


@themed
def picketPlot(df:pd.DataFrame, rt:str='RT', por:str='NPHI', rwa:float=0.018,
               a:float=1., m:float=1.8, n:float=2., res_name:str=None, figsize:tuple=(7, 6),
               hue:str=None, color_code:str=None, cmap=None, ax=None):

    r'''
    Pickett plot: porosity against resistivity on log scales, with Archie lines of constant water
    saturation. Water-bearing points fall on the SW = 1 line, whose slope is -1/m and whose
    intercept at porosity 1 is a Rw. Read more here: https://wiki.seg.org/wiki/Pickett_plot

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    rt : str
        Resistivity column

    por : str
        Porosity column

    rwa : float default 0.018
        Formation water resistivity

    a : float default 1
        Tortuosity factor

    m : float default 1.8
        Cementation exponent

    n : float default 2
        Saturation exponent

    res_name : str
        Reservoir/zone name, shown under the title

    figsize : tuple
        Size of plot

    hue : str
        Column to color code the points by

    color_code : str default None
        'num' for a continuous `hue`, 'cat' for a categorical one; inferred from the column when None

    cmap : str
        Colour map for a continuous `hue`

    ax : matplotlib Axes default None
        Axes to draw on

    Returns
    -------
    matplotlib Axes

    Example
    -------
    >>> from petrolib.plots import picketPlot
    >>> picketPlot(df, rt='RT', por='PHIE', rwa=.03, m=2, hue='VShale')
    '''

    ax = _axes(ax, figsize)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(.1, 1000); ax.set_ylim(.01, 1)
    sw = (1., .8, .6, .4, .2, .1)
    phi = np.array([.01, 1.])
    for s in sw:
        r = a * rwa / s**n / phi**m
        _ref(ax, r, phi, color=st.INK2 if s == 1 else st.MUTED, lw=1.1 if s == 1 else .8)
        #label each line along its slope at porosity 0.4, above the usual data
        x_label = a * rwa / s**n / .4**m
        if .1 < x_label < 800:
            angle = np.degrees(np.arctan2(-.4 / (m * x_label), 1.))
            st.line_label(ax, x_label * 1.06, .4, f'SW {s * 100:g}%', ha='left', va='bottom', rotation=angle,
                          rotation_mode='anchor', transform_rotates_text=True)
    _colored(ax, df, df[rt].to_numpy(float), df[por].to_numpy(float), hue, _hue_type(df, hue, color_code), cmap)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(.1, 1000); ax.set_ylim(.01, 1)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.grid(True, which='minor', color=st.GRID_MINOR, lw=.4)
    _style(ax, 'Pickett plot', f'{rt} (ohm.m)', f'{por} (v/v)',
           ' · '.join(s for s in (res_name, f'Rw {rwa:g} ohm.m, a {a:g}, m {m:g}, n {n:g}') if s))
    ax.grid(True, which='minor', color=st.GRID_MINOR, lw=.4)
    return ax
