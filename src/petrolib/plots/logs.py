'''
Log, combo and lithofacies track plots

Every track has a header box with the curve name, unit and the scale at each edge. Curve types are
recognised from their mnemonics (petrolib.mnemonics), which set the colour and the default scale:
gamma ray 0-150 gAPI, resistivity 0.2-2000 ohm.m on a log scale, and neutron and density on
limestone-compatible scales (0.45 to -0.15 v/v and 1.95 to 2.95 g/cm3), so that the two curves
overlie in clean water-bearing limestone and cross over in gas.
'''
from __future__ import annotations
import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..mnemonics import log_kind
from . import style as st
from .style import themed
from .tracks import LogLayout, Curve


def _window(df:pd.DataFrame, depth:str, top:float, bottom:float) -> pd.DataFrame:
    #samples in the plotted interval, with a margin so curves reach the track edges
    lo, hi = min(top, bottom), max(top, bottom)
    pad = .02 * (hi - lo)
    return df[df[depth].between(lo - pad, hi + pad)]


def _depth_label(depth:str) -> str:
    return 'Depth' if str(depth).upper() in ('DEPTH', 'DEPT') else str(depth)


def _gr_fill(limit:str, palette_op:str) -> dict:
    #sand-shale shading about 75 gAPI, or a colour ramp from one edge when a side or palette is given
    if limit is None and palette_op is None:
        return {'fill': 'sand', 'baseline': 75.}
    return {'fill': limit or 'left', 'cmap': palette_op}


def _crossover(fill:str) -> str:
    #1.x names: 'left' shows porous (crossover) zones, 'right' non-porous (separation), 'both' both
    return {'left': 'gas', 'right': 'shale', 'both': 'both'}.get(fill)


def _combo(layout:LogLayout, df:pd.DataFrame, depth:str, gr:str, res:str, nphi:str, rhob:str, res_thres:float,
           fill:str, palette_op:str, limit:str):
    #gamma ray, resistivity and neutron-density tracks 0-2
    layout.curves(0, df, depth, [Curve(gr, color=st.CURVE_COLORS['GR'])], **_gr_fill(limit, palette_op))
    layout.curves(1, df, depth, [Curve(res, color=st.CURVE_COLORS['RT'], log=True,
                                       scale=None if log_kind(res) else (.2, 2000.))], fill='above', threshold=res_thres)
    #neutron in percent or density in kg/m3 get the same scales in their units
    pct = df[nphi].median() > 1.5
    kgm3 = df[rhob].median() > 100
    layout.curves(2, df, depth, [Curve(nphi, color=st.CURVE_COLORS['NPHI'], scale=(45, -15) if pct else (.45, -.15),
                                       unit='%' if pct else 'v/v', ls=(0, (4, 1.5))),
                                 Curve(rhob, color=st.CURVE_COLORS['RHOB'], scale=(1950, 2950) if kgm3 else (1.95, 2.95),
                                       unit='kg/m3' if kgm3 else 'g/cm3')],
                  crossover=_crossover(fill))


def _combo_rows(fill:str, limit:str, palette_op:str) -> int:
    #header rows the combo tracks need: curves plus the fills shown in the header
    nd = 2 + {'gas': 1, 'shale': 1, 'both': 2}.get(_crossover(fill), 0)
    gr = 2 if limit is None and palette_op is None else 1
    return max(nd, gr, 2)


@themed
def plotFaciesZoneCombo(data:pd.DataFrame, depth:str, gr:str, res:str, nphi:str, rhob:str, facies:str,
                        ztop:float, zbot:float, facies_colors:list[str], facies_labels:list[str],
                        ztops:list[float], zbots:list[float], zonename:list[str], limit:str=None, res_thres:float=10.,
                        palette_op:str=None, fill:str=None, flag_name:str='Lithofacies',
                        title:str='Log plot', figsize:tuple=None) -> plt.Figure:
    r'''
    Gamma ray, resistivity and neutron-density tracks with a zone track and a facies track

    Parameters
    ----------
    data : pd.DataFrame
        Log data

    depth, gr, res, nphi, rhob : str
        Depth, gamma ray, resistivity, neutron porosity and bulk density columns

    facies : str
        Facies column, as codes 0, 1, ... in the order of `facies_labels`, or as the labels themselves

    ztop, zbot : float
        Depth range shown

    facies_colors : list of str
        Colour of each facies, in the order of `facies_labels`

    facies_labels : list of str
        Name of each facies

    ztops, zbots : list of float
        Top and base depth of each zone

    zonename : list of str
        Name of each zone

    limit : str default None
        Edge the gamma ray is shaded from with a colour ramp, 'left' or 'right'. With neither `limit` nor
        `palette_op`, sand (yellow) and shale (grey) are shaded on either side of 75 gAPI

    res_thres : float default 10
        Resistivity above which the resistivity track is shaded as hydrocarbon bearing

    palette_op : str default None
        Matplotlib colour map for the gamma ray ramp; defaults to a sand-to-shale ramp

    fill : str default None
        Neutron-density shading: 'left' where they cross over (porous, gas), 'right' where they separate
        (shale), 'both', or None

    flag_name : str
        Header of the facies track

    title : str
        Title of the plot

    figsize : tuple default None
        Size of the figure in inches; sized from the number of tracks when None

    Returns
    -------
    matplotlib Figure

    Example
    -------
    >>> from petrolib.plots import plotFaciesZoneCombo
    >>> plotFaciesZoneCombo(well11, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', 'LITHO', 5500, 9000,
    ...                     ['#f2cf6b', '#b4aa98'], ['Sandstone', 'Shale'],
    ...                     ztops=[5561.4, 7116.5], zbots=[5761.5, 7351.2], zonename=['SAND A', 'SAND B'],
    ...                     fill='both', title='ATAGA-11')
    '''

    assert len(data[facies].dropna().unique()) <= len(facies_labels) == len(facies_colors), \
        'Number of unique facies must not exceed the number of facies labels and colors'
    assert len(ztops) == len(zbots) == len(zonename), 'Number of tops, bottoms and zonename must be equal'
    df = _window(data, depth, ztop, zbot)
    layout = LogLayout(ztop, zbot, [1, 1, 1.3, .75, .55], _combo_rows(fill, limit, palette_op), title,
                       f'{min(ztop, zbot):g}–{max(ztop, zbot):g}', figsize, _depth_label(depth), legend_rows=1)
    _combo(layout, df, depth, gr, res, nphi, rhob, res_thres, fill, palette_op, limit)
    layout.zones(3, ztops, zbots, zonename)
    codes = df[facies]
    if not pd.api.types.is_numeric_dtype(codes):
        codes = codes.map({label: i for i, label in enumerate(facies_labels)})
    layout.facies(4, df, depth, codes, facies_colors, facies_labels, flag_name)
    return layout.finish()


@themed
def tripleCombo(data:pd.DataFrame, depth:str, gr:str, res:str, nphi:str, rhob:str, ztop:float, zbot:float,
                res_thres:float=10.0, fill:str=None, palette_op:str=None,
                limit:str=None, figsize:tuple=None, title:str='Three Combo Log Plot') -> plt.Figure:

    r'''
    Triple-combo display: gamma ray, resistivity and neutron-density tracks

    Parameters
    ----------
    data : pd.DataFrame
        Log data

    depth, gr, res, nphi, rhob : str
        Depth, gamma ray, resistivity, neutron porosity and bulk density columns

    ztop, zbot : float
        Depth range shown

    res_thres : float default 10
        Resistivity above which the resistivity track is shaded as hydrocarbon bearing

    fill : str default None
        Neutron-density shading: 'left' where they cross over (porous, gas), 'right' where they separate
        (shale), 'both', or None

    palette_op : str default None
        Matplotlib colour map for the gamma ray ramp; defaults to a sand-to-shale ramp

    limit : str default None
        Edge the gamma ray is shaded from with a colour ramp, 'left' or 'right'. With neither `limit` nor
        `palette_op`, sand (yellow) and shale (grey) are shaded on either side of 75 gAPI

    figsize : tuple default None
        Size of the figure in inches; sized from the number of tracks when None

    title : str
        Title of the plot

    Returns
    -------
    matplotlib Figure

    Example
    -------
    >>> from petrolib.plots import tripleCombo
    >>> tripleCombo(df, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', ztop=3300, zbot=3450, res_thres=10, fill='both')
    '''

    df = _window(data, depth, ztop, zbot)
    layout = LogLayout(ztop, zbot, [1, 1, 1.3], _combo_rows(fill, limit, palette_op), title,
                       f'{min(ztop, zbot):g}–{max(ztop, zbot):g}', figsize, _depth_label(depth))
    _combo(layout, df, depth, gr, res, nphi, rhob, res_thres, fill, palette_op, limit)
    return layout.finish()


def _track_curves(track) -> list:
    return list(track) if isinstance(track, (list, tuple)) else [track]


def _overlay_curves(names:list) -> list:
    #a neutron and a density curve in one track use the compatible scales, neutron dashed
    kinds = [log_kind(n) for n in names]
    curves = []
    for n, k in zip(names, kinds):
        if k == 'NPHI' and 'RHOB' in kinds:
            curves.append(Curve(n, scale=(.45, -.15), ls=(0, (4, 1.5))))
        elif k == 'RHOB' and 'NPHI' in kinds:
            curves.append(Curve(n, scale=(1.95, 2.95)))
        else:
            curves.append(Curve(n))
    #curves of one type share a colour; give repeats the next categorical colours
    seen = {}
    for c in curves:
        color = st.curve_color(c.name)
        if color in seen.values():
            color = next(s for s in st.SERIES if s not in seen.values())
        seen[c.name] = color
        c.color = color
    return curves


def _log_tracks(df:pd.DataFrame, depth:str, logs:list, top:float, bottom:float, title:str, figsize:tuple,
                extra:int=0, legend_rows:int=0) -> LogLayout:
    tracks = [_overlay_curves(_track_curves(t)) for t in logs]
    rows = max([len(t) for t in tracks] + [2])
    widths = [1.15 if len(t) > 1 else 1. for t in tracks] + [.6] * extra
    layout = LogLayout(top, bottom, widths, rows, title, f'{min(top, bottom):g}–{max(top, bottom):g}', figsize,
                       _depth_label(depth), legend_rows)
    for i, curves in enumerate(tracks):
        layout.curves(i, df, depth, curves)
    return layout


@themed
def plotLog(df:pd.DataFrame, depth:str, logs:List[str], top:float, bottom:float, title:str='Log Plot',
            figsize:tuple=None) -> plt.Figure:

    r'''
    Plots each log curve in its own track. To overlay curves in one track use `plotLogs`

    Parameters
    ----------
    df : pd.DataFrame
        Log data

    depth : str
        Depth column

    logs : list of str
        Curves to plot, one track each

    top, bottom : float
        Depth range shown

    title : str
        Title of the plot

    figsize : tuple default None
        Size of the figure in inches; sized from the number of tracks when None

    Returns
    -------
    matplotlib Figure

    Example
    -------
    >>> from petrolib.plots import plotLog
    >>> plotLog(df, 'DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, 'Volve')
    '''

    logs = [logs] if isinstance(logs, str) else list(logs)
    return _log_tracks(_window(df, depth, top, bottom), depth, logs, top, bottom, title, figsize).finish()


@themed
def plotZoneCombo(data:pd.DataFrame, depth:str, gr:str, res:str, nphi:str, rhob:str, ztop:float, zbot:float,
                  ztops:List[float], zbots:List[float], zonename:List[str], limit:str=None, res_thres:float=10.,
                  palette_op:str=None, fill:str=None, title:str='Log plot', figsize:tuple=None) -> plt.Figure:

    r'''
    Gamma ray, resistivity and neutron-density tracks with a zone track; zone boundaries cross every track

    Parameters
    ----------
    data : pd.DataFrame
        Log data

    depth, gr, res, nphi, rhob : str
        Depth, gamma ray, resistivity, neutron porosity and bulk density columns

    ztop, zbot : float
        Depth range shown

    ztops, zbots : list of float
        Top and base depth of each zone

    zonename : list of str
        Name of each zone

    limit : str default None
        Edge the gamma ray is shaded from with a colour ramp, 'left' or 'right'. With neither `limit` nor
        `palette_op`, sand (yellow) and shale (grey) are shaded on either side of 75 gAPI

    res_thres : float default 10
        Resistivity above which the resistivity track is shaded as hydrocarbon bearing

    palette_op : str default None
        Matplotlib colour map for the gamma ray ramp; defaults to a sand-to-shale ramp

    fill : str default None
        Neutron-density shading: 'left' where they cross over (porous, gas), 'right' where they separate
        (shale), 'both', or None

    title : str
        Title of the plot

    figsize : tuple default None
        Size of the figure in inches; sized from the number of tracks when None

    Returns
    -------
    matplotlib Figure

    Example
    -------
    >>> from petrolib.plots import plotZoneCombo
    >>> plotZoneCombo(well11, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', min(ztop), max(zbot),
    ...               ztop, zbot, zn, fill='both', title='ATAGA-11')
    '''

    assert len(ztops) == len(zbots) == len(zonename), 'Number of tops, bottoms and zonename must be equal'
    df = _window(data, depth, ztop, zbot)
    layout = LogLayout(ztop, zbot, [1, 1, 1.3, .8], _combo_rows(fill, limit, palette_op), title,
                       f'{min(ztop, zbot):g}–{max(ztop, zbot):g}', figsize, _depth_label(depth))
    _combo(layout, df, depth, gr, res, nphi, rhob, res_thres, fill, palette_op, limit)
    layout.zones(3, ztops, zbots, zonename)
    return layout.finish()


def _lithology_table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'litho_info.csv'))


@themed
def plotLogFacies(df:pd.DataFrame, depth:str, logs:List[list], top:float, bottom:float, facies:str=None,
                  title:str='Log Plot', figsize:tuple=None) -> plt.Figure:

    r'''
    Log tracks, with curves in a list sharing one track, and an optional lithology track

    Parameters
    ----------
    df : pd.DataFrame
        Log data

    depth : str
        Depth column

    logs : list
        Curves to plot; a list inside the list puts those curves in one track, e.g.
        ['GR', 'RT', ['NPHI', 'RHOB']]

    top, bottom : float
        Depth range shown

    facies : str default None
        Lithology column, as names (e.g. 'Sandstone', 'Shale', from `model_facies`) or as the
        lithology numbers of petrolib's lithology table

    title : str
        Title of the plot

    figsize : tuple default None
        Size of the figure in inches; sized from the number of tracks when None

    Returns
    -------
    matplotlib Figure

    Example
    -------
    >>> from petrolib.plots import plotLogFacies
    >>> plotLogFacies(df, 'DEPTH', ['GR', 'RT', ['NPHI', 'RHOB']], 3300, 3600, facies='litho', title='Volve')
    '''

    data = _window(df, depth, top, bottom)
    layout = _log_tracks(data, depth, list(logs), top, bottom, title, figsize, extra=facies is not None,
                         legend_rows=facies is not None)
    if facies is not None:
        table = _lithology_table()
        values = data[facies]
        if pd.api.types.is_numeric_dtype(values):
            names = dict(zip(table['lith_num'], table['lith']))
            unknown = set(values.dropna().unique()) - set(names)
            values = values.map(names)
        else:
            unknown = set(values.dropna().unique()) - set(table['lith'])
        if unknown:
            raise ValueError(f"Unidentified facies in '{facies}': {sorted(map(str, unknown))}")
        order = [l for l in table['lith'] if l in set(values.dropna())]
        colors = [st.LITHOLOGY_COLORS.get(l, c) for l, c in zip(table['lith'], table['color']) if l in order]
        layout.facies(len(logs), data, depth, values.map({l: i for i, l in enumerate(order)}), colors, order, 'Lithology')
    return layout.finish()


@themed
def plotLogs(df:pd.DataFrame, depth:str, logs:List[List[str]], top:float, bottom:float,
             title:str='Log Plot', figsize:tuple=None) -> plt.Figure:

    r'''
    Log tracks, with curves in a list sharing one track (overlay tracks)

    Parameters
    ----------
    df : pd.DataFrame
        Log data

    depth : str
        Depth column

    logs : list
        Curves to plot; a list inside the list puts those curves in one track, e.g.
        ['GR', 'RT', ['NPHI', 'RHOB']]. Neutron and density in one track use compatible scales

    top, bottom : float
        Depth range shown

    title : str
        Title of the plot

    figsize : tuple default None
        Size of the figure in inches; sized from the number of tracks when None

    Returns
    -------
    matplotlib Figure

    Example
    -------
    >>> from petrolib.plots import plotLogs
    >>> plotLogs(df, 'DEPTH', ['GR', 'RT', ['NPHI', 'RHOB'], 'DT'], 3300, 3600, 'Volve')
    '''

    return _log_tracks(_window(df, depth, top, bottom), depth, list(logs), top, bottom, title, figsize).finish()
