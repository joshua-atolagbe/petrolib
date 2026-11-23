'''
Log track layout shared by the log plots: header boxes with curve scales, depth tracks, fills, zones,
flags and facies

Each track is a pair of axes, a header above and the track below. Curves are drawn on a 0-1 scale
inside the track, so several curves with different scales share one track without twin axes, and
fills between curves (such as the neutron-density crossover) are computed on that common scale.

Example
-------
>>> from petrolib.plots.tracks import LogLayout, Curve
>>> lay = LogLayout(top=2760, bottom=3000, widths=[1, 1, 1.3], title='KPD-12')
>>> lay.curves(0, df, 'DEPTH', [Curve('GR')], fill='sand')
'''
from __future__ import annotations
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from ..mnemonics import log_kind
from . import style as st

#default track scales by curve type: (left, right, logarithmic, unit)
SCALES = {
    'GR':   (0., 150., False, 'gAPI'),
    'SP':   (-100., 20., False, 'mV'),
    'CALI': (6., 16., False, 'in'),
    'BS':   (6., 16., False, 'in'),
    'RT':   (.2, 2000., True, 'ohm.m'),
    'RM':   (.2, 2000., True, 'ohm.m'),
    'RXO':  (.2, 2000., True, 'ohm.m'),
    'NPHI': (.45, -.15, False, 'v/v'),
    'RHOB': (1.95, 2.95, False, 'g/cm3'),
    'DRHO': (-.25, .25, False, 'g/cm3'),
    'PEF':  (0., 10., False, 'b/e'),
    'DT':   (140., 40., False, 'us/ft'),
    'DTS':  (240., 40., False, 'us/ft'),
}
#computed curves petrolib writes
COMPUTED = {
    'VSHALE': (0., 1., False, 'v/v'), 'VSH': (0., 1., False, 'v/v'), 'IGR': (0., 1., False, 'v/v'),
    'PHIE': (.5, 0., False, 'v/v'), 'PHIT': (.5, 0., False, 'v/v'), 'PHID': (.5, 0., False, 'v/v'),
    'PHIN': (.5, 0., False, 'v/v'), 'PHIS': (.5, 0., False, 'v/v'),
    'SW': (1., 0., False, 'v/v'), 'SXO': (1., 0., False, 'v/v'), 'BVW': (.5, 0., False, 'v/v'),
    'PERM': (.01, 10000., True, 'mD'), 'K': (.01, 10000., True, 'mD'),
}

HEADER_ROW = .30        # inches per header row
TITLE_BAND = .62        # inches for the title and subtitle


@dataclass
class Curve:

    r'''
    A curve in a track: column name, header label, unit, colour, scale (left, right), log scale and line style.
    Missing values are filled from the curve type of the column name
    '''

    name: str
    label: str = None
    unit: str = None
    color: str = None
    scale: tuple = None
    log: bool = None
    ls: str = '-'
    lw: float = 1.

    def resolved(self, data:pd.Series=None) -> 'Curve':
        kind = log_kind(self.name)
        key = str(self.name).upper()
        lo, hi, log, unit = SCALES.get(kind) or COMPUTED.get(key) or COMPUTED.get(key.split('_')[0]) or (None, None, None, '')
        if self.scale is not None:
            lo, hi = self.scale
        log = self.log if self.log is not None else bool(log)
        if lo is None and data is not None:
            lo, hi = data_scale(data, log)
        return Curve(self.name, self.label or str(self.name), unit if self.unit is None else self.unit,
                     self.color or st.curve_color(self.name), (lo, hi), log, self.ls, self.lw)


def data_scale(values, log:bool=False) -> tuple:

    r'''
    A rounded scale covering the 1st to 99th percentile of the values
    '''

    v = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    if v.size == 0:
        return (.1, 1000.) if log else (0., 1.)
    lo, hi = np.percentile(v, [1, 99])
    if log:
        return 10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi))
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    step = st.nice_step(hi - lo, 5)
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def normalize(values, scale:tuple, log:bool=False) -> np.ndarray:

    r'''
    Position of values across a track, 0 at the left edge and 1 at the right edge of the scale
    '''

    v = np.asarray(pd.to_numeric(pd.Series(values), errors='coerce'), dtype=float)
    lo, hi = scale
    if log:
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(v > 0, np.log10(np.where(v > 0, v, 1)), np.nan)
        lo, hi = np.log10(lo), np.log10(hi)
    return (v - lo) / (hi - lo)


def _fmt(x:float) -> str:
    return f'{x:g}' if abs(x) >= .01 or x == 0 else f'{x:.0e}'


class LogLayout:

    r'''
    Figure of side-by-side log tracks sharing a depth axis, each with a header box above it

    Parameters
    ----------
    top, bottom : float
        Depth range shown

    widths : list of float
        Relative width of each track

    header_rows : int
        Number of header rows (one per curve or fill); defaults to 3

    title, subtitle : str
        Text at the top left

    figsize : tuple
        Figure size in inches; the width defaults to 1.35 in per unit of track width

    depth_label : str
        Label of the depth axis

    legend_rows : int
        Rows reserved below the tracks for a facies legend
    '''

    def __init__(self, top:float, bottom:float, widths:list, header_rows:int=3, title:str=None, subtitle:str=None,
                 figsize:tuple=None, depth_label:str='Depth', legend_rows:int=0):

        self.top, self.bottom = float(min(top, bottom)), float(max(top, bottom))
        widths = list(widths)
        if figsize is None:
            figsize = (max(4., 1.35 * sum(widths) + .9), 11.)
        W, H = figsize
        self.fig = plt.figure(figsize=figsize)
        left, right = .72 / W, 1 - .12 / W
        head_h = (header_rows * HEADER_ROW + .1) / H
        title_h = (TITLE_BAND if title or subtitle else .15) / H
        bottom_m = (.25 + .22 * legend_rows) / H
        track_top = 1 - title_h - head_h
        edges = left + (right - left) * np.concatenate([[0], np.cumsum(widths)]) / sum(widths)
        self.headers, self.tracks = [], []
        for i in range(len(widths)):
            x0, w = edges[i], edges[i + 1] - edges[i]
            hax = self.fig.add_axes([x0, track_top, w, head_h])
            ax = self.fig.add_axes([x0, bottom_m, w, track_top - bottom_m], sharey=self.tracks[0] if self.tracks else None)
            self._header_box(hax, header_rows)
            self._track_box(ax, first=i == 0)
            self.headers.append(hax)
            self.tracks.append(ax)
        self.rows = header_rows
        self._used = [0] * len(widths)
        self.track_curves = [[] for _ in widths]
        self.fig.petrolib_tracks = self.track_curves
        self.tracks[0].set_ylim(self.bottom, self.top)
        self.fig.text(.18 / W, bottom_m + (track_top - bottom_m) / 2, depth_label, rotation=90, ha='left', va='center',
                      color=st.INK2, fontsize=8.5)
        if title or subtitle:
            st.figure_title(self.fig, title, subtitle, x=left, y=1 - .14 / H)
        self._fill_rows = {}

    #---- boxes -------------------------------------------------------------------------------------
    def _header_box(self, hax, rows):
        hax.set_xlim(0, 1)
        hax.set_ylim(0, rows)
        hax.set_xticks([]); hax.set_yticks([])
        hax.grid(False)
        hax.set_facecolor('#fafaf8')
        for s in hax.spines.values():
            s.set_visible(True); s.set_color(st.AXIS); s.set_linewidth(.8)

    def _track_box(self, ax, first:bool):
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(st.AXIS); s.set_linewidth(.8)
        span = abs(self.bottom - self.top)
        major = st.nice_step(span, 8)
        ax.yaxis.set_major_locator(MultipleLocator(major))
        ax.yaxis.set_minor_locator(MultipleLocator(major / 5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.grid(True, axis='y', which='major', color=st.GRID, lw=.6)
        ax.grid(True, axis='y', which='minor', color=st.GRID_MINOR, lw=.4)
        ax.tick_params(axis='y', which='both', length=0 if not first else 2.5, labelleft=first, labelsize=8)
        ax.set_axisbelow(True)

    def _vgrid(self, ax, curve:Curve):
        if curve.log:
            lo, hi = sorted(np.log10(curve.scale))
            for d in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
                for m in range(1, 10):
                    x = normalize([m * 10. ** d], curve.scale, True)[0]
                    if 0 < x < 1:
                        ax.axvline(x, color=st.GRID if m == 1 else st.GRID_MINOR, lw=.6 if m == 1 else .4, zorder=0)
        else:
            for x in np.arange(.2, 1, .2):
                ax.axvline(x, color=st.GRID, lw=.6, zorder=0)

    #---- headers -----------------------------------------------------------------------------------
    def _row_y(self, row:int) -> float:
        return self.rows - row - .5

    def _next_row(self, i:int) -> int:
        row = self._used[i]
        self._used[i] += 1
        return row

    def header_curve(self, i:int, row:int, curve:Curve):
        hax, y = self.headers[i], self._row_y(row)
        hax.plot([.05, .95], [y - .2] * 2, color=curve.color, ls=curve.ls, lw=1.5, solid_capstyle='butt')
        label = curve.label + (f'  {curve.unit}' if curve.unit else '')
        hax.text(.5, y + .12, label, ha='center', va='center', fontsize=7.5, color=st.INK, clip_on=True)
        lo, hi = curve.scale
        hax.text(.04, y + .12, _fmt(lo), ha='left', va='center', fontsize=6.5, color=st.INK2)
        hax.text(.96, y + .12, _fmt(hi), ha='right', va='center', fontsize=6.5, color=st.INK2)

    def header_fill(self, i:int, row:int, label:str, color:str, alpha:float=1.):
        hax, y = self.headers[i], self._row_y(row)
        hax.add_patch(Rectangle((.05, y - .28), .14, .5, facecolor=color, alpha=alpha, lw=0))
        hax.text(.24, y - .03, label, ha='left', va='center', fontsize=7, color=st.INK2, clip_on=True)

    def header_title(self, i:int, text:str):
        hax = self.headers[i]
        hax.text(.5, self.rows / 2, text, ha='center', va='center', fontsize=8, color=st.INK, fontweight='semibold',
                 wrap=True)

    #---- tracks ------------------------------------------------------------------------------------
    def curves(self, i:int, df:pd.DataFrame, depth:str, curves:list, fill:str=None, fill_curve:int=0,
               baseline:float=None, cmap=None, threshold:float=None, crossover:str=None, grid:bool=True) -> list:

        r'''
        Draws curves in track i with their headers and an optional fill

        fill : 'sand' shades sand (yellow) and shale (grey) on either side of `baseline`; 'left' or 'right'
            shades from the curve to that edge with `cmap` coloured by the curve value; 'above' shades where
            the curve exceeds `threshold` (resistivity of hydrocarbon zones)
        crossover : 'gas', 'shale' or 'both' fills between the first two curves where they cross
            (neutron-density): 'gas' where the first plots right of the second
        '''

        ax = self.tracks[i]
        z = df[depth].to_numpy(dtype=float)
        curves = [Curve(c) if isinstance(c, str) else c for c in curves]
        curves = [c.resolved(df[c.name]) for c in curves]
        self.track_curves[i].extend(curves)
        if grid:
            self._vgrid(ax, curves[0])
        xs = [normalize(df[c.name], c.scale, c.log) for c in curves]
        first = self._used[i]
        self._used[i] += len(curves)
        row = 0
        if fill is not None:
            c, x = curves[fill_curve], xs[fill_curve]
            if fill == 'sand':
                b = normalize([baseline if baseline is not None else 75.], c.scale, c.log)[0]
                left_is_low = c.scale[0] < c.scale[1]
                sand, shale = (x <= b, x >= b) if left_is_low else (x >= b, x <= b)
                ax.fill_betweenx(z, b, x, where=sand, interpolate=True, color=st.SAND, lw=0, zorder=1)
                ax.fill_betweenx(z, b, x, where=shale, interpolate=True, color=st.SHALE, alpha=.75, lw=0, zorder=1)
                ax.axvline(b, color=st.MUTED, lw=.6, ls=(0, (3, 2)), zorder=2)
                self.header_fill(i, self._next_row(i), f'sand < {baseline if baseline is not None else 75:g}', st.SAND)
            elif fill in ('left', 'right'):
                self.gradient(ax, z, x, 0. if fill == 'left' else 1., df[c.name].to_numpy(dtype=float),
                              cmap or st.GR_RAMP, c.scale)
            elif fill == 'above':
                t = normalize([threshold], c.scale, c.log)[0]
                ax.fill_betweenx(z, t, x, where=x >= t, interpolate=True, color=st.HYDROCARBON, alpha=.45, lw=0,
                                 zorder=1)
                self.header_fill(i, self._next_row(i), f'> {threshold:g} {c.unit}', st.HYDROCARBON, .45)
        if crossover and len(xs) >= 2:
            a, b = xs[0], xs[1]
            if crossover in ('gas', 'both'):
                ax.fill_betweenx(z, a, b, where=a > b, interpolate=True, color=st.GAS, alpha=.85, lw=0, zorder=1)
                self.header_fill(i, self._next_row(i), 'crossover', st.GAS, .85)
            if crossover in ('shale', 'both'):
                ax.fill_betweenx(z, a, b, where=a < b, interpolate=True, color=st.SHALE, alpha=.45, lw=0, zorder=1)
                self.header_fill(i, self._next_row(i), 'separation', st.SHALE, .45)
        for r, (c, x) in enumerate(zip(curves, xs)):
            ax.plot(x, z, color=c.color, ls=c.ls, lw=c.lw, zorder=3)
            self.header_curve(i, first + r, c)
        return curves

    def shade(self, i:int, df:pd.DataFrame, depth:str, curve, to, color:str, label:str=None, alpha:float=1.,
              where=None):

        r'''
        Shades track i between a curve and another curve, a value on the curve's scale, or the 'left' or
        'right' edge, with a header row for `label`
        '''

        ax, z = self.tracks[i], df[depth].to_numpy(dtype=float)
        c = (Curve(curve) if isinstance(curve, str) else curve).resolved(df[curve if isinstance(curve, str) else curve.name])
        x = normalize(df[c.name], c.scale, c.log)
        if isinstance(to, str) and to in ('left', 'right'):
            x2 = np.full_like(x, 0. if to == 'left' else 1.)
        elif isinstance(to, (str, Curve)):
            c2 = (Curve(to) if isinstance(to, str) else to).resolved(df[to if isinstance(to, str) else to.name])
            x2 = normalize(df[c2.name], c2.scale, c2.log)
        else:
            x2 = np.full_like(x, normalize([to], c.scale, c.log)[0])
        ok = np.isfinite(x) & np.isfinite(x2)
        where = ok if where is None else ok & np.asarray(where, dtype=bool)
        ax.fill_betweenx(z, np.clip(x, 0, 1), np.clip(x2, 0, 1), where=where, interpolate=True, color=color,
                         alpha=alpha, lw=0, zorder=1)
        if label:
            self.header_fill(i, self._next_row(i), label, color, alpha)

    def gradient(self, ax, z, x, edge:float, values, cmap, scale:tuple):

        r'''
        Fills from the curve to an edge with colours from cmap by the curve value
        '''

        poly = ax.fill_betweenx(z, edge, x, facecolor='none', lw=0)
        ok = np.isfinite(values) & np.isfinite(z)
        if ok.sum() < 2:
            return
        grid = np.linspace(self.top, self.bottom, 2000)
        order = np.argsort(z[ok])
        v = np.interp(grid, z[ok][order], values[ok][order])
        cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        lo, hi = sorted(scale)
        im = ax.imshow(v[:, None], aspect='auto', cmap=cmap, vmin=lo, vmax=hi, origin='upper', zorder=1,
                       extent=[0, 1, self.bottom, self.top], interpolation='bilinear')
        paths = poly.get_paths()
        if paths:
            from matplotlib.path import Path
            im.set_clip_path(Path.make_compound_path(*paths), transform=ax.transData)
        ax.set_xlim(0, 1)
        ax.set_ylim(self.bottom, self.top)

    def flags(self, i:int, df:pd.DataFrame, depth:str, flags:list, labels:list=None, colors:list=None):

        r'''
        Net flags side by side in track i, one column per flag, shaded where the flag is 1
        '''

        ax, z = self.tracks[i], df[depth].to_numpy(dtype=float)
        n = len(flags)
        labels = labels or [str(f).replace('_NET_FLAG', '').replace('_FLAG', '') for f in flags]
        colors = colors or [st.FLAG_COLORS.get(l.upper(), st.SERIES[k]) for k, l in enumerate(labels)]
        for k, (f, label, color) in enumerate(zip(flags, labels, colors)):
            x0, x1 = k / n, (k + 1) / n
            on = pd.to_numeric(df[f], errors='coerce').fillna(0).to_numpy() > 0
            ax.fill_betweenx(z, x0, x1, where=on, step='mid', color=color, lw=0)
            if k:
                ax.axvline(x0, color=st.AXIS, lw=.6)
            self.headers[i].text((x0 + x1) / 2, self.rows / 2, label, ha='center', va='center', fontsize=7,
                                 color=st.INK, fontweight='semibold', rotation=90 if n > 4 else 0)
            self.headers[i].add_patch(Rectangle((x0 + .02, .12), (x1 - x0) - .04, .22, facecolor=color, lw=0))

    def facies(self, i:int, df:pd.DataFrame, depth:str, codes, colors:list, labels:list, name:str='Facies'):

        r'''
        Facies as coloured intervals in track i, with the classes in a legend under the tracks
        '''

        ax, z = self.tracks[i], df[depth].to_numpy(dtype=float)
        codes = pd.Series(codes, index=df.index)
        handles = []
        for code, color, label in zip(sorted(codes.dropna().unique(), key=str) if labels is None else range(len(labels)),
                                      colors, labels):
            on = (codes == code).to_numpy()
            ax.fill_betweenx(z, 0, 1, where=on, step='mid', color=color, lw=0)
            handles.append(Rectangle((0, 0), 1, 1, facecolor=color, lw=0, label=label))
        self.header_title(i, name)
        self.fig.legend(handles=handles, loc='lower left', ncol=min(len(handles), 6), frameon=False, fontsize=7.5,
                        bbox_to_anchor=(self.tracks[0].get_position().x0, 0), handlelength=1.2, columnspacing=1.2)

    def zones(self, i:int, tops:list, bases:list, names:list, lines:bool=True):

        r'''
        Zone track i with tinted intervals and names, and zone boundaries across every track
        '''

        ax = self.tracks[i]
        fig_w = ax.get_position().width * self.fig.get_figwidth()
        chars = max(int(fig_w * 72 / 4.6), 6)
        for k, (t, b, name) in enumerate(zip(tops, bases, names)):
            ax.axhspan(t, b, color=st.SERIES[k % len(st.SERIES)], alpha=.13, lw=0)
            mid = (t + b) / 2
            if self.top <= mid <= self.bottom:
                ax.text(.5, mid, textwrap.fill(str(name), chars), ha='center', va='center', fontsize=7.5,
                        color=st.INK, clip_on=True)
        if lines:
            for a in self.tracks:
                for d in sorted(set(tops) | set(bases)):
                    a.axhline(d, color=st.INK2, lw=.7, alpha=.8, zorder=4)
        self.header_title(i, 'Zones')
        ax.set_ylim(self.bottom, self.top)

    def finish(self) -> plt.Figure:
        for ax in self.tracks:
            ax.set_xlim(0, 1)
        self.tracks[0].set_ylim(self.bottom, self.top)
        return self.fig
