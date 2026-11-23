'''
The petrolib look shared by every plot: colours, fonts, the plot theme and small drawing helpers

Curves keep the colours of industry log displays (gamma ray green, neutron blue, density red); other
series take a colour-blind-safe categorical order, magnitudes a one-hue blue ramp and signed values a
blue-grey-red ramp. Every plot function draws inside `theme()`, so the global matplotlib settings of
the user are left alone; `set_theme(False)` turns the theme off.

Example
-------
>>> from petrolib.plots import style
>>> style.set_theme(False)          # plot with your own matplotlib style
'''
from __future__ import annotations
import functools
from contextlib import contextmanager, nullcontext

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

#ink and chrome
INK = '#1d1d1b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e4e3dc'
GRID_MINOR = '#f0efea'
AXIS = '#b9b8ae'
SURFACE = '#ffffff'

#categorical order, validated for colour-blind separation of neighbours
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

#log curves by curve type
CURVE_COLORS = {
    'GR': '#008300', 'SP': '#1baf7a', 'CALI': '#6b6a64', 'BS': MUTED,
    'RT': INK, 'RM': '#eb6834', 'RXO': '#2a78d6',
    'NPHI': '#2a78d6', 'RHOB': '#e34948', 'DRHO': '#e87ba4', 'PEF': '#e87ba4',
    'DT': '#6250d6', 'DTS': '#e87ba4',
}

#fills and flags
SAND = '#f2cf6b'
SHALE = '#b4aa98'
GAS = '#f6d55c'
HYDROCARBON = '#e5736f'
WATER = '#79aee8'
FLAG_COLORS = {'ROCK': '#e3a600', 'RES': '#21a179', 'PAY': '#a8322d'}   # colour-blind safe as a set
LITHOLOGY_COLORS = {'Sandstone': SAND, 'Sandstone/Shale': '#d9c48c', 'Shale': SHALE, 'Marl': '#a9c98a',
                    'Dolomite': '#a59bd8', 'Limestone': '#8ec3e6', 'Chalk': '#cfe4f2', 'Halite': '#e4d9f0',
                    'Anhydrite': '#e7a3c4', 'Tuff': '#e9a06b', 'Coal': '#2b2b29', 'Basement': '#b05a7a'}

#magnitude, signed values, and gamma ray from sand to shale
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'petrolib', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list(
    'petrolib_diverging', ['#104281', '#3987e5', '#b7d3f6', '#f0efec', '#f4b3ac', '#e34948', '#8f1f1d'])
GR_RAMP = LinearSegmentedColormap.from_list('petrolib_gr', ['#f7e3a3', '#e9c25c', '#c7a36a', '#9c9282', '#6f675c'])
for _cmap in (SEQUENTIAL, DIVERGING, GR_RAMP):
    if _cmap.name not in mpl.colormaps:
        mpl.colormaps.register(_cmap)

FONTS = ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans']

RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': FONTS,
    'font.size': 9,
    'text.color': INK,
    'axes.titlesize': 10.5,
    'axes.titleweight': 'semibold',
    'axes.titlelocation': 'left',
    'axes.titlepad': 9,
    'axes.titlecolor': INK,
    'axes.labelsize': 9,
    'axes.labelcolor': INK2,
    'axes.labelpad': 5,
    'axes.edgecolor': AXIS,
    'axes.linewidth': .8,
    'axes.facecolor': SURFACE,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.prop_cycle': mpl.cycler(color=SERIES),
    'grid.color': GRID,
    'grid.linewidth': .6,
    'grid.linestyle': '-',
    'xtick.color': AXIS,
    'ytick.color': AXIS,
    'xtick.labelcolor': INK2,
    'ytick.labelcolor': INK2,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.width': .8,
    'ytick.major.width': .8,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'legend.title_fontsize': 8,
    'legend.handlelength': 1.6,
    'legend.borderaxespad': .6,
    'figure.facecolor': SURFACE,
    'figure.titlesize': 12,
    'figure.titleweight': 'semibold',
    'lines.linewidth': 1.2,
    'lines.solid_capstyle': 'round',
    'scatter.edgecolors': 'none',
    'image.cmap': 'petrolib',
    'patch.linewidth': 0,
    'savefig.facecolor': SURFACE,
}

_ENABLED = True


def set_theme(enabled:bool=True):

    r'''
    Turns the petrolib plot theme on (the default) or off, to draw with your own matplotlib style
    '''

    global _ENABLED
    _ENABLED = bool(enabled)


@contextmanager
def theme():

    r'''
    Context in which the petrolib theme is applied, without changing the global matplotlib settings
    '''

    if not _ENABLED:
        yield
        return
    import logging
    level = logging.getLogger('matplotlib.font_manager').level
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    try:
        with mpl.rc_context(RC):
            yield
    finally:
        logging.getLogger('matplotlib.font_manager').setLevel(level)


def themed(func):

    r'''
    Decorator that draws a plot function inside `theme()`
    '''

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with theme():
            return func(*args, **kwargs)
    return wrapper


def curve_color(name, kind:str=None) -> str:

    r'''
    Colour of a log curve from its curve type, e.g. green for any gamma ray mnemonic
    '''

    from ..mnemonics import log_kind
    return CURVE_COLORS.get(kind or log_kind(name) or '', INK)


def categorical(n:int) -> list:

    r'''
    The first n colours of the categorical order; past eight, colours come from a qualitative map
    '''

    if n <= len(SERIES):
        return SERIES[:n]
    return [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20')(np.linspace(0, 1, n))]


def ordinal(n:int) -> list:

    r'''
    n colours of the blue ramp for ordered classes (rock types, tiers), light to dark
    '''

    return [mpl.colors.to_hex(SEQUENTIAL(x)) for x in np.linspace(.25, 1, n)] if n > 1 else [SERIES[0]]


def title(ax, text:str=None, subtitle:str=None):

    r'''
    Left-aligned title with an optional subtitle in secondary ink
    '''

    if text:
        ax.set_title(text, loc='left', pad=22 if subtitle else 9)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5, color=INK2)


def figure_title(fig, text:str=None, subtitle:str=None, x:float=None, y:float=None):

    r'''
    Title at the top left of a figure, aligned with its first axes
    '''

    x = fig.axes[0].get_position().x0 if x is None and fig.axes else (x or .02)
    y = .985 if y is None else y
    if text:
        fig.text(x, y, text, ha='left', va='top', fontsize=12, fontweight='semibold', color=INK)
    if subtitle:
        fig.text(x, y - .3 / fig.get_figheight(), subtitle, ha='left', va='top', fontsize=8.5, color=INK2)


def frame(ax, grid:str='both'):

    r'''
    Hairline left and bottom axes with a recessive grid; grid is 'both', 'x', 'y' or None
    '''

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(AXIS)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=.6)
    ax.set_axisbelow(True)


def colorbar(mappable, ax, label:str=None, **kwargs):

    r'''
    Slim colour bar without an outline
    '''

    cb = ax.figure.colorbar(mappable, ax=ax, fraction=kwargs.pop('fraction', .045), pad=kwargs.pop('pad', .025),
                            aspect=kwargs.pop('aspect', 30), **kwargs)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7.5, labelcolor=INK2)
    if label:
        cb.set_label(label, color=INK2, fontsize=8.5)
    return cb


def line_label(ax, x, y, text:str, color:str=MUTED, ha:str='left', va:str='center', **kwargs):

    r'''
    Direct label for a reference line, in muted ink with a white halo so it stays legible over points
    '''

    import matplotlib.patheffects as pe
    return ax.text(x, y, text, color=color, ha=ha, va=va, fontsize=kwargs.pop('fontsize', 7.5),
                   path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)], clip_on=True, **kwargs)


def legend(ax, *args, **kwargs):

    r'''
    Frameless legend in secondary ink
    '''

    leg = ax.legend(*args, **kwargs)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


def scatter_size(n:int) -> float:

    r'''
    Marker area that keeps dense cross plots readable: larger for a few points, smaller for many
    '''

    return float(np.clip(900 / max(n, 1) ** .5, 4, 28))


def nice_step(span:float, target:int=8) -> float:

    r'''
    A 1-2-2.5-5 step that splits span into about `target` intervals
    '''

    raw = span / max(target, 1)
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def figure_context(ax):

    r'''
    `theme()` when a new figure is drawn, and a no-op when the caller passes its own axes
    '''

    return theme() if ax is None else nullcontext()
