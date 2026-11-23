'''
Petrophysics plots: Buckles, Winland R35, RQI/FZI, Lorenz, M-N, MID and Thomas-Stieber cross plots,
cutoff sensitivity, mineral volumes and saturation uncertainty

Each function draws on `ax` (a new figure if None) and returns the Axes. Reference lines (constant
BVW, R35, FZI, SW or MHI) are thin grey lines labelled where they leave the plot
'''
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import LogFormatterSciNotation, FuncFormatter

from ..petrophysics.rocktype import rqi, normalized_porosity, winland_r35, lorenz, PORE_TYPES
from ..petrophysics.lithology import minerals as _minerals, m_n, rhomaa, umaa
from ..petrophysics.netpay import sample_thickness
from . import style as st
from .style import themed

_MINERAL_COLORS = {'Quartz': '#e3a600', 'Calcite': '#2a78d6', 'Dolomite': '#6250d6', 'Anhydrite': '#e87ba4',
                   'Halite': '#1baf7a', 'Gypsum': '#008300'}
_VOLUME_COLORS = {**_MINERAL_COLORS, 'Quartz': st.SAND, 'Shale': st.SHALE, 'Clay': st.SHALE, 'Illite': st.SHALE,
                  'Water': st.WATER, 'Oil': '#3f9e5a', 'Gas': st.HYDROCARBON}


def _axes(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _is_ordinal(values:pd.Series) -> bool:
    #a few whole numbers, such as discrete rock types, get a stepped colour bar
    v = pd.to_numeric(values, errors='coerce').dropna()
    return 1 < v.nunique() <= 12 and bool(np.all(np.isclose(v, np.round(v))))


def _scatter(ax, df:pd.DataFrame, x, y, hue:str=None, cmap:str=None, cbar:dict=None, **kwargs):

    '''
    Scatter of x against y, colored by a numeric or categorical `hue` column
    '''

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = int(np.sum(np.isfinite(x) & np.isfinite(y)))
    kwargs = {'s': st.scatter_size(n), 'linewidths': 0, **kwargs}
    if hue is None:
        return ax.scatter(x, y, color=st.SERIES[0], alpha=.55, **kwargs)
    values = df[hue]
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        values = values.to_numpy(dtype=float)
        if _is_ordinal(pd.Series(values)):
            levels = np.arange(np.nanmin(values), np.nanmax(values) + 1)
            colors = st.ordinal(len(levels)) if cmap is None else plt.get_cmap(cmap)(np.linspace(0, 1, len(levels)))
            norm = BoundaryNorm(np.append(levels, levels[-1] + 1) - .5, len(levels))
            sc = ax.scatter(x, y, c=values, cmap=ListedColormap(colors), norm=norm, alpha=.85, **kwargs)
            cb = st.colorbar(sc, ax, hue, ticks=levels if len(levels) <= 12 else None, **(cbar or {}))
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        else:
            sc = ax.scatter(x, y, c=values, cmap=cmap or st.SEQUENTIAL, alpha=.8, **kwargs)
            st.colorbar(sc, ax, hue, **(cbar or {}))
        return sc
    groups = pd.unique(values.dropna())
    for value, color in zip(groups, st.categorical(len(groups))):
        on = (values == value).to_numpy()
        ax.scatter(x[on], y[on], color=color, alpha=.75, label=str(value), **kwargs)
    st.legend(ax, title=hue, loc='best', markerscale=1.6)


def _style(ax, title:str, xlabel:str, ylabel:str, subtitle:str=None):
    st.title(ax, title, subtitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.frame(ax)


def _ref(ax, x, y, **kwargs):
    #reference line: thin, grey, under the points
    return ax.plot(x, y, color=kwargs.pop('color', st.MUTED), lw=kwargs.pop('lw', .8), zorder=kwargs.pop('zorder', 1.5),
                   **kwargs)


def _n(df:pd.DataFrame, *cols) -> str:
    n = int(df[list(cols)].notna().all(axis=1).sum())
    return f'{n:,} samples'


@themed
def bucklesPlot(df:pd.DataFrame, phi:str='PHIE', sw:str='SW', bvw:tuple=(.02, .04, .06, .08, .1),
                hue:str=None, cmap:str=None, title:str='Buckles plot', figsize:tuple=(7, 5.5), ax=None):

    r'''
    Buckles plot: water saturation against porosity, with lines of constant bulk volume water
    (PHI SW). Points along one line are at irreducible saturation; their BVW is the Buckles number
    to use in `petrolib.petrophysics.swirr_buckles`

    Example
    -------
    >>> from petrolib.plots import bucklesPlot
    >>> bucklesPlot(df[df['PAY_NET_FLAG'] == 1], hue='VShale')
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, df[phi], df[sw], hue, cmap)
    xmax = max(.3, float(np.nanpercentile(df[phi], 99.5)) * 1.1)
    p = np.linspace(.004, xmax, 300)
    for b in bvw:
        _ref(ax, p, b / p)
        x = xmax if b / xmax <= .97 else b / .97
        st.line_label(ax, x - .01 * xmax, min(b / x, .97), f'BVW {b:g}', ha='right', va='bottom')
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1)
    _style(ax, title, f'{phi} (v/v)', f'{sw} (v/v)', _n(df, phi, sw))
    return ax


def _decades(values, lo:float, hi:float) -> tuple:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return lo, hi
    return (max(lo, 10 ** (np.floor(np.log10(np.percentile(v, .5))) - 1)),
            min(hi, 10 ** (np.ceil(np.log10(np.percentile(v, 99.5))) + 1)))


@themed
def winlandPlot(df:pd.DataFrame, phi:str='PHIE', k:str='Perm', r35:tuple=(.1, .5, 2., 10.), hue:str=None,
                cmap:str=None, title:str='Winland R35 plot', figsize:tuple=(7, 5.5), ax=None):

    r'''
    Porosity-permeability cross plot with Winland R35 pore throat radius lines (microns), which
    separate the nanoport, microport, mesoport, macroport and megaport classes

    Example
    -------
    >>> from petrolib.plots import winlandPlot
    >>> winlandPlot(core, phi='CPOR', k='CKH')
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, df[phi], df[k], hue, cmap)
    xmax = max(.3, float(np.nanpercentile(df[phi], 99.5)) * 1.15)
    ylo, yhi = _decades(df[k], 1e-3, 1e5)
    p = np.linspace(.005, xmax, 300)
    for r in r35:
        kline = 10**((np.log10(r) - .732 + .864 * np.log10(100 * p)) / .588)
        _ref(ax, p, kline)
        inside = np.flatnonzero(kline <= yhi / 1.5)
        if inside.size:
            j = inside[-1]
            st.line_label(ax, p[j] - .01 * xmax, kline[j], f'R35 {r:g} µm', ha='right', va='bottom')
    ax.set_yscale('log'); ax.set_xlim(0, xmax); ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    _style(ax, title, f'{phi} (v/v)', f'{k} (mD)', _n(df, phi, k))
    return ax


@themed
def rqiPlot(df:pd.DataFrame, phi:str='PHIE', k:str='Perm', fzi_lines:tuple=(.1, .3, 1., 3., 10.), hue:str=None,
            cmap:str=None, title:str='RQI against normalised porosity', figsize:tuple=(7, 5.5), ax=None):

    r'''
    Rock quality index against normalized porosity, on log scales. Each hydraulic flow unit plots
    along a unit-slope line of constant flow zone indicator (FZI = RQI / PHIZ)

    Example
    -------
    >>> from petrolib.plots import rqiPlot
    >>> rqiPlot(df, hue='DRT')
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, normalized_porosity(df[phi]), rqi(df[k], df[phi]), hue, cmap)
    z = np.logspace(-2, 0, 50)
    for f in fzi_lines:
        _ref(ax, z, f * z)
        zl = min(1., 100 / f) / 1.15
        st.line_label(ax, zl, f * zl, f'FZI {f:g}', ha='right', va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.01, 1); ax.set_ylim(.001, 100)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    _style(ax, title, 'PHIZ = PHIE / (1 − PHIE)', 'RQI (µm)', _n(df, phi, k))
    return ax


@themed
def lorenzPlot(df:pd.DataFrame, phi:str='PHIE', k:str='Perm', depth:str=None, stratigraphic:bool=True,
               title:str=None, figsize:tuple=(5.5, 5.5), ax=None):

    r'''
    (Stratigraphic modified) Lorenz plot: cumulative flow capacity against cumulative storage capacity.
    Steep segments are speed zones, flat ones baffles. The Lorenz coefficient is written on the plot

    Parameters
    ----------
    depth : str default None
        Depth column. When given, samples are weighted by their thickness and ordered by depth

    stratigraphic : bool default True
        Keep depth order (SMLP). If False, sort by decreasing k/PHI (classic Lorenz plot)

    Example
    -------
    >>> from petrolib.plots import lorenzPlot
    >>> lorenzPlot(zone_df, depth='DEPTH')
    '''

    ax = _axes(ax, figsize)
    data = df.sort_values(depth) if depth is not None else df
    h = sample_thickness(data[depth], data[depth].min(), data[depth].max()) if depth is not None else None
    curve, lc = lorenz(data[k], data[phi], h, stratigraphic=stratigraphic)
    ax.fill_between(curve['STORAGE'], curve['STORAGE'], curve['FLOW'], color=st.SERIES[0], alpha=.10, lw=0)
    _ref(ax, [0, 1], [0, 1])
    ax.plot(curve['STORAGE'], curve['FLOW'], color=st.SERIES[0], lw=1.8, zorder=3)
    st.line_label(ax, .62, .58, 'homogeneous', ha='left', va='top', rotation=45, rotation_mode='anchor',
                  transform_rotates_text=True)
    ax.text(.97, .05, f'Lorenz coefficient {lc:.2f}', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=st.INK, fontweight='semibold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    name = 'Stratigraphic modified Lorenz plot' if stratigraphic else 'Lorenz plot'
    _style(ax, title or name, 'Cumulative storage capacity (φh)', 'Cumulative flow capacity (kh)',
           'depth order' if stratigraphic else 'sorted by k/φ')
    return ax


def _mineral_points(ax, x, y, names):
    import matplotlib.patheffects as pe
    for name, xi, yi in zip(names, x, y):
        ax.scatter(xi, yi, s=70, marker='D', color=_MINERAL_COLORS.get(name, st.INK), edgecolor=st.SURFACE,
                   linewidths=1.2, zorder=6)
        ax.annotate(name, (xi, yi), xytext=(7, 4), textcoords='offset points', fontsize=8, fontweight='semibold',
                    color=st.INK, zorder=7, path_effects=[pe.withStroke(linewidth=3, foreground=st.SURFACE)])
    triangle = [names.index(m) for m in ('Quartz', 'Calcite', 'Dolomite', 'Quartz') if m in names]
    if len(triangle) == 4:
        ax.plot(np.asarray(x)[triangle], np.asarray(y)[triangle], '-', color=st.INK2, lw=.9, zorder=5)


def _limits(data, points, pad:float=.08, q:float=.5) -> tuple:
    #range of the data without the extreme 0.5%, widened to show every mineral point
    v = np.asarray(data, dtype=float)
    p = np.asarray(points, dtype=float)
    v, p = v[np.isfinite(v)], p[np.isfinite(p)]
    lo, hi = np.percentile(v, [q, 100 - q]) if v.size else (p.min(), p.max())
    lo, hi = min(lo, p.min()), max(hi, p.max())
    span = hi - lo or 1.
    return lo - pad * span, hi + pad * span


@themed
def mnPlot(df:pd.DataFrame, rhob:str='RHOB', nphi:str='NPHI', dt:str='DT', hue:str=None, cmap:str=None,
           mineral_names:tuple=('Quartz', 'Calcite', 'Dolomite', 'Anhydrite', 'Gypsum'), rho_fluid:float=1.,
           dt_fluid:float=189., title:str='M–N plot', figsize:tuple=(7, 5.5), ax=None):

    r'''
    M-N lithology plot, with mineral points computed from `petrolib.petrophysics.minerals()`.
    Shale plots below the quartz-calcite-dolomite triangle and gas above it

    Example
    -------
    >>> from petrolib.plots import mnPlot
    >>> mnPlot(df, hue='GR')
    '''

    ax = _axes(ax, figsize)
    M, N = m_n(df[rhob], df[nphi], df[dt], rho_fluid=rho_fluid, dt_fluid=dt_fluid)
    _scatter(ax, df, N, M, hue, cmap)
    mm = _minerals().loc[list(mineral_names)]
    Mm, Nm = m_n(mm['RHOB'], mm['NPHI'], mm['DT'], rho_fluid=rho_fluid, dt_fluid=dt_fluid)
    _mineral_points(ax, Nm, Mm, list(mineral_names))
    ax.set_xlim(*_limits(N, Nm)); ax.set_ylim(*_limits(M, Mm))
    _style(ax, title, 'N', 'M', _n(df, rhob, nphi, dt))
    return ax


@themed
def midPlot(df:pd.DataFrame, rhob:str='RHOB', pef:str='PEF', phit:str='PHIT', hue:str=None, cmap:str=None,
            mineral_names:tuple=('Quartz', 'Calcite', 'Dolomite', 'Anhydrite', 'Halite'), rho_fluid:float=1.,
            u_fluid:float=.398, title:str='Matrix identification (MID) plot', figsize:tuple=(7, 5.5), ax=None):

    r'''
    Matrix identification plot: apparent matrix density (RHOMAA) against apparent matrix
    volumetric photoelectric factor (UMAA), with mineral points

    Example
    -------
    >>> from petrolib.plots import midPlot
    >>> midPlot(df, phit='PHIT', hue='GR')
    '''

    ax = _axes(ax, figsize)
    u, r = umaa(df[pef], df[rhob], df[phit], u_fluid), rhomaa(df[rhob], df[phit], rho_fluid)
    _scatter(ax, df, u, r, hue, cmap)
    mm = _minerals().loc[list(mineral_names)]
    um = umaa(mm['PEF'], mm['RHOB'], 0.)
    _mineral_points(ax, um, mm['RHOB'], list(mineral_names))
    lo, hi = _limits(r, mm['RHOB'])
    ax.set_xlim(*_limits(u, um)); ax.set_ylim(hi, lo)
    _style(ax, title, 'UMAA (barns/cm³)', 'RHOMAA (g/cm³)', _n(df, rhob, pef, phit))
    return ax


@themed
def thomasStieberPlot(df:pd.DataFrame, phi_sand:float, phi_shale:float, vsh:str='VShale', phit:str='PHIT',
                      hue:str=None, cmap:str=None, title:str='Thomas–Stieber plot', figsize:tuple=(7, 5.5), ax=None):

    r'''
    Thomas-Stieber (1975) plot of total porosity against volume of shale, with the laminar,
    dispersed and structural shale lines

    Parameters
    ----------
    phi_sand : float
        Total porosity of the clean sand

    phi_shale : float
        Total porosity of the shale

    Example
    -------
    >>> from petrolib.plots import thomasStieberPlot
    >>> thomasStieberPlot(df, phi_sand=.25, phi_shale=.12)
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, df[vsh], df[phit], hue, cmap)
    clean, shale = (0, phi_sand), (1, phi_shale)
    dispersed = (phi_sand, phi_sand * phi_shale)
    structural = (1 - phi_sand, phi_sand + (1 - phi_sand) * phi_shale)
    ymax = max(phi_sand, phi_shale, float(np.nanpercentile(df[phit], 99.5))) * 1.15
    lines = [(clean, shale, 'laminar', '-'), (clean, dispersed, 'dispersed', (0, (5, 2))),
             (dispersed, shale, None, (0, (5, 2))), (clean, structural, 'structural', (0, (1, 1.5)))]
    for a, b, label, ls in lines:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=st.INK2, lw=1.1, ls=ls, zorder=4)
        if label:
            t = .55 if label != 'structural' else .5
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            if y > ymax:
                t = (ymax * .93 - a[1]) / (b[1] - a[1])
                x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            st.line_label(ax, x, y, label, color=st.INK2, ha='center', va='bottom', fontsize=8)
    for (x, y), label, ha in [(clean, 'clean sand', 'left'), (shale, 'shale', 'right')]:
        ax.scatter(*[[v] for v in (x, y)], s=40, color=st.INK, zorder=6, clip_on=False)
        st.line_label(ax, x + (.015 if ha == 'left' else -.015), y + .012 * ymax, label, color=st.INK, ha=ha,
                      va='bottom', fontsize=8, fontweight='semibold')
    ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
    _style(ax, title, f'{vsh} (v/v)', f'{phit} (v/v)', _n(df, vsh, phit))
    return ax


def _split(ax, ratio:tuple=(1.25, 1)):
    #two stacked axes in the space of one, sharing x
    fig = ax.figure
    spec = ax.get_subplotspec()
    if spec is not None:
        gs = spec.subgridspec(2, 1, hspace=.12, height_ratios=ratio)
        ax.remove()
        top = fig.add_subplot(gs[0])
        return top, fig.add_subplot(gs[1], sharex=top)
    box = ax.get_position()
    ax.remove()
    h = box.height * .44
    top = fig.add_axes([box.x0, box.y1 - h, box.width, h])
    return top, fig.add_axes([box.x0, box.y0, box.width, h * .82], sharex=top)


@themed
def cutoffPlot(sensitivity:pd.DataFrame, right:str='Average Water Saturation', title:str=None,
               figsize:tuple=(7, 5.5), ax=None):

    r'''
    Net pay against cutoff value, from `petrolib.petrophysics.cutoff_sensitivity`, with a second
    summation column (default average water saturation) in a panel below on the same cutoff axis

    Returns
    -------
    The two Axes (net pay, second column)

    Example
    -------
    >>> sens = cutoff_sensitivity(zone_df, cutoff='por', values=np.arange(0, .2, .01))
    >>> cutoffPlot(sens, right='HCPV')
    '''

    if ax is None:
        fig, (top, bottom) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': (1.25, 1),
                                                                                          'hspace': .12})
    else:
        top, bottom = _split(ax)
    cutoff = sensitivity['cutoff'].iloc[0]
    x = sensitivity['value']
    for a, col, color in ((top, 'Net', st.SERIES[0]), (bottom, right, st.SERIES[1])):
        a.plot(x, sensitivity[col], color=color, lw=1.8, zorder=3)
        a.scatter(x, sensitivity[col], s=22, color=color, edgecolor=st.SURFACE, linewidths=1, zorder=4)
        st.frame(a)
    top.set_ylabel('Net pay')
    bottom.set_ylabel(right.replace('Average ', 'Avg ') if len(right) > 18 else right)
    bottom.set_xlabel(f'{cutoff.upper()} cutoff')
    top.tick_params(labelbottom=False)
    st.title(top, title or f'Net pay against the {cutoff.upper()} cutoff')
    return top, bottom


@themed
def volumePlot(volumes:pd.DataFrame, depth, components:list=None, colors:dict=None, title:str='Volumes',
               figsize:tuple=(3.2, 11), ax=None):

    r'''
    Stacked volume track, e.g. of `petrolib.petrophysics.multimineral` output

    Parameters
    ----------
    volumes : pd.DataFrame
        One column of volume fractions per component

    depth : array-like
        Depth of each row

    components : list default None
        Columns to stack, in order. Default: all except MM_ERROR

    Example
    -------
    >>> vol = multimineral(df, comps)
    >>> volumePlot(vol, df['DEPTH'])
    '''

    ax = _axes(ax, figsize)
    components = components or [c for c in volumes.columns if c != 'MM_ERROR']
    colors = {**_VOLUME_COLORS, **(colors or {})}
    extra = iter(st.SERIES)
    d = np.asarray(depth, dtype=float)
    left = np.zeros(len(volumes))
    for c in components:
        right = left + np.nan_to_num(volumes[c].to_numpy(dtype=float))
        ax.fill_betweenx(d, left, right, color=colors.get(c) or next(extra), label=c, lw=0)
        left = right
    ax.set_xlim(0, 1); ax.set_ylim(np.nanmax(d), np.nanmin(d))
    _track(ax, title, 'Volume fraction')
    st.legend(ax, loc='upper center', bbox_to_anchor=(.5, -.04), ncol=min(len(components), 3), handlelength=1,
              columnspacing=1)
    return ax


def _track(ax, title, xlabel):
    #a depth track drawn on a plain axes: boxed, scale on top
    st.title(ax, title)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(st.AXIS)
    ax.grid(True, color=st.GRID, lw=.6)
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth')
    ax.set_axisbelow(False)
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_zorder(2.5); line.set_alpha(.7)


@themed
def swUncertaintyPlot(mc:pd.DataFrame, depth, sw=None, title:str='SW uncertainty', figsize:tuple=(3.2, 11), ax=None):

    r'''
    Track of the P10-P90 range and P50 of water saturation from `petrolib.petrophysics.monte_carlo_archie`,
    optionally with a deterministic SW curve

    Example
    -------
    >>> mc = monte_carlo_archie(df['PHIE'], df['RT'], rw=(.02, .03, .045), seed=0)
    >>> swUncertaintyPlot(mc, df['DEPTH'], sw=df['SW'])
    '''

    ax = _axes(ax, figsize)
    d = np.asarray(depth, dtype=float)
    ax.fill_betweenx(d, mc['SW_P10'], mc['SW_P90'], color='#b7d3f6', lw=0, label='P10–P90')
    ax.plot(mc['SW_P50'], d, color=st.SERIES[0], lw=.9, label='P50')
    if sw is not None:
        ax.plot(np.asarray(sw, dtype=float), d, color=st.INK, lw=.7, label='deterministic')
    ax.set_xlim(1, 0); ax.set_ylim(np.nanmax(d), np.nanmin(d))
    _track(ax, title, 'SW (v/v)')
    st.legend(ax, loc='upper center', bbox_to_anchor=(.5, -.04), ncol=3, handlelength=1.2, columnspacing=1)
    return ax


@themed
def hinglePlot(df:pd.DataFrame, phi:str='PHIE', rt:str='RT', rw:float=.03, a:float=1., m:float=2., n:float=2.,
               sw_lines:tuple=(1., .8, .6, .4, .2), hue:str=None, cmap:str=None, title:str='Hingle plot',
               figsize:tuple=(7, 5.5), ax=None):

    r'''
    Hingle plot: porosity against RT^(-1/m), so Archie lines of constant SW are straight lines
    through the matrix point. Water-bearing points fall on the SW = 1 line, whose slope gives Rw.
    The right axis is labelled in RT

    Example
    -------
    >>> hinglePlot(zone_df, rw=.03, m=2, hue='VShale')
    '''

    ax = _axes(ax, figsize)
    y = np.asarray(df[rt], dtype=float)**(-1 / m)
    _scatter(ax, df, df[phi], y, hue, cmap, cbar={'pad': .13})
    ymax = float(np.nanpercentile(y, 99.5)) * 1.1
    xmax = max(.3, float(np.nanpercentile(df[phi], 99.5)) * 1.15)
    p = np.linspace(0, xmax, 50)
    for s in sw_lines:
        slope = s**(n / m) / (a * rw)**(1 / m)
        _ref(ax, p, slope * p, color=st.INK2 if s == 1 else st.MUTED, lw=1.1 if s == 1 else .8)
        x_end = min(p[-1], .97 * ymax / slope)                    #label where the line leaves the plot
        st.line_label(ax, x_end, slope * x_end, f'SW {s:g}', ha='right', va='bottom')
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    rt_ticks, last = [], None
    for r in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000):
        if r**(-1 / m) <= ymax and (last is None or last - r**(-1 / m) >= .05 * ymax):
            rt_ticks.append(r)
            last = r**(-1 / m)
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks([r**(-1 / m) for r in rt_ticks])
    right.set_yticklabels([str(r) for r in rt_ticks])
    right.set_ylabel(f'{rt} (ohm.m)')
    right.grid(False)
    right.spines[['top', 'left', 'bottom']].set_visible(False)
    right.spines['right'].set_visible(True); right.spines['right'].set_color(st.AXIS)
    _style(ax, title, f'{phi} (v/v)', f'{rt}$^{{-1/{m:g}}}$', f'Rw {rw:g} ohm.m, m {m:g}, n {n:g}')
    return ax


@themed
def movableHydrocarbonPlot(df:pd.DataFrame, sw:str='SW', sxo:str='SXO', mhi_lines:tuple=(.6, .7, 1.), hue:str=None,
                           cmap:str=None, title:str='Movable hydrocarbons', figsize:tuple=(6.5, 5.5), ax=None):

    r'''
    SW against SXO with lines of constant movable hydrocarbon index (MHI = SW / SXO). Points below
    the 0.7 line (0.6 in carbonates) hold hydrocarbons that moved during invasion

    Example
    -------
    >>> df['SXO'] = flushed_zone_saturation(df['PHIE'], df['RXO'], rmf=.08)
    >>> movableHydrocarbonPlot(df, hue='PHIE')
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, df[sxo], df[sw], hue, cmap)
    x = np.linspace(0, 1, 20)
    for mhi in mhi_lines:
        _ref(ax, x, np.minimum(mhi * x, 1), color=st.INK2 if mhi == 1 else st.MUTED)
        x_label = min(.985, .93 / mhi)                            #inside the top of the plot
        st.line_label(ax, x_label, mhi * x_label, f'MHI {mhi:g}', ha='right', va='bottom')
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_aspect('equal')
    _style(ax, title, f'{sxo} (v/v)', f'{sw} (v/v)', _n(df, sw, sxo))
    return ax


@themed
def histogramPlot(df:pd.DataFrame, curve:str, by:str=None, cutoff:float=None, bins:int=40, log:bool=False,
                  title:str=None, figsize:tuple=(7, 4.5), ax=None):

    r'''
    Histogram of a curve, one outline per group (e.g. per zone), with an optional cutoff line

    Example
    -------
    >>> histogramPlot(logs, 'PHIE', by='Zone', cutoff=.06)
    '''

    ax = _axes(ax, figsize)
    values = df[curve].astype(float)
    if log:
        values = np.log10(values.where(values > 0))
    edges = np.histogram_bin_edges(values.dropna(), bins=bins)
    groups = [(None, values)] if by is None else [(g, values[df[by] == g]) for g in pd.unique(df[by].dropna())]
    colors = st.categorical(len(groups))
    for (name, v), color in zip(groups, colors):
        label = None if name is None else str(name)
        if len(groups) == 1:
            ax.hist(v.dropna(), bins=edges, color=color, alpha=.85, rwidth=.9, label=label)
        else:
            ax.hist(v.dropna(), bins=edges, histtype='step', lw=1.6, color=color, label=label)
    if cutoff is not None:
        xc = np.log10(cutoff) if log else cutoff
        ax.axvline(xc, color=st.INK2, lw=1, ls=(0, (4, 2)), zorder=4)
        ax.text(xc, 1, f' cutoff {cutoff:g}', transform=ax.get_xaxis_transform(), ha='left', va='top', fontsize=8,
                color=st.INK2)
    if by is not None and len(groups) > 1:
        st.legend(ax, loc='upper right')
    _style(ax, title or f'{curve} distribution', f'log10 {curve}' if log else curve, 'Samples',
           f'{int(values.notna().sum()):,} samples' + (f' · {groups[0][0]}' if by is not None and len(groups) == 1 else ''))
    st.frame(ax, 'y')
    return ax


@themed
def paySummaryPlot(summary:pd.DataFrame, title:str='Net thickness by zone', figsize:tuple=(7, 4.5), ax=None):

    r'''
    Gross and ROCK, RES and PAY net thickness of each zone, in the flag colours used by the log plots

    Parameters
    ----------
    summary : pd.DataFrame
        `Quanti.paySummary(...).data`, or a table with Zone, Flag, Gross (m) and Net (m) columns

    Example
    -------
    >>> paySummaryPlot(pp.paySummary('KPD-12').data)
    '''

    ax = _axes(ax, figsize)
    s = summary.rename(columns={'Zones': 'Zone', 'Flag Name': 'Flag', 'Gross (m)': 'Gross', 'Net (m)': 'Net'})
    zones = list(pd.unique(s['Zone']))
    x = np.arange(len(zones))
    gross = s.groupby('Zone', sort=False)['Gross'].first().reindex(zones)
    ax.bar(x, gross, width=.86, color='#ecebe5', label='Gross', zorder=1)
    w = .24
    for i, flag in enumerate(['ROCK', 'RES', 'PAY']):
        net = s[s['Flag'] == flag].set_index('Zone')['Net'].reindex(zones)
        bars = ax.bar(x + (i - 1) * (w + .02), net, width=w, color=st.FLAG_COLORS[flag], label=f'{flag} net', zorder=2)
        for b, v in zip(bars, net):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7.5,
                        color=st.INK2)
    for xi, g in zip(x, gross):
        ax.text(xi + .43, g, f'gross {g:.1f}', ha='right', va='bottom', fontsize=7.5, color=st.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=30 if len(zones) > 4 else 0, ha='right' if len(zones) > 4 else 'center')
    ax.set_ylim(0, float(np.nanmax(gross)) * 1.12)
    st.legend(ax, loc='upper center', bbox_to_anchor=(.5, -.1 if len(zones) <= 4 else -.25), ncol=4,
              handlelength=1, columnspacing=1.2)
    _style(ax, title, '', 'Thickness')
    st.frame(ax, 'y')
    ax.tick_params(axis='x', length=0)
    return ax
