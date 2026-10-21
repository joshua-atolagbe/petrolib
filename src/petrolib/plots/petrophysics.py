'''
Petrophysics plots: Buckles, Winland R35, RQI/FZI, Lorenz, M-N, MID and Thomas-Stieber cross plots,
cutoff sensitivity, mineral volumes and saturation uncertainty

Each function draws on `ax` (a new figure if None) and returns the Axes

'''
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..petrophysics.rocktype import rqi, normalized_porosity, winland_r35, lorenz, PORE_TYPES
from ..petrophysics.lithology import minerals as _minerals, m_n, rhomaa, umaa
from ..petrophysics.netpay import sample_thickness

_MINERAL_COLORS = {'Quartz': 'gold', 'Calcite': 'royalblue', 'Dolomite': 'purple', 'Anhydrite': 'magenta',
                   'Halite': 'cyan', 'Gypsum': 'green'}


def _axes(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _scatter(ax, df:pd.DataFrame, x, y, hue:str=None, cmap:str='viridis', **kwargs):

    '''
    Scatter of x against y, colored by a numeric or categorical `hue` column
    '''

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if hue is None:
        ax.scatter(x, y, s=8, c='tab:blue', alpha=.6, **kwargs)
    elif pd.api.types.is_numeric_dtype(df[hue]):
        sc = ax.scatter(x, y, s=8, c=df[hue], cmap=cmap, alpha=.8, **kwargs)
        plt.colorbar(sc, ax=ax, label=hue)
    else:
        for value in pd.unique(df[hue].dropna()):
            on = (df[hue] == value).to_numpy()
            ax.scatter(x[on], y[on], s=8, alpha=.8, label=str(value), **kwargs)
        ax.legend(title=hue, loc='best', fontsize='small')


def _style(ax, title:str, xlabel:str, ylabel:str):
    ax.set_title(title, size=14, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which='major', linestyle=':', linewidth=.9, color='grey')
    ax.grid(which='minor', linestyle=':', linewidth=.5, color='lightgrey')


def bucklesPlot(df:pd.DataFrame, phi:str='PHIE', sw:str='SW', bvw:tuple=(.02, .04, .06, .08, .1),
                hue:str=None, cmap:str='viridis', title:str='Buckles Plot', figsize:tuple=(8, 6), ax=None):

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
    xmax = max(.35, float(np.nanmax(df[phi])) * 1.05)
    p = np.linspace(.005, xmax, 200)
    for b in bvw:
        ax.plot(p, b / p, '--', color='k', linewidth=.8)
        x = min(xmax * .95, b / .12)
        ax.annotate(f'BVW {b:g}', (x, b / x), fontsize='small', ha='left', va='bottom')
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1)
    _style(ax, title, f'{phi} (v/v)', f'{sw} (v/v)')
    return ax


def winlandPlot(df:pd.DataFrame, phi:str='PHIE', k:str='Perm', r35:tuple=(.1, .5, 2., 10.), hue:str=None,
                cmap:str='viridis', title:str='Winland R35 Plot', figsize:tuple=(8, 6), ax=None):

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
    p = np.linspace(.01, .4, 200)
    for r in r35:
        kline = 10**((np.log10(r) - .732 + .864 * np.log10(100 * p)) / .588)
        ax.plot(p, kline, '--', color='k', linewidth=.8)
        ax.annotate(f'R35 = {r:g} um', (p[-1], kline[-1]), fontsize='small', ha='right', va='bottom')
    ax.set_yscale('log'); ax.set_xlim(0, .4); ax.set_ylim(.001, 1e5)
    _style(ax, title, f'{phi} (v/v)', f'{k} (mD)')
    return ax


def rqiPlot(df:pd.DataFrame, phi:str='PHIE', k:str='Perm', fzi_lines:tuple=(.1, .3, 1., 3., 10.), hue:str=None,
            cmap:str='viridis', title:str='RQI - Normalized Porosity Plot', figsize:tuple=(8, 6), ax=None):

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
        ax.plot(z, f * z, '--', color='k', linewidth=.8)
        ax.annotate(f'FZI {f:g}', (z[-1], f * z[-1]), fontsize='small', ha='right', va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.01, 1); ax.set_ylim(.001, 100)
    _style(ax, title, 'PHIZ = PHIE / (1 - PHIE)', 'RQI (um)')
    return ax


def lorenzPlot(df:pd.DataFrame, phi:str='PHIE', k:str='Perm', depth:str=None, stratigraphic:bool=True,
               title:str=None, figsize:tuple=(6, 6), ax=None):

    r'''
    (Stratigraphic modified) Lorenz plot: cumulative flow capacity against cumulative storage capacity.
    Steep segments are speed zones, flat ones baffles. The title shows the Lorenz coefficient

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
    ax.plot(curve['STORAGE'], curve['FLOW'], '-', color='tab:red', linewidth=1.5)
    ax.plot([0, 1], [0, 1], '--', color='k', linewidth=.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    name = 'Stratigraphic Modified Lorenz Plot' if stratigraphic else 'Lorenz Plot'
    _style(ax, title or f'{name} (Lorenz coefficient {lc:.2f})', 'Cumulative storage capacity (PHI h)',
           'Cumulative flow capacity (k h)')
    return ax


def _mineral_points(ax, x, y, names):
    for name, xi, yi in zip(names, x, y):
        ax.scatter(xi, yi, s=80, marker='D', color=_MINERAL_COLORS.get(name, 'k'), edgecolor='k', zorder=5)
        ax.annotate(name, (xi, yi), xytext=(5, 5), textcoords='offset points', fontweight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=.8))
    triangle = [names.index(m) for m in ('Quartz', 'Calcite', 'Dolomite', 'Quartz') if m in names]
    if len(triangle) == 4:
        ax.plot(np.asarray(x)[triangle], np.asarray(y)[triangle], '-', color='k', linewidth=1)


def mnPlot(df:pd.DataFrame, rhob:str='RHOB', nphi:str='NPHI', dt:str='DT', hue:str=None, cmap:str='viridis',
           mineral_names:tuple=('Quartz', 'Calcite', 'Dolomite', 'Anhydrite', 'Gypsum'), rho_fluid:float=1.,
           dt_fluid:float=189., title:str='M-N Plot', figsize:tuple=(8, 6), ax=None):

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
    ax.set_xlim(.3, 1.); ax.set_ylim(.5, 1.1)
    _style(ax, title, 'N', 'M')
    return ax


def midPlot(df:pd.DataFrame, rhob:str='RHOB', pef:str='PEF', phit:str='PHIT', hue:str=None, cmap:str='viridis',
            mineral_names:tuple=('Quartz', 'Calcite', 'Dolomite', 'Anhydrite', 'Halite'), rho_fluid:float=1.,
            u_fluid:float=.398, title:str='Matrix Identification (MID) Plot', figsize:tuple=(8, 6), ax=None):

    r'''
    Matrix identification plot: apparent matrix density (RHOMAA) against apparent matrix
    volumetric photoelectric factor (UMAA), with mineral points

    Example
    -------
    >>> from petrolib.plots import midPlot
    >>> midPlot(df, phit='PHIT', hue='GR')
    '''

    ax = _axes(ax, figsize)
    _scatter(ax, df, umaa(df[pef], df[rhob], df[phit], u_fluid), rhomaa(df[rhob], df[phit], rho_fluid), hue, cmap)
    mm = _minerals().loc[list(mineral_names)]
    _mineral_points(ax, umaa(mm['PEF'], mm['RHOB'], 0.), mm['RHOB'], list(mineral_names))
    ax.set_xlim(2, 17); ax.set_ylim(3.1, 2.)
    _style(ax, title, 'UMAA (barns/cc)', 'RHOMAA (g/cc)')
    return ax


def thomasStieberPlot(df:pd.DataFrame, phi_sand:float, phi_shale:float, vsh:str='VShale', phit:str='PHIT',
                      hue:str=None, cmap:str='viridis', title:str='Thomas-Stieber Plot', figsize:tuple=(8, 6), ax=None):

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
    for (a, b), label, style in [((clean, shale), 'laminar', '-'), ((clean, dispersed), 'dispersed', '-'),
                                 ((dispersed, shale), '', '-'), ((clean, structural), 'structural', ':')]:
        ax.plot([a[0], b[0]], [a[1], b[1]], style, color='k', linewidth=1.2, label=label or None)
    ax.annotate('clean sand', clean, xytext=(5, 5), textcoords='offset points')
    ax.annotate('shale', shale, xytext=(-40, 5), textcoords='offset points')
    ax.set_xlim(0, 1); ax.set_ylim(0, max(phi_sand, phi_shale) * 1.3)
    ax.legend(loc='upper right', fontsize='small')
    _style(ax, title, f'{vsh} (v/v)', f'{phit} (v/v)')
    return ax


def cutoffPlot(sensitivity:pd.DataFrame, right:str='Average Water Saturation', title:str=None,
               figsize:tuple=(8, 5), ax=None):

    r'''
    Net pay against cutoff value, from `petrolib.petrophysics.cutoff_sensitivity`, with a second
    summation column (default average water saturation) on the right axis

    Example
    -------
    >>> sens = cutoff_sensitivity(zone_df, cutoff='por', values=np.arange(0, .2, .01))
    >>> cutoffPlot(sens, right='HCPV')
    '''

    ax = _axes(ax, figsize)
    cutoff = sensitivity['cutoff'].iloc[0]
    ax.plot(sensitivity['value'], sensitivity['Net'], 'o-', color='tab:green', label='Net pay')
    ax.set_ylabel('Net pay', color='tab:green')
    ax2 = ax.twinx()
    ax2.plot(sensitivity['value'], sensitivity[right], 's--', color='tab:red', label=right)
    ax2.set_ylabel(right, color='tab:red')
    _style(ax, title or f'Net Pay Sensitivity to the {cutoff.upper()} Cutoff', f'{cutoff.upper()} cutoff', 'Net pay')
    ax.yaxis.label.set_color('tab:green')
    return ax


def volumePlot(volumes:pd.DataFrame, depth, components:list=None, colors:dict=None, title:str='Volumes',
               figsize:tuple=(4, 12), ax=None):

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
    colors = {**_MINERAL_COLORS, 'Water': 'deepskyblue', 'Oil': 'green', 'Gas': 'red', **(colors or {})}
    d = np.asarray(depth, dtype=float)
    left = np.zeros(len(volumes))
    for c in components:
        right = left + np.nan_to_num(volumes[c].to_numpy(dtype=float))
        ax.fill_betweenx(d, left, right, color=colors.get(c), label=c, linewidth=0)
        left = right
    ax.set_xlim(0, 1); ax.set_ylim(np.nanmax(d), np.nanmin(d))
    ax.legend(loc='lower right', fontsize='small')
    _style(ax, title, 'Volume fraction', 'Depth')
    return ax


def swUncertaintyPlot(mc:pd.DataFrame, depth, sw=None, title:str='SW Uncertainty', figsize:tuple=(4, 12), ax=None):

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
    ax.fill_betweenx(d, mc['SW_P10'], mc['SW_P90'], color='lightblue', label='P10-P90')
    ax.plot(mc['SW_P50'], d, color='tab:blue', linewidth=.8, label='P50')
    if sw is not None:
        ax.plot(np.asarray(sw, dtype=float), d, color='k', linewidth=.6, label='SW')
    ax.set_xlim(1, 0); ax.set_ylim(np.nanmax(d), np.nanmin(d))
    ax.legend(loc='lower left', fontsize='small')
    _style(ax, title, 'SW (v/v)', 'Depth')
    return ax


def hinglePlot(df:pd.DataFrame, phi:str='PHIE', rt:str='RT', rw:float=.03, a:float=1., m:float=2., n:float=2.,
               sw_lines:tuple=(1., .8, .6, .4, .2), hue:str=None, cmap:str='viridis', title:str='Hingle Plot',
               figsize:tuple=(8, 6), ax=None):

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
    _scatter(ax, df, df[phi], y, hue, cmap)
    ymax = float(np.nanpercentile(y, 99.5)) * 1.1
    p = np.linspace(0, .4, 50)
    for s in sw_lines:
        slope = s**(n / m) / (a * rw)**(1 / m)
        ax.plot(p, slope * p, '--', color='k', linewidth=.8)
        x_end = min(p[-1], .97 * ymax / slope)                    #label where the line leaves the plot
        ax.annotate(f'SW {s:g}', (x_end, slope * x_end), fontsize='small', ha='right', va='bottom')
    ax.set_xlim(0, .4); ax.set_ylim(0, ymax)
    rt_ticks, last = [], None
    for r in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000):
        if r**(-1 / m) <= ymax and (last is None or last - r**(-1 / m) >= .04 * ymax):
            rt_ticks.append(r)
            last = r**(-1 / m)
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks([r**(-1 / m) for r in rt_ticks])
    right.set_yticklabels([str(r) for r in rt_ticks])
    right.set_ylabel(f'{rt} (ohm-m)')
    _style(ax, title, f'{phi} (v/v)', f'{rt}^(-1/{m:g})')
    return ax


def movableHydrocarbonPlot(df:pd.DataFrame, sw:str='SW', sxo:str='SXO', mhi_lines:tuple=(.6, .7, 1.), hue:str=None,
                           cmap:str='viridis', title:str='Movable Hydrocarbons', figsize:tuple=(7, 6), ax=None):

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
        ax.plot(x, np.minimum(mhi * x, 1), '--', color='k', linewidth=.8)
        ax.annotate(f'MHI {mhi:g}', (1, min(mhi, 1)), fontsize='small', ha='right', va='bottom')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _style(ax, title, f'{sxo} (v/v)', f'{sw} (v/v)')
    return ax


def histogramPlot(df:pd.DataFrame, curve:str, by:str=None, cutoff:float=None, bins:int=40, log:bool=False,
                  title:str=None, figsize:tuple=(8, 5), ax=None):

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
    for name, v in groups:
        ax.hist(v.dropna(), bins=edges, histtype='step', linewidth=1.5, label=None if name is None else str(name))
    if cutoff is not None:
        ax.axvline(np.log10(cutoff) if log else cutoff, color='tab:red', linestyle='--', linewidth=1.2,
                   label=f'cutoff {cutoff:g}')
    if by is not None or cutoff is not None:
        ax.legend(loc='best', fontsize='small')
    _style(ax, title or f'{curve} Distribution', f'log10 {curve}' if log else curve, 'Samples')
    return ax


def paySummaryPlot(summary:pd.DataFrame, title:str='Net Thickness by Zone', figsize:tuple=(9, 5), ax=None):

    r'''
    Gross and ROCK, RES and PAY net thickness of each zone, in the colors of the pay summary workbook

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
    colors = {'ROCK': '#FFFF00', 'RES': '#00AA00', 'PAY': '#CC0000'}
    x = np.arange(len(zones))
    gross = s.groupby('Zone', sort=False)['Gross'].first().reindex(zones)
    ax.bar(x, gross, width=.8, color='none', edgecolor='black', linewidth=1, label='Gross')
    for i, flag in enumerate(['ROCK', 'RES', 'PAY']):
        net = s[s['Flag'] == flag].set_index('Zone')['Net'].reindex(zones)
        ax.bar(x - .25 + i * .25, net, width=.25, color=colors[flag], edgecolor='black', linewidth=.5, label=f'{flag} net')
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=30 if len(zones) > 4 else 0, ha='right' if len(zones) > 4 else 'center')
    ax.legend(loc='best', fontsize='small')
    _style(ax, title, 'Zone', 'Thickness')
    return ax
