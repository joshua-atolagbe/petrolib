import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from petrolib import plots
from petrolib import petrophysics as pp
from petrolib.io import Zonation


@pytest.fixture
def evaluated(well):
    return well.assign(VShale=well['VCL'], PHIE=well['PHI'], PHIT=well['PHI'] + .1 * well['VCL'], SW=well['SW_TRUE'],
                       Perm=pp.timur(well['PHI'], well['SW_TRUE']))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('plot, kwargs', [
    (plots.bucklesPlot, dict(hue='VShale')),
    (plots.winlandPlot, dict()),
    (plots.rqiPlot, dict(hue='WELL')),
    (plots.lorenzPlot, dict(depth='DEPTH')),
    (plots.mnPlot, dict(hue='GR')),
    (plots.midPlot, dict()),
    (plots.thomasStieberPlot, dict(phi_sand=.27, phi_shale=.1)),
])
def test_new_crossplots_return_axes(evaluated, plot, kwargs):
    assert isinstance(plot(evaluated, **kwargs), plt.Axes)


def test_track_plots(evaluated):
    sens = pp.cutoff_sensitivity(evaluated, cutoff='sw', values=np.arange(.2, 1, .1))
    assert isinstance(plots.cutoffPlot(sens, right='HCPV'), plt.Axes)
    vol = pp.multimineral(evaluated, {'Quartz': {'RHOB': 2.65, 'GR': 20}, 'Illite': {'RHOB': 2.55, 'GR': 150},
                                      'Water': {'RHOB': 1., 'GR': 0}})
    assert isinstance(plots.volumePlot(vol, evaluated['DEPTH']), plt.Axes)
    mc = pp.monte_carlo_archie(evaluated['PHIE'], evaluated['RT'], rw=(.04, .05, .06), n_iter=50, seed=0)
    assert isinstance(plots.swUncertaintyPlot(mc, evaluated['DEPTH'], sw=evaluated['SW']), plt.Axes)


def test_existing_plots_still_run(evaluated):
    plots.tripleCombo(evaluated, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', 1000, 1150, fill='right', palette_op='rainbow',
                      limit='left')
    plots.plotLog(evaluated, 'DEPTH', ['GR', 'RT', 'RHOB'], 1000, 1150)
    plots.plotLogs(evaluated, 'DEPTH', ['GR', ['RHOB', 'NPHI']], 1000, 1150)
    plots.crossPlot(evaluated, 'NPHI', 'RHOB', hue='GR', color_code='num')
    plots.picketPlot(evaluated, rt='RT', por='PHIE')
    Zonation(evaluated, zones=[{'A': [1010, 1100]}]).plotZone('DEPTH', ['GR', 'RT'], 1000, 1150)


def test_facies_combo_with_facies_labels(evaluated):
    from petrolib.processing import model_facies
    fac = model_facies(evaluated.copy(), gr='GR', env='SS')
    plots.plotFaciesZoneCombo(fac, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', 'litho', 1000, 1150, ['#ffff00', '#bebebe'],
                              sorted(fac['litho'].unique()), [1010.], [1100.], ['A'], limit='left', fill='right',
                              palette_op='rainbow')
    facies_track = plt.gcf().axes[4]
    assert list(facies_track.images[0].get_extent()) == [0, 1, fac['DEPTH'].max(), fac['DEPTH'].min()]   #in depth
    assert tuple(facies_track.get_ylim()) == (1150, 1000)


def test_more_analytical_plots(evaluated):
    df = evaluated.assign(SXO=np.clip(evaluated['SW_TRUE'] + .3, 0, 1), Zone=np.where(evaluated['DEPTH'] < 1100, 'A', 'B'))
    for ax in [plots.hinglePlot(df, rw=.05, hue='VShale'), plots.movableHydrocarbonPlot(df, hue='PHIE'),
               plots.histogramPlot(df, 'PHIE', by='Zone', cutoff=.1), plots.histogramPlot(df, 'RT', log=True)]:
        assert isinstance(ax, plt.Axes)
        assert all(line.get_linestyle() == ':' for line in ax.get_xgridlines() + ax.get_ygridlines())   #dotted grid
    summary = pd.DataFrame({'Zones': ['A'] * 3 + ['B'] * 3, 'Flag Name': ['ROCK', 'RES', 'PAY'] * 2,
                            'Gross': [90.] * 3 + [80.] * 3, 'Net': [80., 60., 40., 70., 50., 20.]})
    assert isinstance(plots.paySummaryPlot(summary), plt.Axes)


def test_geomechanics_plots(evaluated):
    mech = evaluated.join(pp.elastic_moduli(evaluated['RHOB'], evaluated['DT'], 1.8 * evaluated['DT']))
    mech['BI'] = pp.brittleness(mech['E'], mech['PR'])
    mech['UCS'] = pp.ucs_mcnally(mech['DT'])
    assert len(plots.mechanicalPlot(mech)) == 4
    assert len(plots.mechanicalPlot(mech, curves=['E', 'G', 'K'])) == 3
    z = mech['DEPTH']
    sv, ph = pp.overburden(z, mech['RHOB']), pp.hydrostatic_pressure(z)
    pore = pp.pore_pressure_eaton(sv, ph, mech['DT'], pp.normal_compaction_trend(z, mech['DT']))
    for gradient in (False, True):
        assert isinstance(plots.pressurePlot(z, ph, pore, sv, pp.fracture_pressure_eaton(sv, pore, mech['PR']),
                                             mud_weight=1.1, gradient=gradient), plt.Axes)
    assert isinstance(plots.rockPhysicsPlot(mech, hue='SW'), plt.Axes)


def test_map_extent_ignores_far_parts_of_an_area():
    geometry = pytest.importorskip('shapely.geometry')
    from types import SimpleNamespace
    from petrolib.plots.maps import _extent
    #a mainland and an overseas island, as in the Natural Earth outline of the Netherlands
    country = geometry.MultiPolygon([geometry.box(3.3, 50.7, 7.2, 53.6), geometry.box(-68.5, 12., -68., 12.4)])
    xlim, ylim = _extent(SimpleNamespace(geometry=[country]), np.array([6.64, 6.78]), np.array([53.19, 53.26]))
    assert 3 < xlim[0] < 3.3 and 7.2 < xlim[1] < 7.5
    assert 50.4 < ylim[0] < 50.7 and 53.6 < ylim[1] < 53.9
