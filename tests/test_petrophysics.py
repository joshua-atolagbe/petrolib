import numpy as np
import pandas as pd
import pytest

from petrolib import petrophysics as pp
from petrolib.petrophysics import Quanti

from conftest import COMPONENTS, RW


# ---- equations --------------------------------------------------------------------------------

def test_series_in_series_out_and_arrays():
    s = pd.Series([.1, .2], index=[5, 7])
    out = pp.archie(s, pd.Series([10., 20.], index=[5, 7]), rw=.05)
    assert isinstance(out, pd.Series) and list(out.index) == [5, 7]
    assert isinstance(pp.archie(np.array([.1, .2]), np.array([10., 20.])), np.ndarray)
    assert isinstance(pp.archie(.2, 10.), float)


def test_vshale_methods():
    igr = pp.gamma_ray_index(np.array([0., 50., 100., 150.]), 0., 100.)
    assert np.allclose(igr, [0, .5, 1, 1])
    assert np.allclose(pp.vshale(.5, 'clavier'), 1.7 - np.sqrt(3.38 - 1.44))
    assert np.allclose(pp.vshale(.5, 'larionov_ter'), .083 * (2**1.85 - 1))
    assert np.allclose(pp.vshale(.5, 'stieber_m_pliocene'), .5 / 2)
    for m in pp.VSHALE_METHODS:
        v = pp.vshale(np.linspace(0, 1, 11), m)
        assert v[0] == 0 and np.all(np.diff(v) >= 0) and v.max() <= 1
    with pytest.raises(ValueError):
        pp.vshale(.5, 'nope')


def test_porosity():
    assert pp.density_porosity(2.65) == 0 and np.isclose(pp.density_porosity(2.32), .2)
    assert np.isclose(pp.neutron_density_porosity(.2, .1), .15)                      # liquid: mean
    assert np.isclose(pp.neutron_density_porosity(.1, .2), np.sqrt((.01 + .04) / 2))   # gas: RMS
    assert np.isclose(pp.sonic_porosity(55.5), 0) and np.isclose(pp.sonic_porosity(189.), 1)
    phi = np.linspace(0, .35, 8)
    dt = 1 / ((1 - phi)**2 / 55.5 + phi / 189.)                                         # RHG forward model
    assert np.allclose(pp.raymer_porosity(dt), phi)
    assert pp.effective_porosity(.1, .8, .2) == 0


def test_shaly_sand_models_reduce_to_archie():
    phi, rt = np.array([.1, .2, .25]), np.array([5., 20., 80.])
    archie = pp.archie(phi, rt, rw=.05, m=2.1, n=2.)
    zero = np.zeros(3)
    assert np.allclose(pp.simandoux(phi, rt, zero, rw=.05, m=2.1), archie)
    assert np.allclose(pp.indonesia(phi, rt, zero, rw=.05, m=2.1), archie)
    assert np.allclose(pp.waxman_smits(phi, rt, qv=0., rw=.05, b=4., m=2.1), archie, atol=1e-9)
    assert np.allclose(pp.dual_water(phi, rt, swb=0., rw=.05, m=2.1), archie, atol=1e-9)
    ws = pp.waxman_smits(phi, rt, qv=.5, rw=.05, b=4., m=2.1)
    assert np.all(ws[archie < 1] < archie[archie < 1])                                   # clay conductivity lowers SW
    assert pp.archie(0., 10.) == 1                                                       # no pore space


def test_fluids():
    assert np.isclose(pp.arps(.05, 75, 75), .05)
    assert np.isclose(pp.arps(.1, 75, 150), .1 * 81.77 / 156.77)
    assert np.isclose(pp.arps(.1, 24, 100, unit='C'), .1 * 45.5 / 121.5)
    assert np.isclose(pp.formation_temperature(1000, 20, bht=80, td=2000), 50)
    assert np.isclose(pp.rwa(1 / .04, .2), 1.)
    assert np.isclose(pp.rw_from_salinity(100000), .0123 + 3647.5 / 100000**.955)


def test_permeability():
    assert np.isclose(pp.timur(.2, .2), 8581 * .2**4.4 / .04)
    assert np.isclose(pp.timur(.2, 0.), 8581 * .2**4.4 / .05**2)                         # SWIRR floor
    assert np.isclose(pp.coates(.2, .2), (100 * .04 * .8 / .2)**2)
    assert np.isclose(pp.morris_biggs(.2, .2, 'gas'), (79 * .008 / .2)**2)
    assert np.isclose(pp.swirr_buckles(.2, .04), .2)
    with pytest.raises(ValueError):
        pp.permeability(.2, .2, 'nope')


def test_rock_typing():
    k, phi = np.array([1., 100.]), np.array([.1, .25])
    assert np.allclose(pp.rqi(k, phi), .0314 * np.sqrt(k / phi))
    assert np.allclose(pp.fzi(k, phi), pp.rqi(k, phi) / (phi / (1 - phi)))
    assert np.allclose(pp.drt(np.array([1., np.e])), [11, 13])
    r35 = pp.winland_r35(k, phi)
    assert np.allclose(np.log10(r35), .732 + .588 * np.log10(k) - .864 * np.log10(100 * phi))
    assert list(pp.pore_type(np.array([20, 5, 1, .2, .05]))) == ['megaport', 'macroport', 'mesoport', 'microport', 'nanoport']
    _, lc_uniform = pp.lorenz(np.ones(50), np.full(50, .2))
    _, lc_varied = pp.lorenz(np.logspace(-2, 3, 50), np.full(50, .2))
    assert abs(lc_uniform) < 1e-9 and .5 < lc_varied < 1


def test_capillary():
    pc = pp.pc_from_height(30., .8)
    assert np.isclose(pc, .433 * .8 * 30 * 3.28084)
    assert np.isclose(pp.height_from_pc(pc, .8), 30.)
    j = np.logspace(-1, 1, 20)
    a, b = pp.fit_saturation_height(j, .3 * j**-.6)
    assert np.isclose(a, .3) and np.isclose(b, -.6)


def test_lithology_and_multimineral(well):
    mm = pp.minerals()
    assert mm.loc['Quartz', 'RHOB'] == 2.65
    M, N = pp.m_n(mm.loc['Calcite', 'RHOB'], mm.loc['Calcite', 'NPHI'], mm.loc['Calcite', 'DT'])
    assert np.isclose(M, .01 * (189 - 47.5) / 1.71) and np.isclose(N, 1 / 1.71)
    assert np.isclose(pp.rhomaa(2.32, .2), (2.32 - .2) / .8)
    assert np.isclose(pp.umaa(5.08, 2.71, 0.), 5.08 * (2.71 + .1883) / 1.0704)

    vol = pp.multimineral(well, COMPONENTS, logs=['RHOB', 'NPHI', 'GR'])
    assert np.allclose(vol['Quartz'], well['VQ'], atol=1e-3)
    assert np.allclose(vol['Illite'], well['VCL'], atol=1e-3)
    assert np.allclose(vol['Water'], well['PHI'], atol=1e-3)
    assert vol['MM_ERROR'].max() < .01


def test_volumetrics_and_uncertainty():
    assert np.isclose(pp.stoiip(1, 1, .2, .3, bo=1.), 7758 * .14)
    assert np.isclose(pp.giip(1e4, 10, .2, .3, bg=.01, unit='metric'), 1e4 * 10 * .14 / .01)
    phi, rt = pd.Series([.15, .2, .25]), pd.Series([10., 20., 40.])
    mc = pp.monte_carlo_archie(phi, rt, rw=(.03, .05, .07), m=(1.8, 2., 2.2), seed=1)
    assert (mc['SW_P10'] <= mc['SW_P50']).all() and (mc['SW_P50'] <= mc['SW_P90']).all()
    fixed = pp.monte_carlo_archie(phi, rt, rw=.05, n_iter=5)
    assert np.allclose(fixed['SW_P50'], pp.archie(phi, rt, rw=.05))


def test_net_pay_summation():
    depth = pd.Series(np.r_[np.arange(100, 105, .5), np.arange(110, 115, .5)])           # gap 105-110
    flag = np.ones(len(depth))
    s = pp.summarize(depth, np.zeros(len(depth)), np.full(len(depth), .2), np.full(len(depth), .5), flag, 100, 115)
    assert np.isclose(s['Gross'], 15) and np.isclose(s['Net'], 9.75)                    # gap is not net
    assert np.isclose(s['HCPV'], .2 * .5 * s['Net'])
    assert np.isclose(pp.weighted_mean(pd.Series([.2, .8]), pd.Series([3., 1.])), .35)
    flags = pp.net_flags(pd.Series([.1, .1, .1, .9]), pd.Series([.2, .2, .01, .2]), pd.Series([.3, .9, .3, .3]), .5, .05, .6)
    assert flags['PAY_NET_FLAG'].tolist() == [1, 0, 0, 0] and flags['RES_NET_FLAG'].tolist() == [1, 1, 0, 0]


def test_cutoff_sensitivity(well):
    well = well.assign(VShale=well['VCL'], PHIE=well['PHI'], SW=well['SW_TRUE'])
    sens = pp.cutoff_sensitivity(well, cutoff='por', values=np.arange(0, .3, .05))
    assert sens['Net'].is_monotonic_decreasing


# ---- Quanti -----------------------------------------------------------------------------------

def run_quanti(well, **kw):
    q = Quanti(well, ['A'], [1010.], [1180.], [1095.], 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', sonic='DT')
    q.vshale(method='linear')
    q.porosity(method=kw.get('por', 'density'), rhob_matrix=2.65, rhob_shale=2.55, nphi_shale=.35)
    q.water_saturation(method=kw.get('sw', 'archie'), rw=RW, rsh=3.)
    q.permeability(method=kw.get('k', 'timur'))
    logs = pd.concat(q.flags(vsh_cutoff=.5, por_cutoff=.05, sw_cutoff=.6))
    return q, logs, q.paySummary('W-1').data


def test_quanti_matches_the_equations(well):
    q, logs, summary = run_quanti(well)
    phid_sh = pp.density_porosity(2.55)
    phie = pp.effective_porosity(pp.density_porosity(logs['RHOB']).clip(lower=0), logs['VShale'], phid_sh)
    assert np.allclose(logs['PHIE'], phie)
    assert np.allclose(logs['SW'], pp.archie(phie, logs['RT'], rw=RW))                   # rw kept through flags()
    assert np.allclose(logs['Perm'], pp.timur(logs['PHIE'], logs['SW']))
    assert (logs['PAY_NET_FLAG'] <= logs['RES_NET_FLAG']).all() and (logs['RES_NET_FLAG'] <= logs['ROCK_NET_FLAG']).all()
    assert set(summary['Flag Name']) == {'ROCK', 'RES', 'PAY'}
    assert np.allclose(summary['Gross'], 170.) and (summary['Net'] <= summary['Gross']).all()
    assert {'PHIH', 'HCPV', 'Average Permeability'} <= set(summary.columns)


@pytest.mark.parametrize('por, sw, k', [('neutron_density', 'simandoux', 'coates'), ('sonic', 'indonesia', 'morris_biggs'),
                                        ('raymer', 'simmandoux', 'timur')])
def test_quanti_methods(well, por, sw, k):
    q, logs, summary = run_quanti(well, por=por, sw=sw, k=k)
    assert logs[['PHIE', 'SW', 'Perm']].notna().all().all()
    assert q.report().data.loc['Permeability'].iloc[0] == k.title()


def test_quanti_unknown_methods(well):
    q = Quanti(well, ['A'], [1010.], [1180.], [1095.], 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')
    with pytest.raises(ValueError):
        q.vshale(method='nope')
    with pytest.raises(ValueError):
        q.porosity(method='nope')


def test_quanti_adjacent_zones(well):
    # zones sharing a boundary, as in most tops files: the boundary sample belongs to both
    q = Quanti(well, ['A', 'B'], [1010., 1100.], [1100., 1180.], [1055., 1140.])
    q.vshale(method='clavier', show_plot=True, palette_op='cubehelix', figsize=(6, 8))
    q.porosity(method='density', rhob_shale=2.55)
    q.water_saturation(rw=RW)
    q.permeability()
    zones = q.flags(vsh_cutoff=.5, por_cutoff=.05, sw_cutoff=.6, show_plot=True, palette_op='cubehelix', figsize=(12, 8))
    assert zones[0]['DEPTH'].max() == zones[1]['DEPTH'].min() == 1100.
    summary = q.paySummary('W-1').data
    rock = summary[summary['Flag Name'] == 'ROCK']
    assert rock['Gross'].tolist() == [90., 80.] and (summary['Net'] <= summary['Gross']).all()
    import matplotlib.pyplot as plt
    plt.close('all')
