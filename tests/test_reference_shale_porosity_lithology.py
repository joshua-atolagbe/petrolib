'''
Reference tests for petrophysics.shale, petrophysics.porosity and petrophysics.lithology

Expected values are written out from the published equations or hand-calculated (arithmetic in the
comments), independent of petrolib. Sources:

[AK]   Asquith G and Krygowski D (2004) Basic Well Log Analysis, AAPG Methods in Exploration 16,
       pp. 31-35: gamma ray index, Larionov (1969), Clavier (1971), Stieber (1970). Checked through
       Saputra I (2008) "Shale Volume Calculation", HRS course slides citing [AK]
       https://www.slideshare.net/ArshAttri1/883shalevolumecalculationpdf
[JK]   Johanna U and Kusumah EP (2023) JGEET 8(02-2), doi:10.25299/jgeet.2023.8.02-2.13880:
       Stieber (1970) Vsh = IGR / (3 - 2 IGR)
[SLB]  Schlumberger Log Interpretation Charts, 2013 edition
       https://www.spec2000.net/freepubs-pdf/SLB%202013.pdf
       Por-1 (sonic), Por-5 example (density), Por-11 (CNL-density), Lith-5/6 (Umaa), Lith-7 (M-N),
       Lith-9 (rhomaa, tmaa), Appendix B (mineral responses), Appendix C (acoustic), Appendix E (M, N)
[KGS]  Kansas Geological Survey, Nuclear logs: Pe quartz 1.81, dolomite 3.14, calcite 5.08 b/e
       https://www.kgs.ku.edu/Publications/Bulletins/LA/04_nuclear.html
[RHG]  Raymer, Hunt and Gardner (1980), SPWLA 21st symposium: V = (1 - phi)^2 Vma + phi Vf, phi < 37%
       (as quoted by Dvorkin and Nur, "Time-average equation revisited", Geophysics 1998)
[GAS]  Root mean square neutron-density porosity for gas, https://en.wikipedia.org/wiki/Gas_porosity
'''
import numpy as np
import pandas as pd
import pytest

import petrolib.petrophysics as pp

IGR = np.linspace(0, 1, 101)

#published forms [AK], [JK], written out independently of petrolib
VSH_REFERENCE = {
    'linear':             lambda i: i,
    'clavier':            lambda i: 1.7 - np.sqrt(3.38 - (i + .7)**2),
    'larionov_ter':       lambda i: .083 * (2**(3.7 * i) - 1),
    'larionov_older':     lambda i: .33 * (2**(2 * i) - 1),
    'stieber_1':          lambda i: i / (2 - i),          #Stieber family, A = 2
    'stieber_2':          lambda i: i / (4 - 3 * i),      #Stieber family, A = 4
    'stieber_m_pliocene': lambda i: i / (3 - 2 * i),      #Stieber (1970), A = 3
}


# ---------------------------------------------------------------- gamma ray index and shale volume

def test_gamma_ray_index_hand_values():
    # (70 - 20) / (120 - 20) = 0.5 ; (45 - 20) / 100 = 0.25 ; below sand -> 0 ; above shale -> 1
    gr = pd.Series([70., 45., 10., 150., np.nan], index=[10, 11, 12, 13, 14])
    igr = pp.gamma_ray_index(gr, 20, 120)
    assert isinstance(igr, pd.Series) and list(igr.index) == [10, 11, 12, 13, 14]
    assert np.allclose(igr.iloc[:4], [.5, .25, 0., 1.])
    assert np.isnan(igr.iloc[4])
    assert isinstance(pp.gamma_ray_index(70., 20, 120), float)


def test_gamma_ray_index_degenerate_baselines_keep_nan():
    # gr_shale <= gr_sand: 0.5 by convention, but missing GR stays missing
    igr = pp.gamma_ray_index(np.array([np.nan, 10., 60.]), 50, 50)
    assert np.isnan(igr[0]) and np.allclose(igr[1:], .5)


def test_vshale_matches_published_equations():
    for method, f in VSH_REFERENCE.items():
        assert np.allclose(pp.vshale(IGR, method), np.clip(f(IGR), 0, 1), atol=1e-12), method


def test_vshale_hand_values_at_half():
    # clavier:        1.7 - sqrt(3.38 - 1.2^2) = 1.7 - sqrt(1.94) = 1.7 - 1.392839 = 0.307161
    # larionov_ter:   0.083 (2^1.85 - 1) = 0.083 * 2.604997 = 0.216215
    # larionov_older: 0.33 (2^1 - 1) = 0.33
    # stieber:        .5/1.5 = .333333 ; .5/2.5 = .2 ; .5/2 = .25
    expected = {'linear': .5, 'clavier': .307161, 'larionov_ter': .216215, 'larionov_older': .33,
                'stieber_1': 1 / 3, 'stieber_2': .2, 'stieber_m_pliocene': .25}
    for method, v in expected.items():
        assert np.isclose(pp.vshale(.5, method), v, atol=1e-6), method


def test_vshale_endpoints():
    for method in pp.VSHALE_METHODS:
        assert pp.vshale(0., method) == 0, method
    for method in ('linear', 'clavier', 'stieber_1', 'stieber_2', 'stieber_m_pliocene'):
        assert np.isclose(pp.vshale(1., method), 1.), method
    # the published Larionov forms stop just short of 1 at IGR = 1
    assert np.isclose(pp.vshale(1., 'larionov_ter'), .083 * (2**3.7 - 1))     # 0.995671
    assert np.isclose(pp.vshale(1., 'larionov_older'), .99)                    # 0.33 * 3


def test_vshale_monotonic_and_below_linear():
    # [AK]: the non-linear relations give less shale than the linear one
    for method in pp.VSHALE_METHODS:
        v = pp.vshale(IGR, method)
        assert np.all(np.diff(v) >= 0), method
        assert np.all(v <= IGR + 1e-12), method
        assert np.all((v >= 0) & (v <= 1)), method


def test_vshale_out_of_range_igr_is_clipped():
    # IGR outside [0, 1] behaves like the nearest end point (clavier used to give NaN above ~1.14,
    # stieber used to drop to 0 past its pole)
    igr = np.array([-.5, 1.2, 1.5, 2.5])
    for method in pp.VSHALE_METHODS:
        v = pp.vshale(igr, method)
        assert v[0] == 0, method
        assert np.allclose(v[1:], pp.vshale(1., method)), method


def test_vshale_nan_and_series():
    s = pd.Series([np.nan, .5], index=['a', 'b'])
    for method in pp.VSHALE_METHODS:
        v = pp.vshale(s, method)
        assert list(v.index) == ['a', 'b'] and np.isnan(v['a']), method
    with pytest.raises(ValueError):
        pp.vshale(.5, 'larionov')


def test_vshale_nd_hand_values():
    # (0.30 - 0.15) / (0.33 - 0.18) = 1 ; (0.20 - 0.15) / 0.15 = 0.333333 ; gas crossover -> 0
    v = pp.vshale_nd(np.array([.30, .20, .10, np.nan]), np.array([.15, .15, .25, .1]), .33, .18)
    assert np.allclose(v[:3], [1., 1 / 3, 0.]) and np.isnan(v[3])
    s = pp.vshale_nd(pd.Series([.2], index=[7]), .15, .33, .18)
    assert list(s.index) == [7]


def test_vshale_sp_hand_values():
    # (SP - SSP) / (SPsh - SSP) = 1 - PSP / SSP with SPsh = 0 ; -40 mV of a -80 mV SSP -> 0.5
    sp = np.array([-80., -60., -40., 0., -90., 10.])
    v = pp.vshale_sp(sp, sp_clean=-80, sp_shale=0)
    assert np.allclose(v, [0., .25, .5, 1., 0., 1.])
    assert np.allclose(v[:4], 1 - sp[:4] / -80.)


# ---------------------------------------------------------------- porosity

def test_density_porosity_slb_por5_example():
    # [SLB] Por-5 example: rhob 2.31, rhoma 2.71, rhof 1.1 -> 25 p.u. ; 0.40 / 1.61 = 0.248447
    assert np.isclose(pp.density_porosity(2.31, 2.71, 1.1), .40 / 1.61)
    assert round(pp.density_porosity(2.31, 2.71, 1.1), 2) == .25


def test_density_porosity_inverse_and_limits():
    phi = np.linspace(0, .4, 9)
    for rhoma in (2.65, 2.71, 2.87):
        rhob = phi * 1. + (1 - phi) * rhoma          # mass balance
        assert np.allclose(pp.density_porosity(rhob, rhoma, 1.), phi)
    # 2.32 sandstone: 0.33 / 1.65 = 0.2 ; not clipped: 2.70 sandstone -> -0.05 / 1.65
    assert np.isclose(pp.density_porosity(2.32), .2)
    assert np.isclose(pp.density_porosity(2.70), -.05 / 1.65)
    s = pp.density_porosity(pd.Series([2.32, np.nan], index=[3, 4]))
    assert list(s.index) == [3, 4] and np.isnan(s[4])


def test_neutron_density_porosity_liquid_and_gas():
    # liquid: (0.30 + 0.20) / 2 = 0.25 ; gas [GAS]: sqrt((0.10^2 + 0.25^2) / 2) = sqrt(0.03625) = 0.190394
    assert np.isclose(pp.neutron_density_porosity(.30, .20), .25)
    assert np.isclose(pp.neutron_density_porosity(.10, .25), .190394, atol=1e-6)
    # both forms meet where the logs agree
    assert np.isclose(pp.neutron_density_porosity(.2, .2), .2)
    s = pp.neutron_density_porosity(pd.Series([.3, np.nan], index=[5, 6]), np.array([.2, .2]))
    assert list(s.index) == [5, 6] and np.isnan(s[6])


def test_neutron_density_porosity_bounds():
    rng = np.random.default_rng(0)
    n, d = rng.uniform(0, .4, 500), rng.uniform(0, .4, 500)
    phi = pp.neutron_density_porosity(n, d)
    assert np.all(phi >= np.minimum(n, d) - 1e-12) and np.all(phi <= np.maximum(n, d) + 1e-12)
    assert np.all(phi >= (n + d) / 2 - 1e-12)     # RMS >= mean


def test_neutron_density_slb_por11_example():
    # [SLB] Por-11: CNL 16.5 p.u. (limestone units), rhob 2.38 -> crossplot porosity 18 p.u.
    # PHID(lime) = (2.71 - 2.38) / 1.71 = 0.192982. PHIN < PHID, so the RMS form is used (the quartz
    # in this mix also puts the neutron below the density): sqrt((0.027225 + 0.037242) / 2) = 0.179537
    phid = pp.density_porosity(2.38, 2.71)
    assert np.isclose(pp.neutron_density_porosity(.165, phid), .179537, atol=1e-6)
    assert abs(pp.neutron_density_porosity(.165, phid) - .18) < .01   # chart read to 1 p.u.


def test_shale_corrected_nd_porosity_hand_values():
    # liquid: PHIN_c = 0.30 - 0.5*0.33 = 0.135 ; PHID_c = 0.20 - 0.5*0.18 = 0.11 ; mean = 0.1225
    assert np.isclose(pp.shale_corrected_nd_porosity(.30, .20, .5, .33, .18), .1225)
    # gas: PHIN_c = 0.10 - 0.2*0.33 = 0.034 ; PHID_c = 0.25 - 0.2*0.18 = 0.214
    #      sqrt((0.034^2 + 0.214^2) / 2) = sqrt(0.023476) = 0.153219
    assert np.isclose(pp.shale_corrected_nd_porosity(.10, .25, .2, .33, .18), .153219, atol=1e-6)
    # pure shale reads zero effective porosity ; Vsh = 0 is the plain combination
    assert pp.shale_corrected_nd_porosity(.33, .18, 1., .33, .18) == 0
    assert np.isclose(pp.shale_corrected_nd_porosity(.3, .2, 0., .33, .18), pp.neutron_density_porosity(.3, .2))
    s = pp.shale_corrected_nd_porosity(pd.Series([.3], index=[9]), .2, .5, .33, .18)
    assert list(s.index) == [9]


def test_sonic_porosity_slb_por1_examples():
    # [SLB] Por-1, vf = 5,300 ft/s (189 us/ft)
    # consolidated: dt 76, vma 18,000 ft/s (55.5 us/ft) -> 15 p.u. ; (76 - 55.5) / 133.5 = 0.153558
    assert np.isclose(pp.sonic_porosity(76., 55.5, 189.), 20.5 / 133.5)
    assert round(pp.sonic_porosity(76., 55.5, 189.), 2) == .15
    # unconsolidated: dt 100 in a 28 p.u. water sand -> Bcp = (44.5 / 133.5) / 0.28 = 1.19 (chart 1.2)
    cp = pp.sonic_porosity(100.) / .28
    assert round(cp, 1) == 1.2
    # dt 110 with Bcp 1.2 -> 34 p.u. ; (54.5 / 133.5) / 1.2 = 0.340200
    assert round(pp.sonic_porosity(110., cp=1.2), 2) == .34


def test_sonic_porosity_limits_and_inverse():
    assert pp.sonic_porosity(55.5) == 0 and np.isclose(pp.sonic_porosity(189.), 1)
    phi = np.linspace(0, .4, 9)
    for dtma in (55.5, 47.5, 43.5):
        dt = phi * 189. + (1 - phi) * dtma            # Wyllie time average
        assert np.allclose(pp.sonic_porosity(dt, dtma), phi)
    s = pp.sonic_porosity(pd.Series([76., np.nan], index=[1, 2]))
    assert list(s.index) == [1, 2] and np.isnan(s[2])


def test_raymer_porosity_inverse_of_published_form():
    # [RHG] 1/dt = (1 - phi)^2 / dtma + phi / dtf, valid for phi < 0.37
    phi = np.linspace(0, .37, 38)
    for dtma in (55.5, 47.5, 43.5):
        dt = 1 / ((1 - phi)**2 / dtma + phi / 189.)
        assert np.allclose(pp.raymer_porosity(dt, dtma, 189.), phi, atol=1e-10)


def test_raymer_porosity_hand_value():
    # phi = 0.2 sandstone: 0.64/55.5 + 0.2/189 = 0.011531532 + 0.001058201 = 0.012589733 -> dt = 79.4298
    assert np.isclose(pp.raymer_porosity(79.4298), .2, atol=1e-5)
    assert pp.raymer_porosity(55.5) == 0
    dt = np.linspace(55.5, 150, 50)
    assert np.all(np.diff(pp.raymer_porosity(dt)) > 0)
    s = pp.raymer_porosity(pd.Series([79.4298, np.nan], index=[8, 9]))
    assert list(s.index) == [8, 9] and np.isnan(s[9])


def test_effective_porosity_hand_values():
    # 0.25 - 0.4 * 0.18 = 0.178 ; floored at 0 ; NaN propagates
    assert np.isclose(pp.effective_porosity(.25, .4, .18), .178)
    assert pp.effective_porosity(.05, .8, .18) == 0
    e = pp.effective_porosity(pd.Series([.25, np.nan, .25], index=[1, 2, 3]), np.array([.4, .4, np.nan]), .18)
    assert list(e.index) == [1, 2, 3] and np.isclose(e[1], .178) and e[2:].isna().all()


# ---------------------------------------------------------------- minerals table

#[SLB] Appendix B (rho_log, phi_CNL, Pe and U, rounded as printed) and Appendix C / Por-1 (matrix dt)
SLB_APPENDIX = {
    #            RHOB  NPHI(CNL)  DT     Pe    U
    'Quartz':    (2.64, -.02,     55.5,  1.8,  4.8),     # Por-1 sandstone 18,000 ft/s = 55.5 us/ft
    'Calcite':   (2.71,  .00,     47.6,  5.1, 13.8),     # Appendix C limestone
    'Dolomite':  (2.85,  .01,     43.5,  3.1,  9.0),     # Appendix C dolomite
    'Anhydrite': (2.98, -.02,     50.0,  5.1, 15.),
    'Halite':    (2.04, -.03,     66.6,  4.7,  9.5),
    'Gypsum':    (2.35,  .60,     52.6,  4.0,  9.4),     # CNL printed as "60+"
}


def test_minerals_table_against_published_values():
    mm = pp.minerals()
    for name, (rho, nphi, dt, pe, _) in SLB_APPENDIX.items():
        r = mm.loc[name]
        assert abs(r['RHOB'] - rho) <= .021, name      # grain vs log density: quartz 2.65/2.64, dolomite 2.87/2.85
        assert abs(r['NPHI'] - nphi) <= .001, name     # CNL column (dolomite: SNP 0.02, CNL 0.01)
        assert abs(r['DT'] - dt) <= .61, name         # gypsum 52.0 vs 52.6, halite 67.0 vs 66.6
        assert abs(r['PEF'] - pe) <= .051, name
    # [KGS] Pe to two decimals
    assert (mm.loc['Quartz', 'PEF'], mm.loc['Dolomite', 'PEF'], mm.loc['Calcite', 'PEF']) == (1.81, 3.14, 5.08)
    # fresh water: rho 1, NPHI 1 by definition, dt 189 = 1e6 / 5,300 ft/s [SLB Por-1]
    assert tuple(mm.loc['Water', ['RHOB', 'NPHI']]) == (1., 1.)
    assert abs(mm.loc['Water', 'DT'] - 1e6 / 5300) < .5


def test_minerals_table_u_matches_published_u():
    # U = Pe * rho_e, rho_e = (rho_log + 0.1883) / 1.0704 ; e.g. calcite 5.08 * 2.8983 / 1.0704 = 13.755
    mm = pp.minerals()
    for name, (*_, u) in SLB_APPENDIX.items():
        u_table = mm.loc[name, 'PEF'] * (mm.loc[name, 'RHOB'] + .1883) / 1.0704
        assert np.isclose(pp.umaa(mm.loc[name, 'PEF'], mm.loc[name, 'RHOB'], 0.), u_table)
        assert abs(u_table / u - 1) < .025, name        # published U printed to 2-3 figures
    # fresh water: 0.36 * 1.1883 / 1.0704 = 0.3997, SLB Lith-5 Uf = 0.398
    assert abs(pp.umaa(mm.loc['Water', 'PEF'], 1., 0.) - .398) < .005


# ---------------------------------------------------------------- M-N, MID and Umaa

def test_m_n_hand_values():
    # [SLB] Appendix E: M = 0.01 (tf - t) / (rhob - rhof), N = (phiNf - phiN) / (rhob - rhof)
    # quartz  2.65/-0.02/55.5: M = 1.335/1.65 = 0.809091  N = 1.02/1.65 = 0.618182
    # calcite 2.71/0/47.5:     M = 1.415/1.71 = 0.827485  N = 1/1.71    = 0.584795
    # dolomite 2.87/.02/43.5:  M = 1.455/1.87 = 0.778075  N = 0.98/1.87 = 0.524064
    M, N = pp.m_n(np.array([2.65, 2.71, 2.87]), np.array([-.02, 0., .02]), np.array([55.5, 47.5, 43.5]))
    assert np.allclose(M, [.809091, .827485, .778075], atol=1e-6)
    assert np.allclose(N, [.618182, .584795, .524064], atol=1e-6)


def test_m_n_independent_of_water_porosity():
    # Burke et al. (1969): for liquid-filled porosity M and N are the matrix values at any porosity
    mm = pp.minerals()
    for name in ('Quartz', 'Calcite', 'Dolomite', 'Anhydrite'):
        r, w = mm.loc[name], mm.loc['Water']
        M0, N0 = pp.m_n(r['RHOB'], r['NPHI'], r['DT'])
        for phi in (.05, .15, .3):
            mix = (1 - phi) * r + phi * w
            M, N = pp.m_n(mix['RHOB'], mix['NPHI'], mix['DT'])
            assert np.isclose(M, M0) and np.isclose(N, N0), (name, phi)
    s = pp.m_n(pd.Series([2.4], index=[4]), np.array([.2]), np.array([80.]))
    assert all(list(x.index) == [4] for x in s)


def test_rhomaa_slb_lith9_example_and_inverse():
    # [SLB] Lith-9: phi 20 %, rhob 2.4 -> rhomaa 2.75 ; (2.4 - 0.2) / 0.8 = 2.75
    assert np.isclose(pp.rhomaa(2.4, .2), 2.75)
    phi = np.linspace(0, .35, 8)
    assert np.allclose(pp.rhomaa(phi * 1.1 + (1 - phi) * 2.87, phi, 1.1), 2.87)


def test_dtmaa_raymer_and_time_average_forms():
    # [SLB] Lith-9 example (dt 82, phi 30 %) reads tmaa = 46 us/ft. That chart follows the [RHG]
    # field-observation transform: tma = (1 - phi)^2 / (1/dt - phi/tf) = 0.49 / 0.010608 = 46.19
    assert np.isclose(pp.dtmaa(82., .3), .49 / (1 / 82 - .3 / 189))
    assert round(float(pp.dtmaa(82., .3))) == 46
    # Raymer inverse: 1/dt = (1 - phi)^2 / tma + phi / tf -> tma recovered
    phi = np.linspace(0, .35, 8)
    assert np.allclose(pp.dtmaa(1 / ((1 - phi)**2 / 47.5 + phi / 189.), phi), 47.5)
    # Wyllie inverse: dt = phi tf + (1 - phi) tma -> tma recovered; (82 - 0.3 * 189) / 0.7 = 36.14
    assert np.allclose(pp.dtmaa(phi * 189. + (1 - phi) * 47.5, phi, method='wyllie'), 47.5)
    assert np.isclose(pp.dtmaa(82., .3, method='wyllie'), (82 - .3 * 189) / .7)
    # 1/DT below PHIT/tf (DT above 189 / 0.3 = 630 us/ft): no matrix fits, NaN
    assert np.isnan(pp.dtmaa(700., .3)) and np.isfinite(pp.dtmaa(600., .3))


def test_umaa_slb_lith5_example():
    # [SLB] Lith-5: Pe 4.0, rhob 2.5, phi 25 %, fresh water -> Umaa 13 (chart read)
    # rho_e = 2.6883 / 1.0704 = 2.511491 ; U = 10.045964 ; (10.045964 - 0.0995) / 0.75 = 13.261952
    u = pp.umaa(4.0, 2.5, .25)
    assert np.isclose(u, 13.261952, atol=1e-5)
    assert abs(u - 13) < .5
    s = pp.umaa(pd.Series([4.0], index=['x']), 2.5, .25)
    assert list(s.index) == ['x']


# ---------------------------------------------------------------- multi-mineral solver

def _table_components(names):
    # RHOB, NPHI, DT from the table plus the volumetric U = Pe * rho_e
    mm = pp.minerals()
    comps = {m: mm.loc[m, ['RHOB', 'NPHI', 'DT']].to_dict() for m in names}
    for m in names:
        comps[m]['U'] = mm.loc[m, 'PEF'] * (mm.loc[m, 'RHOB'] + .1883) / 1.0704
    return comps


def _synthetic_logs(comps, names, logs, V, index=None):
    A = np.array([[comps[m][l] for m in names] for l in logs])
    return pd.DataFrame(V @ A.T, columns=logs, index=index)


def test_multimineral_recovers_volumes_from_table_logs():
    names = ['Quartz', 'Calcite', 'Dolomite', 'Water']
    comps = _table_components(names)
    V = np.random.default_rng(3).dirichlet([2, 2, 2, 1.5], 200)
    df = _synthetic_logs(comps, names, ['RHOB', 'NPHI', 'DT'], V, index=np.arange(200) * .5 + 1000)
    out = pp.multimineral(df, comps, logs=['RHOB', 'NPHI', 'DT'])
    assert out.index.equals(df.index)
    assert np.allclose(out[names].to_numpy(), V, atol=1e-6)
    assert out['MM_ERROR'].max() < 1e-6


def test_multimineral_five_components_with_u():
    # U mixes linearly by volume (Pe does not), so RHOB, NPHI, DT, U + unity resolve 5 components
    names = ['Quartz', 'Calcite', 'Dolomite', 'Anhydrite', 'Water']
    comps = _table_components(names)
    V = np.random.default_rng(4).dirichlet([2, 2, 2, 1, 1.5], 100)
    logs = ['RHOB', 'NPHI', 'DT', 'U']
    out = pp.multimineral(_synthetic_logs(comps, names, logs, V), comps, logs=logs)
    assert np.allclose(out[names].to_numpy(), V, atol=1e-6)


def test_multimineral_with_pef_solves_on_u():
    # true volumes; RHOB, NPHI, DT and U mix by volume, and the logged PEF is U / rho_e of the mixture
    names = ['Quartz', 'Calcite', 'Dolomite', 'Water']
    mm = pp.minerals()
    comps = {m: mm.loc[m, ['RHOB', 'NPHI', 'DT', 'PEF']].to_dict() for m in names}
    V = np.array([[.4, .2, .2, .2], [.6, .1, .1, .2], [.1, .5, .3, .1]])
    u = {m: comps[m]['PEF'] * (comps[m]['RHOB'] + .1883) / 1.0704 for m in names}
    df = pd.DataFrame(V @ np.array([[comps[m][l] for l in ('RHOB', 'NPHI', 'DT')] for m in names]),
                      columns=['RHOB', 'NPHI', 'DT'])
    df['PEF'] = V @ np.array([u[m] for m in names]) / ((df['RHOB'] + .1883) / 1.0704)
    out = pp.multimineral(df, comps, logs=['RHOB', 'NPHI', 'DT', 'PEF'])
    assert np.allclose(out[names].to_numpy(), V, atol=1e-6)
    # PEF without RHOB cannot be converted
    with pytest.raises(ValueError):
        pp.multimineral(df, {m: {k: v for k, v in comps[m].items() if k != 'RHOB'} for m in names}, logs=['NPHI', 'DT', 'PEF'])


def test_multimineral_bounds_and_missing_logs():
    names = ['Quartz', 'Calcite', 'Water']
    comps = _table_components(names)
    V = np.array([[.7, .1, .2], [.2, .6, .2], [.5, .3, .2], [.5, .3, .2]])
    df = _synthetic_logs(comps, names, ['RHOB', 'NPHI', 'DT'], V)
    df.loc[1, 'DT'] = np.nan                         # 2 logs + unity still fix 3 components
    df.loc[2, ['RHOB', 'DT']] = np.nan               # 1 log: too few -> NaN
    df.loc[3, :] = np.nan
    out = pp.multimineral(df, comps, logs=['RHOB', 'NPHI', 'DT'])
    assert np.allclose(out.loc[:1, names].to_numpy(), V[:2], atol=1e-6)
    assert out.loc[2:].isna().all().all()
    # noisy logs: volumes stay in [0, 1] and close to unit sum
    rng = np.random.default_rng(5)
    noisy = _synthetic_logs(comps, names, ['RHOB', 'NPHI', 'DT'], np.tile(V[:1], (300, 1)))
    noisy += rng.normal(0, 1, noisy.shape) * np.array([.025, .015, 2.])
    vol = pp.multimineral(noisy, comps)[names]
    assert ((vol >= 0) & (vol <= 1)).all().all()
    assert (vol.sum(axis=1) - 1).abs().max() < .05


def test_multimineral_slb_lith6_example():
    # [SLB] Lith-6: rhomaa 2.74 g/cc and Umaa 13 -> about 80 % calcite, 20 % dolomite (chart read)
    # density alone: (2.74 - 2.71) / (2.87 - 2.71) = 0.19 dolomite ; Umaa alone: (13.755 - 13) / (13.755 - 8.972) = 0.16
    mm = pp.minerals()
    comps = {m: {'RHOMAA': mm.loc[m, 'RHOB'], 'UMAA': pp.umaa(mm.loc[m, 'PEF'], mm.loc[m, 'RHOB'], 0.)}
             for m in ('Calcite', 'Dolomite')}
    out = pp.multimineral(pd.DataFrame({'RHOMAA': [2.74], 'UMAA': [13.]}), comps,
                          uncertainties={'RHOMAA': .025, 'UMAA': .5})
    assert abs(out.loc[0, 'Dolomite'] - .2) < .05 and abs(out.loc[0, 'Calcite'] - .8) < .05
