'''
Reference tests for geomechanics, log cleaning, facies and statistics.

Expected values are computed here from the published equations (or by hand, with the arithmetic in a
comment), never from petrolib itself. Sources:

[Castagna]  Castagna, Batzle & Eastwood (1985), Geophysics 50(4) 571-581: Vp = 1.16 Vs + 1.36 (km/s),
            https://en.wikipedia.org/wiki/Mudrock_line
[Chang]     Chang, Zoback & Khaksar (2006), J. Pet. Sci. Eng. 51, 223-237, Table 1 eq. (2), McNally (1987):
            UCS (MPa) = 1200 exp(-0.036 dt), dt in us/ft,
            https://pangea.stanford.edu/departments/geophysics/dropbox/STRESS/publications/MDZ%20PDF's/2006/2006_Empirical_relations_between_rock_strenght.pdf
[ResFrac]   McClure et al., ResFrac technical writeup, arXiv:1804.02092, eqs. 23-4, 23-7, 23-8 (dynamic PR,
            G = rho Vs^2, E = 2G(1+PR)) and 23-12 (Eaton 1969: Shmin = PR/(1-PR) (Sv - Pp) + Pp)
[Agile]     Agile Scientific, "Which brittleness index?" (2013): Rickman et al. (2008) written out as
            B = 7.14 E - 200 PR + 72.9, i.e. E normalized on 1-8 and PR on 0.15-0.40,
            https://agilescientific.com/blog/2013/12/3/which-brittleness-index.html
[Eaton75]   Eaton (1975) SPE 5544. Sonic form Gp = Go - (Go - Gn)(dt_n/dt_o)^x, x = 3 (Gulf of Mexico):
            US patent 5,128,866, https://patents.google.com/patent/US5128866A/en ; exponents 3 (P-wave)
            and 1.2 (resistivity): Ebrom, Albertin & Heppard (2007),
            https://www.searchanddiscovery.com/abstracts/html/2007/annual/abstracts/lbEbrom.htm
[Gassmann]  Gassmann (1951) in modern notation, https://en.wikipedia.org/wiki/Gassmann%27s_equation
[Chatterjee] Chatterjee (2021), JASA 116(536) 2009-2022, arXiv:1909.10140, eq. (1.1) and the tie-corrected
            form; reference implementation scipy.stats.chatterjeexi
[Mud]       P (psi) = 0.052 MW (ppg) TVD (ft); fresh water 8.33 ppg = 0.433 psi/ft,
            https://en.wikipedia.org/wiki/Pore_pressure_gradient
'''
import math

import numpy as np
import pandas as pd
import pytest

from petrolib import petrophysics as pp

G = 9.80665                       #standard gravity, m/s2
PSI_PER_MPA = 1e6 / 6894.757293   #1 psi = 6894.757 Pa
MPSI_TO_GPA = 6.894757293


# ---------------------------------------------------------------------------------------------
# velocities and elastic moduli
# ---------------------------------------------------------------------------------------------

def test_velocity_from_slowness():
    # 100 us/ft -> 1e6/100 = 10000 ft/s -> 10000 * 0.3048 = 3048 m/s
    assert np.isclose(pp.velocity(100.), 3048.)
    # 400 us/m -> 1e6/400 = 2500 m/s
    assert np.isclose(pp.velocity(400., unit='us/m'), 2500.)
    dt = pd.Series([50., 100., np.nan], index=[1000.5, 1001., 1001.5])
    v = pp.velocity(dt)
    assert isinstance(v, pd.Series) and v.index.equals(dt.index) and np.isnan(v.iloc[2])
    assert np.allclose(v.iloc[:2] / .3048, 1e6 / dt.iloc[:2])                     #ft/s = 1e6 / dt


def test_castagna_mudrock_line():
    # Vp = 1.16 Vs + 1.36 km/s -> Vs = (Vp - 1.36) / 1.16; Vp = 3 km/s: Vs = 1.64 / 1.16 = 1.41379 km/s
    vp = pd.Series([2000., 3000., 4500.], index=[5, 6, 7])
    vs = pp.vs_castagna(vp)
    assert vs.index.equals(vp.index)
    assert np.allclose(vs, (vp / 1000 - 1.36) / 1.16 * 1000, atol=.5)               #0.8621, 1.1724 are rounded
    assert np.isclose(pp.vs_castagna(3000.), 1413.79, atol=.2)


def test_elastic_moduli_hand_calculation():
    # rho 2.65 g/cc, Vp 4000, Vs 2400 m/s (76.2 and 127 us/ft). With g/cc and m/s, 1e-6 gives GPa:
    # G = 2.65 * 2400^2 * 1e-6 = 15.264 ; K = 2.65 * (4000^2 - 4/3 2400^2) * 1e-6 = 2.65 * 8.32 = 22.048
    # PR = (16 - 11.52) / (2 (16 - 5.76)) = 4.48 / 20.48 = 0.21875 ; E = 2 * 15.264 * 1.21875 = 37.206
    m = pp.elastic_moduli(pd.Series([2.65]), pd.Series([304800 / 4000]), pd.Series([304800 / 2400])).iloc[0]
    assert np.isclose(m['VP'], 4000) and np.isclose(m['VS'], 2400) and np.isclose(m['VPVS'], 4000 / 2400)
    assert np.isclose(m['G'], 15.264) and np.isclose(m['K'], 22.048)
    assert np.isclose(m['PR'], .21875) and np.isclose(m['E'], 37.206)
    assert np.isclose(m['AI'], 4000 * 2.65)                                              #m/s g/cc
    # same rock, slowness in us/m
    mm = pp.elastic_moduli(2.65, 1e6 / 4000, 1e6 / 2400, unit='us/m').iloc[0]
    assert np.allclose(mm[['G', 'K', 'E', 'PR']].astype(float), m[['G', 'K', 'E', 'PR']].astype(float))


def test_elastic_identities():
    rng = np.random.default_rng(3)
    rho = rng.uniform(2., 2.9, 200)
    vp = rng.uniform(2000., 6000., 200)
    vs = vp / rng.uniform(1.5, 2.4, 200)
    m = pp.elastic_moduli(rho, 304800 / vp, 304800 / vs)
    E, G_, K, nu = m['E'], m['G'], m['K'], m['PR']
    assert np.allclose(E, 2 * G_ * (1 + nu))
    assert np.allclose(K, E / (3 * (1 - 2 * nu)))
    assert np.allclose(E, 9 * K * G_ / (3 * K + G_))
    # E = rho Vs^2 (3 Vp^2 - 4 Vs^2) / (Vp^2 - Vs^2)
    assert np.allclose(E, rho * vs**2 * (3 * vp**2 - 4 * vs**2) / (vp**2 - vs**2) * 1e-6)
    # PR from Vp/Vs, ResFrac eq. 23-4
    r = vp / vs
    assert np.allclose(nu, (.5 * r**2 - 1) / (r**2 - 1))


def test_elastic_moduli_index_and_nan():
    idx = pd.Index([2000., 2000.5, 2001.], name='DEPTH')
    rho = pd.Series([2.4, np.nan, 2.5], index=idx)
    dtc = pd.Series([90., 90., 80.], index=idx)
    dts = pd.Series([160., 160., np.nan], index=idx)
    m = pp.elastic_moduli(rho, dtc, dts)
    assert m.index.equals(idx)
    assert np.isnan(m.loc[2000.5, 'G']) and np.isfinite(m.loc[2000.5, 'VP'])
    assert np.isnan(m.loc[2001., 'PR']) and np.isfinite(m.loc[2001., 'VP'])
    # index taken from whichever input is a Series
    assert pp.elastic_moduli(rho.to_numpy(), dtc, dts).index.equals(idx)


# ---------------------------------------------------------------------------------------------
# brittleness and strength
# ---------------------------------------------------------------------------------------------

def test_brittleness_rickman():
    # Rickman ranges (E 1-8, PR 0.15-0.40) written out by [Agile] as B = 50/7 E - 200 PR + (80 - 50/7)
    # E = 4.5, PR = 0.25: 50 (3.5/7 + 0.15/0.25) = 50 (0.5 + 0.6) = 55
    e = pd.Series([4.5, 2., 7., 6.], index=list('abcd'))
    nu = pd.Series([.25, .35, .18, .2], index=list('abcd'))
    bi = pp.brittleness(e, nu, e_range=(1, 8), pr_range=(.15, .40))
    assert bi.index.equals(e.index) and np.isclose(bi['a'], 55.)
    assert np.allclose(bi, 50 / 7 * e - 200 * nu + 80 - 50 / 7)
    # out of range: clipped to [0, 100]; NaN kept
    out = pp.brittleness(np.array([10., .5, np.nan]), np.array([.1, .45, .2]), e_range=(1, 8), pr_range=(.15, .4))
    assert out[0] == 100 and out[1] == 0 and np.isnan(out[2])
    # only the units must agree: E in GPa with ranges in GPa gives the same index as Mpsi
    gpa = pp.brittleness(e * MPSI_TO_GPA, nu, e_range=(MPSI_TO_GPA, 8 * MPSI_TO_GPA), pr_range=(.15, .4))
    assert np.allclose(gpa, bi)


def test_brittleness_default_normalization():
    rng = np.random.default_rng(0)
    e, nu = rng.uniform(10, 60, 500), rng.uniform(.1, .4, 500)
    e_lo, e_hi = np.percentile(e, [1, 99])
    p_lo, p_hi = np.percentile(nu, [1, 99])
    expected = np.clip(50 * ((e - e_lo) / (e_hi - e_lo) + (p_hi - nu) / (p_hi - p_lo)), 0, 100)
    assert np.allclose(pp.brittleness(e, nu), expected)


def test_ucs_mcnally():
    # [Chang] eq. (2): UCS = 1200 exp(-0.036 dt). dt = 100: 1200 exp(-3.6) = 1200 * 0.0273237 = 32.788 MPa
    assert np.isclose(pp.ucs_mcnally(100.), 32.788, atol=1e-3)
    dt = pd.Series([60., 100., np.nan], index=[3, 4, 5])
    ucs = pp.ucs_mcnally(dt)
    assert ucs.index.equals(dt.index) and np.isnan(ucs[5])
    assert np.isclose(ucs[3], 1200 * math.exp(-.036 * 60))                               #138.39 MPa


# ---------------------------------------------------------------------------------------------
# stresses and pressures
# ---------------------------------------------------------------------------------------------

def test_hydrostatic_pressure():
    # fresh water: 1000 kg/m3 * 9.80665 * 3048 m = 29.891 MPa at 10000 ft = 4335.3 psi (0.4335 psi/ft)
    p = pp.hydrostatic_pressure(10000., rho_water=1.0, unit='psi', depth_unit='ft')
    assert np.isclose(p, 1000 * G * 3048 / 1e6 * PSI_PER_MPA)
    assert np.isclose(p / 10000, .433, rtol=2e-3)                                         #[Mud] 0.433 psi/ft
    # sea water 1.03 g/cc at 1000 m: 1030 * 9.80665 * 1000 / 1e6 = 10.1008 MPa
    z = pd.Series([1000., 2000.], index=['a', 'b'])
    ph = pp.hydrostatic_pressure(z)
    assert ph.index.equals(z.index) and np.isclose(ph['a'], 10.1008, atol=1e-4)


def test_overburden_constant_density_is_rho_g_z():
    z = np.arange(0., 4000.1, .1524)
    for rho in (1.0, 2.3):
        sv = pp.overburden(z, np.full(len(z), rho))
        assert np.allclose(sv, rho * 1000 * G * z / 1e6)
    # in field units: 1 g/cc over 10000 ft = 0.4335 psi/ft
    zf = np.arange(0., 10000.1, .5)
    svf = pp.overburden(zf, np.ones(len(zf)), unit='psi', depth_unit='ft')
    assert np.isclose(svf[-1], 1000 * G * 3048 / 1e6 * PSI_PER_MPA)


def test_overburden_integrates_a_density_profile():
    # rho(z) = 2.0 + 2e-4 z g/cc from the surface: Sv = g (2.0 z + 1e-4 z^2) * 1000 / 1e6 MPa
    z = np.arange(0., 3000.01, .1524)
    sv = pp.overburden(z, 2.0 + 2e-4 * z)
    exact = G * (2.0 * z + 1e-4 * z**2) * 1e-3
    assert np.allclose(sv[1:], exact[1:], rtol=1e-4)


def test_overburden_unlogged_top_water_and_nan():
    # logs from 1000 m in 400 m of water; 600 m of unlogged sediment at rho_top = 1.9:
    # Sv(1000) = 9.80665e-3 (1.03 * 400 + 1.9 * 600) = 9.80665e-3 * 1552 = 15.2199 MPa
    z = pd.Series([1000., 1010., 1020.], index=[7, 8, 9])
    rhob = pd.Series([2.2, np.nan, 2.4], index=[7, 8, 9])
    sv = pp.overburden(z, rhob, rho_top=1.9, water_depth=400.)
    assert sv.index.equals(z.index) and np.isclose(sv[7], 15.2199, atol=1e-4)
    # missing RHOB is replaced by rho_top: + 1.9 * 10 * 9.80665e-3, then + 2.4 * 10 * 9.80665e-3
    assert np.isclose(sv[8] - sv[7], .186326, atol=1e-6) and np.isclose(sv[9] - sv[8], .23536, atol=1e-5)


def test_normal_compaction_trend():
    z = np.linspace(500., 3000., 120)
    dt = 10**(2.3 - 1.2e-4 * z)
    over = dt.copy()
    over[z > 2500] *= 1.3                                                              #overpressured: not fitted
    trend = pp.normal_compaction_trend(pd.Series(z), pd.Series(over), fit=z <= 2500)
    assert np.allclose(trend, dt)


def test_eaton_pore_pressure():
    # Sv = 50, Ph = 20 MPa. Sonic dt_n = 100, dt_o = 125: Pp = 50 - 30 (100/125)^3 = 50 - 30 * 0.512 = 34.64
    assert np.isclose(pp.pore_pressure_eaton(50., 20., 125., 100.), 34.64)
    # resistivity R_o = 0.8, R_n = 1.6: Pp = 50 - 30 * 0.5^1.2 = 50 - 30 * 0.435275 = 36.9418
    assert np.isclose(pp.pore_pressure_eaton(50., 20., .8, 1.6, method='resistivity'), 36.9418, atol=1e-4)
    # on the normal trend: hydrostatic, for both logs
    sv, ph = pd.Series([40., 60.], index=[1, 2]), pd.Series([18., 27.], index=[1, 2])
    for method in ('sonic', 'resistivity'):
        out = pp.pore_pressure_eaton(sv, ph, pd.Series([3., 4.], index=[1, 2]), np.array([3., 4.]), method=method)
        assert out.index.equals(sv.index) and np.allclose(out, ph)
    # undercompaction (slower sonic, lower resistivity) gives overpressure, bounded by Sv
    assert 20 < pp.pore_pressure_eaton(50., 20., 140., 100.) < 50
    assert 20 < pp.pore_pressure_eaton(50., 20., .5, 1., method='resistivity') < 50
    # Eaton wrote the equation in gradients; multiplying through by depth gives the same pressure
    d, sv_g, ph_g = 3000., 1.0, .465                                                    #psi/ft
    grad = sv_g - (sv_g - ph_g) * (100 / 125)**3
    assert np.isclose(pp.pore_pressure_eaton(sv_g * d, ph_g * d, 125., 100.), grad * d)
    assert np.isnan(pp.pore_pressure_eaton(50., 20., np.nan, 100.))


def test_eaton_fracture_pressure():
    # [ResFrac] 23-12: PR = 0.25 -> 0.25/0.75 = 1/3; Sv 50, Pp 20: 30 / 3 + 20 = 30 MPa
    assert np.isclose(pp.fracture_pressure_eaton(50., 20., .25), 30.)
    # limits: PR = 0 gives Pp, PR = 0.5 gives Sv
    assert np.isclose(pp.fracture_pressure_eaton(50., 20., 0.), 20.)
    assert np.isclose(pp.fracture_pressure_eaton(50., 20., .5), 50.)
    pr = pd.Series([.2, .3], index=['x', 'y'])
    assert pp.fracture_pressure_eaton(np.array([50., 60.]), np.array([20., 25.]), pr).index.equals(pr.index)


def test_gradients_and_mud_weight():
    # 9.80665 MPa at 1000 m = 1 g/cc = 9.80665 MPa/km = 0.43352 psi/ft
    p, z = pd.Series([9.80665]), pd.Series([1000.])
    assert np.isclose(pp.to_gradient(p, z)[0], 1.)
    assert np.isclose(pp.to_gradient(p, z, unit='MPa/km')[0], 9.80665)
    psi_ft = pp.to_gradient(p, z, unit='psi/ft')[0]
    assert np.isclose(psi_ft, 9.80665 * PSI_PER_MPA / (1000 / .3048))
    # [Mud] ppg = (psi/ft) / 0.052 ; 1 g/cc = 8.345 ppg (453.59237 g/lb, 3785.41 cm3/gal)
    assert np.isclose(psi_ft / .052, 3785.411784 / 453.59237, rtol=2e-3)
    # depth in ft gives the same gradient
    assert np.isclose(pp.to_gradient(9.80665, 1000 / .3048, depth_unit='ft'), 1.)
    with pytest.raises(ValueError):
        pp.to_gradient(p, z, unit='ppg')


# ---------------------------------------------------------------------------------------------
# fluid substitution
# ---------------------------------------------------------------------------------------------

def gassmann_forward(k_dry, km, kf, phi):
    '''[Gassmann], written out independently of petrolib'''
    return k_dry + (1 - k_dry / km)**2 / (phi * (1 / kf - 1 / km) + (1 - k_dry / km) / km)


def test_wood_fluid_modulus():
    # Sw = 0.5, Kw 2.8, Kg 0.05: 1 / (0.5/2.8 + 0.5/0.05) = 1 / 10.178571 = 0.098246 GPa
    assert np.isclose(pp.wood_fluid_modulus(.5, 2.8, .05), .098246, atol=1e-6)
    sw = pd.Series([0., .3, 1.], index=[4, 5, 6])
    k = pp.wood_fluid_modulus(sw, 2.25, 1.)
    assert k.index.equals(sw.index) and np.isclose(k[4], 1.) and np.isclose(k[6], 2.25)
    assert np.all(k <= sw * 2.25 + (1 - sw) * 1. + 1e-12)                              #Reuss <= Voigt


def test_gassmann_hand_example():
    # K_dry 12, Km 36, Kf 2.2, phi 0.2: (1 - 1/3)^2 = 0.44444 ;
    # 0.2/2.2 + 0.8/36 - 12/1296 = 0.090909 + 0.022222 - 0.009259 = 0.103872 ; K_sat = 12 + 4.2788 = 16.2788
    k_sat = gassmann_forward(12., 36., 2.2, .2)
    assert np.isclose(k_sat, 16.2788, atol=1e-4)
    # from brine to a zero-modulus fluid returns the dry modulus
    assert np.isclose(pp.gassmann(k_sat, 36., 2.2, 0., .2), 12.)
    # K_dry from Gassmann's relation K_sat/(Km-K_sat) - Kf/(phi(Km-Kf)) = K_dry/(Km-K_dry)
    # with K_sat = 16, Kf = 2.2: 0.8 - 2.2/6.76 = 0.474556 -> K_dry = 36 * 0.474556 / 1.474556 = 11.5859
    assert np.isclose(pp.gassmann(16., 36., 2.2, 0., .2), 11.5859, atol=1e-4)


def test_gassmann_properties():
    rng = np.random.default_rng(7)
    n = 300
    km = 36.6
    phi = rng.uniform(.05, .35, n)
    k_dry = km * (1 - phi / .4) * rng.uniform(.3, .9, n)                              #within 0 and the Voigt bound
    kf1, kf2 = rng.uniform(.02, .2, n), rng.uniform(1., 3., n)
    k1 = gassmann_forward(k_dry, km, kf1, phi)
    k2 = pp.gassmann(pd.Series(k1), km, kf1, kf2, pd.Series(phi))
    assert isinstance(k2, pd.Series)
    assert np.allclose(k2, gassmann_forward(k_dry, km, kf2, phi))                       #substitution
    assert np.allclose(pp.gassmann(k1, km, kf1, 0., phi), k_dry)                       #dry rock back
    assert np.allclose(pp.gassmann(k1, km, kf1, kf1, phi), k1)                         #same fluid: unchanged
    assert np.allclose(pp.gassmann(k1, km, kf1, km, phi), km)                          #fluid = mineral: Km
    # bounds: K_dry <= K_sat, Reuss <= K_sat <= Voigt, and K_sat rises with the fluid modulus
    voigt = (1 - phi) * km + phi * kf2
    reuss = 1 / (phi / kf2 + (1 - phi) / km)
    assert np.all(k2 >= k_dry) and np.all(k2 <= voigt + 1e-9) and np.all(k2 >= reuss - 1e-9)
    assert np.all(k2 > k1)
    # Biot-Gassmann form: K_sat = K_dry + beta^2 M, beta = 1 - K_dry/Km, 1/M = phi/Kf + (beta - phi)/Km
    beta = 1 - k_dry / km
    assert np.allclose(k2, k_dry + beta**2 / (phi / kf2 + (beta - phi) / km))


# ---------------------------------------------------------------------------------------------
# cleaning and facies
# ---------------------------------------------------------------------------------------------

def _raw_logs():
    return pd.DataFrame({'DEPTH': [1000., 1000.5, 1001., 1001.5, 1002.],
                         'GR': [-5., 60., 180., -999.25, np.nan],
                         'RT': [.1, 5., 3000., -999., 2.],
                         'NPHI': [-.2, .25, .6, .1, -999.25],
                         'RHOB': [1.5, 2.4, 3.1, 2.5, -999.25],
                         'DT': [30., 90., 210., -999.25, 160.]})


def test_process_data_clips_and_nulls():
    raw = _raw_logs()
    before = raw.copy()
    out = pp_process(raw, 'both')
    pd.testing.assert_frame_equal(raw, before)                                           #input unchanged
    assert out['GR'].tolist()[:3] == [0., 60., 150.] and out['GR'].iloc[3:].isna().all()
    assert out['RT'].tolist()[:3] == [.2, 5., 2000.] and np.isnan(out['RT'].iloc[3])
    assert out['NPHI'].tolist()[:4] == [-.15, .25, .45, .1] and np.isnan(out['NPHI'].iloc[4])
    assert out['RHOB'].tolist()[:4] == [1.95, 2.4, 2.95, 2.5] and np.isnan(out['RHOB'].iloc[4])
    assert out['DT'].tolist()[:3] == [40., 90., 200.] and np.isnan(out['DT'].iloc[3])
    assert out['DEPTH'].equals(raw['DEPTH'])
    hi = pp_process(raw, 'max')
    assert hi['GR'].iloc[0] == -5. and hi['GR'].iloc[2] == 150. and hi['RHOB'].iloc[0] == 1.5
    lo = pp_process(raw, 'min')
    assert lo['GR'].iloc[0] == 0. and lo['GR'].iloc[2] == 180. and lo['RHOB'].iloc[2] == 3.1


def pp_process(df, trim):
    from petrolib.processing import process_data
    return process_data(df, 'GR', 'RT', 'NPHI', 'RHOB', dt='DT', trim=trim)


def test_trim():
    from petrolib.processing import trim
    df = pd.DataFrame({'GR': [-10., 50., 250., np.nan]}, index=[10, 11, 12, 13])
    out = trim(df.copy(), 'GR', lower=0, upper=200)
    assert out['GR'].tolist()[:3] == [0., 50., 200.] and np.isnan(out['GR'].iloc[3])
    assert out.index.equals(df.index)
    with pytest.raises(AssertionError):
        trim(df, 'RT', 0, 1)


def test_set_alias():
    from petrolib.processing import set_alias
    df = pd.DataFrame({'GR_X': [1.], 'RES': [2.], 'NEU': [.2], 'DEN': [2.3]},
                      index=pd.Index([1000.], name='DEPT'))
    out = set_alias(df, GR='GR_X', RT='RES', NPHI='NEU', RHOB='DEN')
    assert {'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB'} <= set(out.columns) and out['DEPTH'].iloc[0] == 1000.


def test_model_facies_gr_cutoffs():
    from petrolib.processing import model_facies
    df = pd.DataFrame({'GR': [20., 74.9, 75., 120., np.nan]}, index=[5, 6, 7, 8, 9])
    ss = model_facies(df.copy(), 'GR', env='SS')['litho']
    assert ss.tolist()[:4] == ['Sandstone', 'Sandstone', 'Shale', 'Shale'] and pd.isna(ss[9])
    co = pd.DataFrame({'GR': [9., 12., 30., 50., 90., 3., np.nan]})
    lith = model_facies(co, 'GR', env='CO')['litho']
    assert lith.tolist()[:5] == ['Anhydrite', 'Limestone', 'Dolomite', 'Sandstone', 'Shale']
    assert lith.iloc[5:].isna().all()                                                    #below 8 API and missing


def _clusters(seed=0, n=60):
    '''Three well separated log clusters; GR means 20 (clean), 80, 140 (shale)'''
    rng = np.random.default_rng(seed)
    true = np.repeat([2, 0, 1], n)                                                       #not in GR order
    centre = {0: (20., 50., -.01, 2.62), 1: (80., 5., .2, 2.4), 2: (140., 1., .35, 2.55)}
    rows = [np.array(centre[t]) + rng.normal(0, [3., 0, .01, .01]) for t in true]
    X = np.array(rows)
    rt = X[:, 1] * 10**rng.normal(0, .03, len(true))
    df = pd.DataFrame({'GR': X[:, 0], 'RT': rt, 'NPHI': X[:, 2], 'RHOB': X[:, 3]},
                      index=pd.Index(1000. + .5 * np.arange(len(true)), name='DEPTH'))
    return df, true


def test_electrofacies_recovers_separated_clusters():
    from petrolib.processing import electrofacies
    df, true = _clusters()
    df.iloc[[0, 70, 150], 1] = np.nan                                                    #missing RT
    f = electrofacies(df, ['GR', 'RT', 'NPHI', 'RHOB'], n_facies=3, seed=1)
    assert f.index.equals(df.index)
    assert f.iloc[[0, 70, 150]].isna().all() and f.notna().sum() == len(df) - 3
    ok = f.notna().to_numpy()
    assert np.array_equal(f[ok].to_numpy().astype(int), true[ok])                         #numbered by mean GR
    # repeatable with a seed; invariant to the units of a curve (standardized) and to scaling RT (log10)
    again = electrofacies(df, ['GR', 'RT', 'NPHI', 'RHOB'], n_facies=3, seed=1)
    scaled = electrofacies(df.assign(NPHI=df['NPHI'] * 100, RT=df['RT'] * 10), ['GR', 'RT', 'NPHI', 'RHOB'],
                           n_facies=3, seed=1)
    assert f.equals(again) and f.equals(scaled)


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------

def test_chatterjee_hand_examples():
    from petrolib.stats import Correlation
    # no ties: y ranks in x order 1 3 2 5 4 -> sum |diff| = 2 + 1 + 3 + 1 = 7 ; xi = 1 - 3 * 7 / 24 = 0.125
    c = Correlation(pd.DataFrame({'x': [1., 2., 3., 4., 5.], 'y': [1., 3., 2., 5., 4.]}))
    assert np.isclose(c._chatterjee(c._df['x'], c._df['y']), .125)
    # ties in y: y = 1 1 2 2 ; r (= #y_j <= y_i) = 2 2 4 4, sum |diff| = 2 ; l (= #y_j >= y_i) = 4 4 2 2,
    # sum l (n - l) = 0 + 0 + 4 + 4 = 8 ; xi = 1 - 4 * 2 / (2 * 8) = 0.5
    t = Correlation(pd.DataFrame({'x': [1., 2., 3., 4.], 'y': [1., 1., 2., 2.]}))
    assert np.isclose(t._chatterjee(t._df['x'], t._df['y']), .5)


def test_chatterjee_matches_scipy():
    stats = pytest.importorskip('scipy.stats')
    if not hasattr(stats, 'chatterjeexi'):
        pytest.skip('scipy < 1.15')
    from petrolib.stats import Correlation
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    df = pd.DataFrame({'x': x, 'y': x**2 + rng.normal(0, .3, 300),
                       'yt': np.round(x**2 + rng.normal(0, .3, 300)),                     #ties in y
                       'xt': np.round(x * 2)})                                            #ties in x
    df.loc[:39, 'x'] = np.nan
    df.loc[250:, 'y'] = np.nan
    m = Correlation(df).corr('chatterjee')
    for xc, yc in [('x', 'y'), ('y', 'x'), ('x', 'yt'), ('xt', 'yt')]:
        ok = df[[xc, yc]].dropna()
        ref = stats.chatterjeexi(ok[xc].to_numpy(), ok[yc].to_numpy()).statistic
        assert np.isclose(m.loc[yc, xc], ref, atol=1e-6)                                  #row = Y, column = X
    # no ties: the simple form of eq. (1.1); xi(x, x) = (n - 2) / (n + 1)
    n = df['x'].notna().sum()
    assert np.isclose(m.loc['x', 'x'], (n - 2) / (n + 1), atol=1e-6)


def test_pearson_listwise_and_no_side_effect():
    from petrolib.stats import Correlation
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=list('abc'))
    df['b'] += df['a']
    df.loc[:9, 'c'] = np.nan
    c = Correlation(df)
    m = c.corr('pearsonr')
    assert np.allclose(m, df.dropna().corr(), atol=1e-6)                                   #complete rows (float32)
    assert len(c._df) == 100                                                              #data not trimmed
    assert np.isclose(m.loc['b', 'a'], np.corrcoef(df['a'][10:], df['b'][10:])[0, 1], atol=1e-6)
