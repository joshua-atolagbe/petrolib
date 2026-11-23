'''
Reference tests for petrolib.petrophysics saturation, fluids and uncertainty

Expected values are computed independently of petrolib: the published equation is written out in
the test, or the arithmetic is shown in a comment. Sources:

[A]  Archie (1942); resistivity form Rt = a Rw phi^-m Sw^-n, https://en.wikipedia.org/wiki/Archie%27s_law
[HK] Hari Kumar (2010), On the application of Simandoux and Indonesian shaly sand models, ONGC, eqs 1(b)-1(d), 2
     https://www.spec2000.net/freepubs/B1968-Simandoux-Indonesia-Saturation-Equation.pdf
[DM] Dacy and Martin (2006), SCA2006-29, eq 1 (Waxman-Smits-Thomas) and eq 5 (Juhasz 1981 B, T in degC)
     https://jgmaas.com/SCA/2006/SCA2006-29.pdf
[DW] Clavier, Coates and Dumanoir (1984) dual water, as quoted in US patent 7168310 B2 eq 17.b
     https://patents.google.com/patent/US7168310B2/en
[KGS] Doveton, Temperature and water resistivity, Kansas Geological Survey (Arps 6.77 degF / 21.5 degC,
     formation temperature and Rw worked example), https://www.kgs.ku.edu/PRS/Info/pdf/doveton3.PDF
[USGS] Jorgensen (1989), USGS Water-Supply Paper 2321, fig 5 (SP method after Bateman and Konen 1977)
     https://pubs.usgs.gov/wsp/2321/report.pdf
'''
import math

import numpy as np
import pandas as pd
import pytest

import petrolib.petrophysics as pp


#---------------------------------------------------------------- Archie

def archie_ref(phi, rt, rw, a=1., m=2., n=2.):
    #[A]: Rt = a Rw phi^-m Sw^-n  ->  Sw = (a Rw / (phi^m Rt))^(1/n)
    return (a * rw / (phi**m * rt))**(1 / n)


def test_archie_hand_calculation():
    #phi .2, m 2: F = 1/.04 = 25; Ro = 25 * .05 = 1.25; Sw = sqrt(1.25 / 20) = sqrt(.0625) = .25
    assert math.isclose(pp.archie(.2, 20., rw=.05), .25, rel_tol=1e-12)
    #Humble constants a .62, m 2.15
    assert math.isclose(pp.archie(.25, 10., rw=.04, a=.62, m=2.15), archie_ref(.25, 10., .04, .62, 2.15), rel_tol=1e-12)
    #n = 2.5: Sw = (1 * .05 / (.04 * 20))^(1/2.5) = .0625^.4 = .3299
    assert math.isclose(pp.archie(.2, 20., rw=.05, n=2.5), .0625**.4, rel_tol=1e-12)


def test_archie_grid_round_trip_and_edges():
    rng = np.random.default_rng(0)
    phi, sw = rng.uniform(.05, .35, 200), rng.uniform(.05, 1., 200)
    a, m, n, rw = .81, 1.9, 2.2, .045
    rt = a * rw / (phi**m * sw**n)                                                      # forward [A]
    assert np.allclose(pp.archie(phi, rt, rw=rw, a=a, m=m, n=n), sw, rtol=1e-12)
    out = pp.archie(np.array([.2, np.nan, 0., .2]), np.array([10., 10., 10., .01]), rw=.05)
    assert np.isnan(out[1])                                                             # NaN in, NaN out
    assert out[2] == 1 and out[3] == 1                                                  # no porosity; Sw > 1 clipped


#---------------------------------------------------------------- Simandoux

def test_simandoux_hand_calculation_n2():
    #[HK] 1(b): phi^m Sw^2 / ((1 - Vsh) a Rw) + Vsh Sw / Rsh - 1/Rt = 0
    #phi .2, Rw .05, Vsh .2, Rsh 2: clean coefficient .04 / (.8 * .05) = 1, shale coefficient .2 / 2 = .1
    #Rt = 1/.3: Sw^2 + .1 Sw - .3 = 0  ->  Sw = (-.1 + sqrt(.01 + 1.2)) / 2 = (-.1 + 1.1) / 2 = .5
    assert math.isclose(pp.simandoux(.2, 1 / .3, .2, rw=.05, rsh=2.), .5, rel_tol=1e-12)


def test_simandoux_quadratic_root_n2():
    #[HK] 1(c), the positive root of the quadratic, written out
    phi, rt, v, rw, rsh, a, m = .18, 8., .3, .06, 3., .81, 1.9
    c2 = phi**m / ((1 - v) * a * rw)
    c1 = v / rsh
    expected = (-c1 + math.sqrt(c1**2 + 4 * c2 / rt)) / (2 * c2)
    assert math.isclose(pp.simandoux(phi, rt, v, rw=rw, rsh=rsh, a=a, m=m), expected, rel_tol=1e-12)


@pytest.mark.parametrize('n', [1.7, 2., 2.4])
def test_simandoux_round_trip_general_n(n):
    #For n != 2 the code solves the generalised form [HK] 1(d) with x = 1:
    #  phi^m Sw^n / ((1 - Vsh) a Rw) + Vsh Sw^(n/2) / Rsh = 1/Rt  (exact; n = 2 gives 1(b))
    rng = np.random.default_rng(1)
    phi, v, sw = rng.uniform(.08, .3, 100), rng.uniform(0., .6, 100), rng.uniform(.1, 1., 100)
    rw, rsh, a, m = .05, 2.5, 1., 2.
    ct = phi**m * sw**n / ((1 - v) * a * rw) + v * sw**(n / 2) / rsh
    assert np.allclose(pp.simandoux(phi, 1 / ct, v, rw=rw, rsh=rsh, a=a, m=m, n=n), sw, rtol=1e-10)


#---------------------------------------------------------------- Indonesia

def indonesia_ref(phi, rt, v, rw, rsh, a=1., m=2., n=2.):
    #[HK] eq 2 (Poupon and Leveaux 1971), general n:
    #1/sqrt(Rt) = (Vsh^(1 - Vsh/2) / sqrt(Rsh) + sqrt(phi^m / (a Rw))) Sw^(n/2)
    return ((1 / math.sqrt(rt)) / (v**(1 - v / 2) / math.sqrt(rsh) + math.sqrt(phi**m / (a * rw))))**(2 / n)


def test_indonesia_hand_calculation():
    #Vsh .25: .25^(1 - .125) = .25^.875 = .29730; / sqrt(4) = .14865
    #sqrt(.2^2 / .04) = 1; 1/sqrt(16) = .25; Sw = .25 / 1.14865 = .21765
    sw = pp.indonesia(.2, 16., .25, rw=.04, rsh=4.)
    assert math.isclose(sw, .25 / (1 + .25**.875 / 2), rel_tol=1e-12)
    assert round(sw, 5) == .21765
    assert math.isclose(pp.indonesia(.17, 9., .35, rw=.08, rsh=3., a=.62, m=2.15, n=2.3),
                        indonesia_ref(.17, 9., .35, .08, 3., .62, 2.15, 2.3), rel_tol=1e-12)


def test_indonesia_round_trip():
    rng = np.random.default_rng(2)
    phi, v, sw = rng.uniform(.08, .3, 100), rng.uniform(0., .7, 100), rng.uniform(.1, 1., 100)
    rw, rsh, a, m, n = .2, 4., 1., 2., 1.8
    inv_sqrt_rt = (v**(1 - v / 2) / np.sqrt(rsh) + np.sqrt(phi**m / (a * rw))) * sw**(n / 2)
    assert np.allclose(pp.indonesia(phi, inv_sqrt_rt**-2, v, rw=rw, rsh=rsh, a=a, m=m, n=n), sw, rtol=1e-12)


#---------------------------------------------------------------- Waxman-Smits

def test_juhasz_b_hand_calculation():
    #[DM] eq 5, T = 80 degC, Rw = .05 ohm-m
    #numerator   -1.28 + .225*80 - .0004059*80^2 = -1.28 + 18 - 2.59776 = 14.12224
    #denominator 1 + .05^1.23 * (.045*80 - .27) = 1 + .025103 * 3.33 = 1.083594;  B = 13.0328
    t, rw = 80., .05
    expected = (-1.28 + .225 * t - .0004059 * t**2) / (1 + rw**1.23 * (.045 * t - .27))
    assert math.isclose(pp.juhasz_b(rw, t), expected, rel_tol=1e-12)
    assert round(pp.juhasz_b(rw, t), 4) == 13.0328


def test_qv_from_cec_units():
    #CEC 5 meq/100 g, grain density 2.65 g/cc, PHIT .2
    #grain mass per cc of rock 2.65 * .8 = 2.12 g -> 2.12 * 5 / 100 = .106 meq per cc of rock -> / .2 = .53 meq/cc pore
    assert math.isclose(pp.qv_from_cec(5., .2), .53, rel_tol=1e-12)


def test_waxman_smits_hand_calculation():
    #[DM] eq 1 as conductivity: Ct = phi^m Sw^n / a * (Cw + B Qv / Sw)
    #phi .25, Sw .4, Rw .05, B 4, Qv .3: Ct = .0625 * .16 * (20 + 1.2/.4) = .01 * 23 = .23 -> Rt = 1/.23
    sw = pp.waxman_smits(np.array([.25]), np.array([1 / .23]), qv=.3, rw=.05, b=4.)
    assert np.allclose(sw, .4, rtol=1e-12)


def test_waxman_smits_satisfies_published_form():
    #[DM] eq 1: Sw^n* = a* Rw / (phi^m* Rt (1 + Rw B Qv / Sw)); the result must satisfy it
    phi, rt, qv, rw, t, a, m, n = np.array([.12, .2, .28]), np.array([5., 12., 40.]), .4, .06, 90., 1., 1.9, 2.1
    sw = pp.waxman_smits(phi, rt, qv, rw=rw, temp_c=t, a=a, m=m, n=n)
    b = (-1.28 + .225 * t - .0004059 * t**2) / (1 + rw**1.23 * (.045 * t - .27))
    assert np.allclose(sw**n, a * rw / (phi**m * rt * (1 + rw * b * qv / sw)), rtol=1e-10)


def test_waxman_smits_round_trip():
    rng = np.random.default_rng(3)
    phi, sw, qv = rng.uniform(.08, .3, 100), rng.uniform(.1, 1., 100), rng.uniform(0., 1.5, 100)
    rw, b, a, m, n = .04, 3.5, 1., 2., 2.3
    ct = phi**m * sw**n / a * (1 / rw + b * qv / sw)
    assert np.allclose(pp.waxman_smits(phi, 1 / ct, qv, rw=rw, b=b, a=a, m=m, n=n), sw, rtol=1e-10)


#---------------------------------------------------------------- Dual water

def test_dual_water_hand_calculation():
    #[DW]: Ct = phit^m Swt^n (Cw + Swb/Swt (Cwb - Cw))
    #phit .25, Swt .5, Swb .2, Rw .05, Rwb .2: Ct = .0625 * .25 * (20 + .4 * (5 - 20)) = .015625 * 14 = .21875
    phi, rt = np.array([.25]), np.array([1 / .21875])
    assert np.allclose(pp.dual_water(phi, rt, swb=.2, rw=.05, rwb=.2), .5, rtol=1e-12)
    #SWE = (Swt - Swb) / (1 - Swb) = .3 / .8 = .375
    assert np.allclose(pp.dual_water(phi, rt, swb=.2, rw=.05, rwb=.2, effective=True), .375, rtol=1e-12)


def test_dual_water_round_trip_and_lower_bound():
    rng = np.random.default_rng(4)
    phi, swb = rng.uniform(.1, .3, 100), rng.uniform(0., .4, 100)
    swt = swb + (1 - swb) * rng.uniform(.05, 1., 100)
    rw, rwb, m, n = .03, .15, 2., 2.2
    ct = phi**m * swt**n * (1 / rw + swb / swt * (1 / rwb - 1 / rw))
    assert np.allclose(pp.dual_water(phi, 1 / ct, swb, rw=rw, rwb=rwb, m=m, n=n), swt, rtol=1e-10)
    #Rt above the all-bound-water value (Swt = Swb: Ct = .0625 * .04 * 5 = .0125, Rt = 80) gives Swt = Swb, SWE = 0
    assert np.allclose(pp.dual_water(np.array([.25]), np.array([1000.]), swb=.2, rw=.05, rwb=.2), .2)
    assert np.allclose(pp.dual_water(np.array([.25]), np.array([1000.]), swb=.2, rw=.05, rwb=.2, effective=True), 0.)


#---------------------------------------------------------------- properties of all the models

PHI = np.linspace(.08, .32, 7)
RT = np.full(7, 6.)


@pytest.mark.parametrize('a, m, n', [(1., 2., 2.), (.62, 2.15, 2.), (.81, 1.8, 2.4), (1., 2.3, 1.7)])
def test_shaly_models_reduce_to_archie(a, m, n):
    expected = archie_ref(PHI, RT, .05, a, m, n)
    expected = np.minimum(expected, 1)
    zero = np.zeros_like(PHI)
    assert np.allclose(pp.archie(PHI, RT, .05, a, m, n), expected, rtol=1e-12)
    assert np.allclose(pp.simandoux(PHI, RT, zero, .05, 2., a, m, n), expected, rtol=1e-12)
    assert np.allclose(pp.indonesia(PHI, RT, zero, .05, 2., a, m, n), expected, rtol=1e-12)
    assert np.allclose(pp.waxman_smits(PHI, RT, 0., .05, b=4., a=a, m=m, n=n), expected, rtol=1e-10)
    assert np.allclose(pp.dual_water(PHI, RT, 0., .05, .2, a, m, n), expected, rtol=1e-10)


def _all_models(rt):
    phi, v = np.full_like(rt, .2), np.full_like(rt, .25)
    return {'archie': pp.archie(phi, rt, .05),
            'simandoux': pp.simandoux(phi, rt, v, .05, 2.5),
            'simandoux_n2.4': pp.simandoux(phi, rt, v, .05, 2.5, n=2.4),
            'indonesia': pp.indonesia(phi, rt, v, .05, 2.5, n=2.2),
            'waxman_smits': pp.waxman_smits(phi, rt, .3, .05, temp_c=70),
            'dual_water': pp.dual_water(phi, rt, .15, .05, .2),
            'dual_water_swe': pp.dual_water(phi, rt, .15, .05, .2, effective=True)}


def test_sw_decreases_as_rt_increases_and_stays_in_range():
    rt = np.geomspace(1.5, 2000., 60)
    for name, sw in _all_models(rt).items():
        assert np.all((sw >= 0) & (sw <= 1)), name
        assert np.all(np.diff(sw) <= 1e-12), name
        assert sw[-1] < sw[0], name


def test_clay_conductivity_lowers_sw_against_archie():
    rt = np.geomspace(3., 200., 20)
    sw = _all_models(rt)
    for name in ['simandoux', 'indonesia', 'waxman_smits']:
        arch = pp.archie(np.full_like(rt, .2), rt, .05, n=2.2 if name == 'indonesia' else 2.)
        assert np.all(sw[name] <= arch + 1e-12), name
    #dual water: bound water more conductive than free water (rwb < rw) lowers SWT below Archie;
    #less conductive (rwb > rw, the usual saline case) raises it, since Ct = phit^m Swt^n (Cw + Swb/Swt (Cwb - Cw))
    #SWT is never below SWB = .15 (solved on [SWB, 1])
    phi, arch = np.full_like(rt, .2), pp.archie(np.full_like(rt, .2), rt, .05)
    low = pp.dual_water(phi, rt, .15, rw=.05, rwb=.02)
    assert np.all(low <= np.maximum(arch, .15) + 1e-12) and np.all(low >= .15)
    assert np.all(pp.dual_water(phi, rt, .15, rw=.05, rwb=.2) >= arch - 1e-12)


def test_nan_zero_and_series_handling():
    idx = [10, 20, 30]
    phi = pd.Series([.2, np.nan, 0.], index=idx)
    rt = pd.Series([10., 10., 10.], index=idx)
    v = pd.Series([.2, .2, .2], index=idx)
    for sw in [pp.archie(phi, rt), pp.simandoux(phi, rt, v), pp.indonesia(phi, rt, v),
               pp.waxman_smits(phi, rt, .2, b=4.), pp.dual_water(phi, rt, .1), pp.flushed_zone_saturation(phi, rt)]:
        assert isinstance(sw, pd.Series) and list(sw.index) == idx
        assert np.isnan(sw[20]) and sw[30] == 1                                         # NaN kept; no porosity -> 1


def test_implicit_models_accept_floats():
    assert isinstance(pp.waxman_smits(.2, 10., .1, rw=.05, b=4.), float)
    assert isinstance(pp.dual_water(.2, 10., .1, rw=.05), float)


#---------------------------------------------------------------- flushed zone and movable hydrocarbons

def test_flushed_zone_and_movable_hydrocarbons():
    #Sxo = sqrt(.1 / (.04 * 10)) = sqrt(.25) = .5
    assert math.isclose(pp.flushed_zone_saturation(.2, 10., rmf=.1), .5, rel_tol=1e-12)
    assert math.isclose(pp.bulk_volume_water(.2, .3), .06, rel_tol=1e-12)
    assert math.isclose(pp.movable_hydrocarbon_index(.3, .6), .5, rel_tol=1e-12)
    #phi .2, Sw .3, Sxo .8: BVW .06, BVXO .16, movable .2 * .5 = .1, residual .2 * .2 = .04
    row = pp.movable_hydrocarbons(.2, .3, .8).iloc[0]
    assert np.allclose(row[['BVW', 'BVXO', 'MOVABLE_HC', 'RESIDUAL_HC']], [.06, .16, .1, .04])
    assert math.isclose(row['BVW'] + row['MOVABLE_HC'] + row['RESIDUAL_HC'], .2)    # pore volume closes


def test_movable_hydrocarbons_index_from_any_series():
    sw = pd.Series([.3, .4], index=[7, 9])
    out = pp.movable_hydrocarbons(np.array([.2, .25]), sw, np.array([.8, .9]))
    assert list(out.index) == [7, 9]


#---------------------------------------------------------------- fluids

def test_formation_temperature_and_arps_published_example():
    #[KGS]: surface 57 degF, BHT 118 degF at TD 5398 ft, zone at 4838 ft -> 57 + 4838 (118 - 57) / 5398 = 112 degF
    t = pp.formation_temperature(4838., 57., bht=118., td=5398.)
    assert math.isclose(t, 57 + 4838 * 61 / 5398, rel_tol=1e-12) and round(t) == 112
    assert math.isclose(pp.formation_temperature(4838., 57., gradient=61 / 5398), t, rel_tol=1e-12)
    #[KGS]: Rw .05 at 75 degF -> .05 (75 + 7) / (112 + 7) = .0345 at the zone (the example rounds 6.77 to 7)
    rw = pp.arps(.05, 75., t)
    assert math.isclose(rw, .05 * (75 + 6.77) / (t + 6.77), rel_tol=1e-12)
    assert round(rw, 4) == .0345


def test_arps_celsius_and_fahrenheit_agree():
    #[KGS]: k = 6.77 (degF), 21.5 (degC); exact conversion gives (6.77 + 32) / 1.8 = 21.54, so they agree to ~0.1 %
    for t1f, t2f in [(75., 150.), (68., 250.), (80., 350.)]:
        rf = pp.arps(.1, t1f, t2f)
        rc = pp.arps(.1, (t1f - 32) / 1.8, (t2f - 32) / 1.8, unit='C')
        assert math.isclose(rf, rc, rel_tol=1e-3)
    assert math.isclose(pp.arps(pp.arps(.07, 20., 95., 'C'), 95., 20., 'c'), .07, rel_tol=1e-12)


def test_rw_from_salinity_bateman_konen():
    #Bateman and Konen (1977): Rw(75 degF) = .0123 + 3647.5 / ppm^.955
    #100,000 ppm: 100000^.955 = 10^4.775 = 59566 -> .0123 + .06124 = .07354 ohm-m
    rw75 = .0123 + 3647.5 / 1e5**.955
    assert round(pp.rw_from_salinity(1e5), 4) == .0735
    assert math.isclose(pp.rw_from_salinity(1e5), rw75, rel_tol=1e-12)
    assert math.isclose(pp.rw_from_salinity(1e5, 150.), rw75 * 81.77 / 156.77, rel_tol=1e-12)
    #degC path: 75 degF = 23.89 degC; agrees with the degF path to Arps' rounding
    assert math.isclose(pp.rw_from_salinity(1e5, (150. - 32) / 1.8, 'C'), pp.rw_from_salinity(1e5, 150.), rel_tol=1e-3)
    ppm = np.array([1e4, 3e4, 1e5, 2.5e5])
    assert np.all(np.diff(pp.rw_from_salinity(ppm, 180.)) < 0)                          # saltier -> lower Rw


def test_rw_from_salinity_series_matched_by_position():
    out = pp.rw_from_salinity(pd.Series([1e5, 5e4], index=[5, 7]), temp=pd.Series([150., 150.], index=[0, 1]))
    assert list(out.index) == [5, 7] and not out.isna().any()
    assert np.allclose(out, [pp.rw_from_salinity(1e5, 150.), pp.rw_from_salinity(5e4, 150.)])


def test_rwa():
    #Rwa = phi^m Rt / a = .04 * 25 / 1 = 1
    assert math.isclose(pp.rwa(25., .2), 1., rel_tol=1e-12)
    #in a water zone Rt = Ro = a Rw / phi^m, so Rwa = Rw and Archie with Rw = Rwa gives Sw = 1
    phi = np.array([.1, .2, .3])
    ro = .62 * .04 / phi**2.15
    assert np.allclose(pp.rwa(ro, phi, a=.62, m=2.15), .04)
    assert np.allclose(pp.archie(phi, ro, rw=.04, a=.62, m=2.15), 1.)


def test_rw_from_sp():
    #SSP = -K log10(Rmfe / Rwe), K = 61 + .133 T(degF) = 65 + .24 T(degC); Rmfe = .85 Rmf
    #150 degF: K = 61 + 19.95 = 80.95; SSP = -80.95 mV -> Rwe = .85 * .2 / 10 = .017
    rwe = lambda *a, **k: pp.rw_from_sp(*a, rwe_to_rw=False, **k)
    assert math.isclose(rwe(-80.95, .2, 150.), .017, rel_tol=1e-12)
    assert math.isclose(rwe(-60., .3, 160.), .85 * .3 * 10**(-60 / (61 + .133 * 160)), rel_tol=1e-12)
    #the degC constants are the degF ones converted and rounded (exact 65.26 + .2394 T):
    #70 degC: K = 81.8 against 82.01 at 158 degF, so at -90 mV Rw differs by 0.7 %
    assert math.isclose(pp.rw_from_sp(-90., .2, 70., 'C'), pp.rw_from_sp(-90., .2, 70 * 1.8 + 32), rel_tol=1e-2)


def test_rw_from_sp_converts_rwe_to_rw():
    #Schlumberger charts 2013, SP-1 and SP-2: ESSP -100 mV, Rmf 0.33 ohm-m at 250 degF -> Rmfeq 0.28,
    #Rweq 0.025 (SP-1), and Rw 0.03 for NaCl water (SP-2)
    assert math.isclose(pp.rw_from_sp(-100., .33, 250., rwe_to_rw=False), .025, rel_tol=.05)
    assert math.isclose(pp.rw_from_sp(-100., .33, 250.), .03, rel_tol=.05)
    #at 75 degF, Rwe 0.05: Rw = (77 * .05 + 5) / (146 - 337 * .05) = 8.85 / 129.15 = 0.068525
    k = 61 + .133 * 75
    ssp = k * math.log10(.05 / (.85 * .5))
    assert math.isclose(pp.rw_from_sp(ssp, .5, 75.), 8.85 / 129.15, rel_tol=1e-9)
    #fresher water (Rwe at 75 degF above 0.12) is returned as Rwe
    assert math.isclose(pp.rw_from_sp(-10., .5, 75.), pp.rw_from_sp(-10., .5, 75., rwe_to_rw=False), rel_tol=1e-12)
    out = pp.rw_from_sp(-70., pd.Series([.2, .3], index=[4, 6]), 150.)
    assert isinstance(out, pd.Series) and list(out.index) == [4, 6]


#---------------------------------------------------------------- Monte Carlo

def test_monte_carlo_fixed_parameters_equal_archie():
    phi, rt = np.array([.15, .25]), np.array([8., 30.])
    mc = pp.monte_carlo_archie(phi, rt, rw=.04, a=.62, m=2.15, n=2., n_iter=7)
    for col in ['SW_P10', 'SW_P50', 'SW_P90']:
        assert np.allclose(mc[col], archie_ref(phi, rt, .04, .62, 2.15, 2.), rtol=1e-12)


def test_monte_carlo_percentiles_match_triangular_quantiles():
    #Sw = sqrt(rw / (phi^2 Rt)) rises with rw, so the Sw percentiles are Sw at the rw percentiles.
    #Triangular (a, c, b) = (.02, .03, .05), (c - a) / (b - a) = 1/3:
    #  P10 = a + sqrt(.1 (b - a)(c - a)) = .02 + sqrt(3e-5) = .025477
    #  P50 = b - sqrt(.5 (b - a)(b - c)) = .05 - sqrt(3e-4) = .032679
    #  P90 = b - sqrt(.1 (b - a)(b - c)) = .05 - sqrt(6e-5) = .042254
    a, c, b = .02, .03, .05
    q = {'SW_P10': a + math.sqrt(.1 * (b - a) * (c - a)), 'SW_P50': b - math.sqrt(.5 * (b - a) * (b - c)),
         'SW_P90': b - math.sqrt(.1 * (b - a) * (b - c))}
    mc = pp.monte_carlo_archie(np.array([.2]), np.array([10.]), rw=(a, c, b), n_iter=40000, seed=11).iloc[0]
    for col, rw in q.items():
        assert abs(mc[col] - math.sqrt(rw / (.04 * 10.))) < 2e-3, col


def test_monte_carlo_seed_and_draw_order():
    phi, rt = pd.Series([.12, .22, .3], index=[3, 4, 5]), pd.Series([6., 20., 50.], index=[3, 4, 5])
    kw = dict(rw=(.03, .04, .06), m=(1.8, 2., 2.2), n=(1.9, 2., 2.3), n_iter=300)
    one, two = pp.monte_carlo_archie(phi, rt, seed=5, **kw), pp.monte_carlo_archie(phi, rt, seed=5, **kw)
    assert one.equals(two) and list(one.index) == [3, 4, 5]
    assert not one.equals(pp.monte_carlo_archie(phi, rt, seed=6, **kw))
    #independent re-run: draws in the order rw, a, m, n from the same generator, Archie written out
    g = np.random.default_rng(5)
    rw = g.triangular(.03, .04, .06, 300)
    a = np.full(300, 1.)                                                                # fixed: no draw
    m = g.triangular(1.8, 2., 2.2, 300)
    n = g.triangular(1.9, 2., 2.3, 300)
    sw = np.minimum((a[:, None] * rw[:, None] / (phi.values**m[:, None] * rt.values))**(1 / n[:, None]), 1)
    assert np.allclose(one['SW_P10'], np.percentile(sw, 10, axis=0), rtol=1e-12)
    assert np.allclose(one['SW_P50'], np.percentile(sw, 50, axis=0), rtol=1e-12)
    assert np.allclose(one['SW_P90'], np.percentile(sw, 90, axis=0), rtol=1e-12)
    assert np.all(one['SW_P10'] <= one['SW_P50']) and np.all(one['SW_P50'] <= one['SW_P90'])


def test_monte_carlo_floats_and_index_from_rt():
    assert len(pp.monte_carlo_archie(.2, 10., rw=(.03, .04, .05), n_iter=20, seed=0)) == 1
    out = pp.monte_carlo_archie(np.array([.2, .25]), pd.Series([10., 20.], index=[8, 9]), n_iter=5)
    assert list(out.index) == [8, 9]
