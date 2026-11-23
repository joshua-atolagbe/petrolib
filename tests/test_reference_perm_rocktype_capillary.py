'''
Reference tests for petrophysics.permeability, rocktype, capillary and volumetrics.

Expected values are computed here from the published equations, from SI unit definitions, or
by hand (arithmetic in the comments), never by calling petrolib. Sources:

Timur (1968), K = 0.136 PHI%^4.4 / SWI%^2: The Log Analyst 9(4); 0.136 (percent) and 8581
    (fraction) per PetroWiki "Estimating permeability based on surface area and water saturation"
    https://petrowiki.spe.org/Estimating_permeability_based_on_surface_area_and_water_saturation
Coates free-fluid model, k = [(PHI/C)^2 (FFI/BVI)]^2, C = 10, PHI in p.u.: Coates, Xiao and Prammer
    (1999) NMR Logging Principles and Applications, Halliburton, Eq. 3.28 and Fig. 1.11
    https://ux.uis.no/~s-skj/KjernMagRes.h02/Notater/NMR-book.pdf
Coates c = 70, Schlumberger chart K-4 (after Coates and Dumanoir 1974):
    https://www.aptiantechnical.com/resources/interactive-crossplots/permeability-from-porosity-and-water-saturation-k4-chart
Morris and Biggs (1967), k = (C PHI^3 / SWIRR)^2, C = 250 medium gravity oil, 79 dry gas:
    Oil & Gas Journal, July 2023, https://www.ipc66.com/s/HETERO_Log-derived-Permeability_OGJ_July_2023.pdf
Winland R35, log R35 = 0.732 + 0.588 log Kair - 0.864 log PHI%, Kolodzie (1980), Pittman (1992):
    https://www.searchanddiscovery.com/documents/pittman/index.htm
DRT = round(2 ln FZI + 10.6), Guo et al. (2005, SPE 97033): as cited in
    https://scitepress.org/Papers/2018/74333/pdf/index.html
Leverett J worked example (lab PHI 16 %, k 80 md, sigma 50 dyne/cm): Example 2-7 of
    https://perminc.com/?p=11673
Unit definitions (exact or NIST): 1 psi = 6894.757293168 Pa, 1 mD = 9.869233e-16 m2,
    g = 9.80665 m/s2, 1 ft = 0.3048 m, 1 acre = 43560 ft2 = 4046.8564224 m2, 1 bbl = 42 US gal
    = 9702 in3 = 5.614583 ft3
'''
import math

import numpy as np
import pandas as pd
import pytest

import petrolib.petrophysics as pp

PSI = 6894.757293168            # Pa
MD = 9.869233e-16               # m2
G = 9.80665                     # m/s2
FT = .3048                      # m
ACRE = 4046.8564224             # m2
BBL_FT3 = 42 * 231 / 1728       # 5.614583 ft3
BBL_PER_ACRE_FT = 43560 / BBL_FT3       # 7758.367
BBL_PER_M3 = 1 / (BBL_FT3 * FT**3)      # 6.289811


# ----------------------------------------------------------------------------------------------
# Permeability
# ----------------------------------------------------------------------------------------------

def test_timur_matches_published_percent_form():
    # 0.136 PHI%^4.4 / SWI%^2 with PHI = 25 %, SWI = 20 %:
    # 25^4.4 = 1,415,585.28 -> 0.136 * 1,415,585.28 / 400 = 481.299 md
    assert np.isclose(0.136 * 25**4.4 / 20**2, 481.299, rtol=1e-5)
    assert np.isclose(pp.timur(.25, .20), 481.299, rtol=1e-4)       # 8581 = 0.136 * 100^2.4 = 8581.02
    phi, sw = np.linspace(.05, .35, 13), np.linspace(.06, .9, 13)
    published = .136 * (100 * phi)**4.4 / (100 * sw)**2
    assert np.allclose(pp.timur(phi, sw), published, rtol=1e-5)


def test_timur_is_wyllie_rose_special_case():
    # sqrt(K) = sqrt(8581) PHI^2.2 / SWIRR
    phi, sw = np.array([.1, .2, .3]), np.array([.1, .25, .5])
    assert np.allclose(pp.wyllie_rose(phi, sw, c=math.sqrt(8581), x=2.2, y=1), pp.timur(phi, sw))


def test_coates_free_fluid_published_example():
    # Coates et al. (1999) Fig. 1.11: MPHI = 36 p.u.
    # FFI = 30, BVI = 6  -> PERM = 4,200 md : ((36/10)^2 (30/6))^2 = (12.96 * 5)^2 = 4199.04
    # FFI = 6,  BVI = 30 -> PERM = 6.7 md   : ((36/10)^2 (6/30))^2 = (12.96 * .2)^2 = 6.718
    # with fractions SWIRR = BVI / MPHI and c = 100
    assert np.isclose(pp.coates(.36, 6 / 36), 4199.04, rtol=1e-9)
    assert np.isclose(pp.coates(.36, 30 / 36), 6.718464, rtol=1e-9)
    assert round(float(pp.coates(.36, 6 / 36)), -2) == 4200 and round(float(pp.coates(.36, 30 / 36)), 1) == 6.7


def test_coates_chart_k4_constant():
    # K = (70 PHI^2 (1 - SWI) / SWI)^2, PHI = .2, SWI = .2: (70 * .04 * 4)^2 = 11.2^2 = 125.44 md
    assert np.isclose(pp.coates(.2, .2, c=70.), 125.44)
    assert pp.coates(.2, 1.) == 0. and pp.coates(.2, 1.3) == 0.         # no free fluid


def test_morris_biggs_oil_and_gas_constants():
    # PHI = .2, SWIRR = .2: oil (250 * .008 / .2)^2 = 10^2 = 100 md, gas (79 * .008 / .2)^2 = 3.16^2 = 9.9856 md
    assert np.isclose(pp.morris_biggs(.2, .2), 100.)
    assert np.isclose(pp.morris_biggs(.2, .2, fluid='gas'), 9.9856)
    assert np.isclose(pp.wyllie_rose(.2, .2, c=250, x=3, y=1), 100.)


@pytest.mark.parametrize('func', [pp.timur, pp.coates, pp.morris_biggs,
                                  lambda p, s: pp.wyllie_rose(p, s, c=93, x=2.2, y=1)])
def test_permeability_monotonic(func):
    phi = np.linspace(.02, .35, 30)
    assert np.all(np.diff(func(phi, .2)) > 0)                        # up with porosity
    sw = np.linspace(.06, .95, 30)                                   # above the .05 floor
    assert np.all(np.diff(func(.2, sw)) < 0)                         # down with SWIRR


@pytest.mark.parametrize('func', [pp.timur, pp.coates, pp.morris_biggs])
def test_permeability_floor_nan_negative_and_index(func):
    assert func(.2, .01) == func(.2, .05) == func(.2, 0.)            # SWIRR floor
    out = func(np.array([np.nan, .2, -.05, 0.]), np.array([.2, np.nan, .2, .2]))
    assert np.isnan(out[:3]).all() and out[3] == 0.                  # negative PHIE is invalid, not mirrored
    s = func(pd.Series([.1, .2], index=[1000.5, 1001.]), pd.Series([.3, .2], index=[1000.5, 1001.]))
    assert isinstance(s, pd.Series) and list(s.index) == [1000.5, 1001.]
    assert isinstance(func(.2, .2), float)


def test_permeability_dispatch():
    assert pp.permeability(.2, .3, 'coates') == pp.coates(.2, .3)
    assert pp.permeability(.2, .3, 'morris_biggs', fluid='gas') == pp.morris_biggs(.2, .3, 'gas')
    with pytest.raises(ValueError):
        pp.permeability(.2, .3, 'kozeny')


def test_swirr_buckles():
    # PHIE * SWIRR = Buckles number where not clipped: .04 / .2 = .2, .04 / .1 = .4
    phi = np.array([.02, .1, .2, .4, 0., -.1, np.nan])
    s = pp.swirr_buckles(phi, .04)
    assert np.allclose(s[:4], [1., .4, .2, .1])                      # .04 / .02 = 2 clipped to 1
    assert s[4] == 1. and np.isnan(s[5:]).all()
    assert np.allclose(phi[1:4] * s[1:4], .04)


# ----------------------------------------------------------------------------------------------
# Rock typing
# ----------------------------------------------------------------------------------------------

def test_rqi_constant_from_units():
    # RQI = sqrt(k / PHI) with k in m2, converted to microns: 1e6 sqrt(9.869233e-16) = 0.0314153
    k, phi = np.array([.1, 1., 100., 5000.]), np.array([.05, .1, .25, .3])
    si = 1e6 * np.sqrt(k * MD / phi)
    assert np.allclose(pp.rqi(k, phi), si, rtol=6e-4)                # 0.0314 is 0.0314153 rounded
    assert np.allclose(pp.rqi(k, phi), .0314 * np.sqrt(k / phi), rtol=1e-12)


def test_rqi_fzi_drt_hand_example():
    # k = 100 md, PHI = .25: RQI = .0314 sqrt(400) = .628 um, PHIZ = .25 / .75 = 1/3
    # FZI = .628 * 3 = 1.884 um; DRT = round(2 ln 1.884 + 10.6) = round(2 * .633397 + 10.6) = round(11.8668) = 12
    assert np.isclose(pp.rqi(100, .25), .628)
    assert np.isclose(pp.normalized_porosity(.25), 1 / 3)
    assert np.isclose(pp.fzi(100, .25), 1.884)
    assert pp.drt(pp.fzi(100, .25)) == 12


def test_fzi_constant_along_hydraulic_unit():
    # samples on one unit-slope line RQI = FZI * PHIZ all return that FZI: k = PHI (FZI PHIZ / .0314)^2
    phi = np.linspace(.05, .3, 11)
    for f in (.3, 1., 4.):
        k = phi * (f * phi / (1 - phi) / .0314)**2
        assert np.allclose(pp.fzi(k, phi), f)


def test_drt_classes():
    # 2 ln FZI + 10.6 = n + .2 -> DRT n
    n = np.arange(5, 17)
    assert np.array_equal(pp.drt(np.exp((n - 10.4) / 2)), n)
    assert np.isnan(pp.drt(np.array([0., -1., np.nan, np.inf]))).all()
    s = pp.drt(pd.Series([1., np.e], index=['a', 'b']))
    assert list(s.index) == ['a', 'b'] and list(s) == [11, 13]       # round(10.6), round(12.6)


def test_winland_r35_hand_example():
    # k = 100 md, PHI = 20 %: log R35 = .732 + .588 * 2 - .864 * log10(20) = 1.908 - 1.124090 = .783910
    # R35 = 10^.783910 = 6.0801 um (porosity in percent; with the fraction it would be 325 um)
    assert np.isclose(pp.winland_r35(100, .2), 6.0801, rtol=1e-4)
    k, phi = np.array([.01, 1., 1000.]), np.array([.05, .15, .3])
    assert np.allclose(np.log10(pp.winland_r35(k, phi)), .732 + .588 * np.log10(k) - .864 * np.log10(100 * phi))
    assert pp.pore_type(np.array([pp.winland_r35(100, .2)]))[0] == 'macroport'


def test_pore_type_boundaries():
    # classes as documented in the module: >= 10, 2-10, .5-2, .1-.5, < .1 um
    r = np.array([10., 9.99, 2., 1.99, .5, .49, .1, .099, 0., -1., np.nan])
    assert list(pp.pore_type(r)) == ['megaport', 'macroport', 'macroport', 'mesoport', 'mesoport', 'microport',
                                     'microport', 'nanoport', 'nanoport', None, None]


def test_lorenz_hand_example():
    # layers (k, PHI, h): A (10, .1, 2), B (100, .2, 1), C (1, .1, 1), in depth order A, B, C
    # k/PHI: A 100, B 500, C 10 -> classic order B, A, C
    # kh: B 100, A 20, C 1 (total 121); PHI h: B .2, A .2, C .1 (total .5)
    # storage 0, .4, .8, 1; flow 0, 100/121, 120/121, 1
    # area = .4 * (100/121) / 2 + .4 * (220/121) / 2 + .2 * (241/121) / 2 = 88.1/121
    # LC = 2 (88.1/121 - .5) = 55.2/121 = .456198
    k, phi, h = np.array([10., 100., 1.]), np.array([.1, .2, .1]), np.array([2., 1., 1.])
    curve, lc = pp.lorenz(k, phi, h)
    assert np.isclose(lc, 55.2 / 121)
    assert np.allclose(curve['STORAGE'], [0, .4, .8, 1]) and np.allclose(curve['FLOW'], [0, 100 / 121, 120 / 121, 1])
    strat, lc_strat = pp.lorenz(k, phi, h, stratigraphic=True)        # depth order A, B, C
    assert np.allclose(strat['STORAGE'], [0, .4, .8, 1]) and np.allclose(strat['FLOW'], [0, 20 / 121, 120 / 121, 1])
    assert lc_strat == lc                                             # coefficient from the classic curve
    _, lc_shuffled = pp.lorenz(k[::-1], phi[::-1], h[::-1])
    assert np.isclose(lc_shuffled, lc)                                # classic LC ignores depth order


def test_lorenz_limits():
    _, lc = pp.lorenz(np.full(40, 50.), np.full(40, .2))
    assert abs(lc) < 1e-12                                            # homogeneous: 0
    _, lc = pp.lorenz(np.r_[1., 1., 1.], np.r_[.1, .2, .3])           # k/PHI varies but ...
    assert lc > 0
    _, lc = pp.lorenz(np.r_[1., 2., 3.], np.r_[.1, .2, .3])           # ... same k/PHI: homogeneous
    assert abs(lc) < 1e-12
    for n in (10, 100, 1000):
        # one layer carries all the flow among n equal PHI h layers: curve (0,0)-(1/n,1)-(1,1),
        # area 1 - 1/(2n), LC = 1 - 1/n
        k = np.full(n, 1e-6); k[n // 2] = 1e6
        _, lc = pp.lorenz(k, np.full(n, .2))
        assert np.isclose(lc, 1 - 1 / n, atol=1e-9)
    k = pd.Series([1., np.nan, 10., 100.]); phi = pd.Series([.1, .2, np.nan, .2])
    curve, lc = pp.lorenz(k, phi)                                     # NaN samples dropped
    # left: (k 100, PHI .2) then (1, .1): storage 0, 2/3, 1; flow 0, 100/101, 1
    area = (2 / 3) * (100 / 101) / 2 + (1 / 3) * (1 + 100 / 101) / 2
    assert len(curve) == 3 and np.isclose(lc, 2 * (area - .5))


# ----------------------------------------------------------------------------------------------
# Capillary pressure
# ----------------------------------------------------------------------------------------------

def test_pc_from_height_si():
    # Pc = drho g h: .8 g/cc, 30 m -> 800 * 9.80665 * 30 = 235,359.6 Pa = 34.136 psi
    # petrolib uses the rounded .433 psi/ft per g/cc (exact .433528), 0.12 % low
    h, drho = np.array([1., 30., 250.]), .8
    si = drho * 1000 * G * h / PSI
    assert np.isclose(si[1], 34.136, rtol=1e-4)
    assert np.allclose(pp.pc_from_height(h, drho), si, rtol=2e-3)
    assert np.allclose(pp.pc_from_height(h / FT, drho, unit='ft'), si, rtol=2e-3)
    assert np.isclose(pp.pc_from_height(100., 1., unit='ft'), 43.3)        # .433 psi/ft x 100 ft


def test_height_pc_round_trip():
    h = pd.Series([0., 5., 50., np.nan], index=[2000., 2005., 2050., 2100.])
    for unit in ('m', 'ft'):
        back = pp.height_from_pc(pp.pc_from_height(h, .35, unit), .35, unit)
        assert list(back.index) == list(h.index)
        assert np.allclose(back, h, equal_nan=True)


def test_leverett_j_constant_and_si_form():
    # J = Pc sqrt(k / PHI) / (sigma cos theta) in SI; with psi, md, dyne/cm the factor is
    # 6894.757 * sqrt(9.869233e-16) / 1e-3 = .216601
    assert np.isclose(PSI * math.sqrt(MD) / 1e-3, .2166, rtol=1e-5)
    pc, k, phi = np.array([.5, 5., 50.]), np.array([1., 100., 2000.]), np.array([.08, .2, .3])
    for sigma, theta in ((72., 0.), (26., 30.), (480., 140.)):
        si = pc * PSI * np.sqrt(k * MD / phi) / (sigma * 1e-3 * np.cos(np.radians(theta)))
        assert np.allclose(pp.leverett_j(pc, k, phi, sigma, theta), si, rtol=1e-5)


def test_leverett_j_published_example():
    # Example 2-7 (perminc.com): lab PHI .16, k 80 md, sigma 50 dyne/cm (cos theta = 1)
    pc_lab = np.array([.5, .6, .75, 1.05, 1.75])
    j_pub = np.array([.048, .058, .073, .102, .169])
    # e.g. .2166 * .5 / 50 * sqrt(80 / .16) = .2166 * .01 * 22.3607 = .04843
    assert np.allclose(pp.leverett_j(pc_lab, 80, .16, sigma=50, theta=0), j_pub, atol=1e-3)
    # reservoir PHI .19, k 120 md, same sigma: published Pc .441 ... 1.553 psi give the same J
    pc_res = np.array([.441, .533, .671, .938, 1.553])
    assert np.allclose(pp.leverett_j(pc_res, 120, .19, sigma=50, theta=0), j_pub, atol=5e-4)


def test_leverett_j_scales_pc_between_rocks():
    # same J -> Pc2 = Pc1 sqrt((k1/PHI1) / (k2/PHI2)) at the same sigma cos theta
    pc1 = np.array([1., 3., 10.])
    j = pp.leverett_j(pc1, 10., .15)
    pc2 = pc1 * math.sqrt((10 / .15) / (500 / .25))
    assert np.allclose(pp.leverett_j(pc2, 500., .25), j)
    s = pp.leverett_j(pd.Series([1., np.nan], index=[7, 8]), 100, .2)
    assert list(s.index) == [7, 8] and np.isnan(s[8])


def test_saturation_height_fit_recovers_parameters():
    # synthetic SW = .2 J^-.45 from heights 1-100 m above FWL, k 50 md, PHI .2, oil-water drho .3
    h = np.linspace(1, 100, 200)
    j = pp.leverett_j(pp.pc_from_height(h, .3), 50., .2)
    a, b = pp.fit_saturation_height(j, .2 * j**-.45)
    assert np.isclose(a, .2, rtol=1e-10) and np.isclose(b, -.45, rtol=1e-10)
    rng = np.random.default_rng(0)
    sw_noisy = .2 * j**-.45 * np.exp(rng.normal(0, .02, j.size))
    sw_noisy[::25] = np.nan                                           # NaN samples skipped
    a, b = pp.fit_saturation_height(j, sw_noisy)
    assert abs(a - .2) < .01 and abs(b + .45) < .01
    sw = pp.sw_from_j(j, .2, -.45, swirr=.1)
    assert np.allclose(sw, np.clip(.2 * j**-.45, .1, 1))
    assert pp.sw_from_j(0., .2, -.45) == 1.                           # at the FWL
    assert np.all(np.diff(sw) <= 0)                                   # SW falls with height


def test_sw_height_round_trip():
    # h -> Pc -> J -> SW, then back: J = (SW / a)^(1/b), Pc from J, h from Pc
    a, b, k, phi, drho = .25, -.6, 200., .22, .25
    h = np.array([2., 10., 40.])
    sw = pp.sw_from_j(pp.leverett_j(pp.pc_from_height(h, drho), k, phi), a, b)
    j_back = (sw / a)**(1 / b)
    pc_back = j_back * 26 * math.cos(math.radians(30)) / (.2166 * math.sqrt(k / phi))
    assert np.allclose(pp.height_from_pc(pc_back, drho), h)


# ----------------------------------------------------------------------------------------------
# Volumetrics
# ----------------------------------------------------------------------------------------------

def test_barrels_per_acre_ft():
    # 43560 ft3 / 5.614583 ft3/bbl = 7758.37 bbl; petrolib uses 7758
    assert np.isclose(BBL_PER_ACRE_FT, 7758.367, rtol=1e-6)
    assert np.isclose(pp.stoiip(1, 1, 1, 0, bo=1), BBL_PER_ACRE_FT, rtol=1e-4)


def test_stoiip_field_hand_calc():
    # 640 acres x 35 ft = 22,400 acre-ft; x .18 = 4,032; x .7 = 2,822.4 acre-ft HCPV
    # x 7758 = 21,896,179.2 rb; / 1.25 = 17,516,943.36 STB
    assert np.isclose(pp.stoiip(area=640, net=35, phi=.18, sw=.3, bo=1.25), 17516943.36)


def test_stoiip_metric_hand_calc_and_unit_consistency():
    # 1 km2 x 10 m x .2 x .7 / 1.2 = 1,166,666.67 sm3
    assert np.isclose(pp.stoiip(1e6, 10, .2, .3, bo=1.2, unit='metric'), 1166666.6667)
    # the docstring field case in metric units, converted back to STB
    metric = pp.stoiip(640 * ACRE, 35 * FT, .18, .3, bo=1.25, unit='metric')
    assert np.isclose(metric * BBL_PER_M3, pp.stoiip(640, 35, .18, .3, bo=1.25), rtol=1e-4)


def test_giip_field_metric_and_bg_conventions():
    # 43560 x 1000 acres x 50 ft x .15 x .75 / .004 = 61,256,250,000 scf (61.3 Bcf)
    field = pp.giip(1000, 50, .15, .25, bg=.004)
    assert np.isclose(field, 6.125625e10)
    metric = pp.giip(1000 * ACRE, 50 * FT, .15, .25, bg=.004, unit='metric')   # Bg rm3/sm3 = rcf/scf
    assert np.isclose(metric / FT**3, field, rtol=1e-9)
    # with Bg in rb/scf the form is 7758 A h PHI (1 - SW) / Bg
    bg_rb = .004 / BBL_FT3
    assert np.isclose(BBL_PER_ACRE_FT * 1000 * 50 * .15 * .75 / bg_rb, field, rtol=1e-9)


def test_volumetrics_series_and_nan():
    phi = pd.Series([.2, np.nan, .25], index=['A', 'B', 'C'])
    s = pp.stoiip(100., 20., phi, .3, bo=1.1)
    assert list(s.index) == ['A', 'B', 'C'] and np.isnan(s['B'])
    assert np.isclose(s['A'], 7758 * 100 * 20 * .2 * .7 / 1.1)
    g = pp.giip(pd.Series([10., 20.], index=[3, 4]), 5., .2, .4, unit='metric')
    assert list(g.index) == [3, 4] and np.isclose(g[4], 20 * 5 * .2 * .6 / .005)
    assert np.all(np.diff(pp.stoiip(1, 1, .2, np.linspace(0, 1, 6))) < 0)
