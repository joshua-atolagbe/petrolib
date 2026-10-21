import numpy as np
import pandas as pd
import pytest

from petrolib import petrophysics as pp


def test_velocity_and_elastic_moduli():
    # Vp = 3000 m/s, Vs = 1500 m/s, rho = 2.5 g/cc: PR = 1/3, G = 5.625 GPa, K = E = 15 GPa
    rho, dtc, dts = pd.Series([2.5]), pd.Series([304800 / 3000]), pd.Series([304800 / 1500])
    m = pp.elastic_moduli(rho, dtc, dts).iloc[0]
    assert np.isclose(m['VP'], 3000) and np.isclose(m['VS'], 1500) and np.isclose(m['VPVS'], 2)
    assert np.isclose(m['PR'], 1 / 3) and np.isclose(m['G'], 5.625) and np.isclose(m['K'], 15) and np.isclose(m['E'], 15)
    assert np.isclose(m['AI'], 7500)
    assert np.isclose(pp.velocity(1e6 / 3000, unit='us/m'), 3000)
    assert np.isclose(pp.vs_castagna(3000.), .8621 * 3000 - 1172.4)


def test_brittleness_and_strength():
    bi = pp.brittleness(np.array([10., 20., 30.]), np.array([.35, .25, .15]), e_range=(10, 30), pr_range=(.15, .35))
    assert np.allclose(bi, [0, 50, 100])
    assert np.isclose(pp.ucs_mcnally(80.), 1200 * np.exp(-.036 * 80))


def test_pressures():
    z = np.arange(0., 3001., 10.)
    sv = pp.overburden(z, np.full(len(z), 2.0))
    assert np.isclose(sv[-1], 2000 * 9.80665 * 3000 / 1e6)                              #constant density
    offshore = pp.overburden(np.array([1000., 1100.]), np.array([2.2, 2.2]), rho_top=2.0, water_depth=500)
    assert np.isclose(offshore[0], (1.03 * 500 + 2.0 * 500) * 9.80665e-3)               #water, then unlogged sediment
    assert np.isclose(offshore[1] - offshore[0], 2.2 * 100 * 9.80665e-3)
    ph = pp.hydrostatic_pressure(z)
    assert np.allclose(pp.to_gradient(ph[1:], z[1:]), 1.03) and np.allclose(pp.to_gradient(sv[1:], z[1:]), 2.0)
    assert np.isclose(pp.hydrostatic_pressure(1000., unit='psi', depth_unit='ft'), 1.03 * 1000 * .3048 * 9.80665e-3 * 145.0377)

    dt = 60 + 100 * np.exp(-z / 2000)
    trend = pp.normal_compaction_trend(z, 10**(2 - z / 5000))
    assert np.allclose(trend, 10**(2 - z / 5000))
    pp_normal = pp.pore_pressure_eaton(sv, ph, dt, dt)                                    #on the trend: hydrostatic
    assert np.allclose(pp_normal, ph)
    pp_high = pp.pore_pressure_eaton(sv, ph, dt * 1.2, dt)                                #slower than trend: overpressure
    assert np.all(pp_high[1:] > ph[1:]) and np.all(pp_high <= sv)
    res = pp.pore_pressure_eaton(sv, ph, np.full(len(z), 1.), np.full(len(z), 2.), method='resistivity')
    assert np.allclose(res, sv - (sv - ph) * .5**1.2)
    pf = pp.fracture_pressure_eaton(sv, ph, .25)
    assert np.allclose(pf, (sv - ph) / 3 + ph)
    with pytest.raises(ValueError):
        pp.pore_pressure_eaton(sv, ph, dt, dt, method='nope')


def test_gassmann():
    kmin, phi = 36.6, .25
    kfl_gas = pp.wood_fluid_modulus(.3, 2.8, .05)
    assert np.isclose(kfl_gas, 1 / (.3 / 2.8 + .7 / .05))
    k_dry = 10.
    k_gas = k_dry + (1 - k_dry / kmin)**2 / (phi / kfl_gas + (1 - phi) / kmin - k_dry / kmin**2)
    assert np.isclose(pp.gassmann(k_gas, kmin, kfl_gas, kfl_gas, phi), k_gas)              #same fluid: unchanged
    k_brine = pp.gassmann(k_gas, kmin, kfl_gas, 2.8, phi)
    expected = k_dry + (1 - k_dry / kmin)**2 / (phi / 2.8 + (1 - phi) / kmin - k_dry / kmin**2)
    assert np.isclose(k_brine, expected) and k_brine > k_gas


def test_invasion_and_sp():
    sxo = pp.flushed_zone_saturation(.2, 10., rmf=.1)
    assert np.isclose(sxo, (.1 / (.04 * 10))**.5)
    assert np.isclose(pp.movable_hydrocarbon_index(.3, .6), .5)
    mov = pp.movable_hydrocarbons(pd.Series([.2]), pd.Series([.3]), pd.Series([.8])).iloc[0]
    assert np.isclose(mov['MOVABLE_HC'], .1) and np.isclose(mov['RESIDUAL_HC'], .04) and np.isclose(mov['BVW'], .06)
    assert np.allclose(pp.vshale_sp(np.array([-80., -40., 0., 10.]), -80, 0), [0, .5, 1, 1])
    k = 61 + .133 * 150
    assert np.isclose(pp.rw_from_sp(0., .2, 150), .85 * .2)                              #no SP deflection: Rw = Rmfe
    assert np.isclose(pp.rw_from_sp(-k, .2, 150), .85 * .2 / 10)
    assert np.isclose(pp.rw_from_sp(-65 - .24 * 60, .2, 60, unit='C'), .85 * .2 / 10)


def test_electrofacies(well):
    from petrolib.processing import electrofacies
    df = well.copy()
    df.loc[:4, 'GR'] = np.nan
    f = electrofacies(df, ['GR', 'RT', 'NPHI', 'RHOB'], n_facies=2, seed=0)
    assert f[:5].isna().all() and set(f.dropna()) == {0, 1}
    by_facies = df.groupby(f)['GR'].mean()
    assert by_facies[0] < by_facies[1]                                                   #0 is the cleaner facies
    assert (df.loc[f == 0, 'VCL'].mean() < df.loc[f == 1, 'VCL'].mean())
    assert f.equals(electrofacies(df, ['GR', 'RT', 'NPHI', 'RHOB'], n_facies=2, seed=0))   #repeatable
