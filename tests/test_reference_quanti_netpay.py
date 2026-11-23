'''
Reference tests of the Quanti workflow, the net pay summation, the pay summary workbook and the
curve name and unit standardisation. Expected values are worked out by hand in the comments, not
with petrolib
'''
import numpy as np
import pandas as pd
import pytest

from petrolib.petrophysics import Quanti, net_flags, sample_thickness, weighted_mean, summarize, cutoff_sensitivity
from petrolib.mnemonics import find_curves, log_kind, standardize


def timur(phi, sw):
    # Timur (1968): K = 8581 PHI^4.4 / SW^2, SW floored at 0.05
    return 8581. * phi**4.4 / max(sw, .05)**2


# ---- sample thickness ------------------------------------------------------------------------

def test_sample_thickness_regular_and_zone_edges():
    depth = pd.Series([100., 100.25, 100.5, 100.75, 101.])
    # step .25: each sample is depth +/- .125; the end samples are cut at the zone top and base
    assert np.allclose(sample_thickness(depth, 100., 101.), [.125, .25, .25, .25, .125])
    # tops between samples: 100.0 -> [99.875, 100.125] cut at 100.1 = .025; 101.0 -> [100.875, 100.9] = .025
    h = sample_thickness(depth, 100.1, 100.9)
    assert np.allclose(h, [.025, .25, .25, .25, .025]) and np.isclose(h.sum(), .8)


def test_sample_thickness_gap_is_not_net():
    # step = median(.5, .5, 1.5, .5) = .5; the 1.5 m gap keeps .25 on each side, 1.0 m carries nothing
    h = sample_thickness(pd.Series([0., .5, 1., 2.5, 3.]), 0., 3.)
    assert np.allclose(h, [.25, .5, .5, .5, .25]) and np.isclose(h.sum(), 2.)


def test_sample_thickness_irregular_sampling_does_not_overlap():
    # step = median(.1, .1, .8, 1, 1) = .8. Each side goes halfway to the neighbour, at most .4:
    #   0  : [0, .05]       .05        .1 : [.05, .15]  .1        .2 : [.15, .6]  .45
    #   1  : [.6, 1.4]      .8         2  : [1.6, 2.4]  .8        3  : [2.6, 3]   .4
    # sum 2.6 <= gross 3 (+/- step/2 for every sample gave 3.5, a net to gross of 1.17)
    depth = pd.Series([0., .1, .2, 1., 2., 3.])
    h = sample_thickness(depth, 0., 3.)
    assert np.allclose(h, [.05, .1, .45, .8, .8, .4]) and h.sum() <= 3.
    # same thickness for each depth whatever the row order
    rev = sample_thickness(depth[::-1], 0., 3.)
    assert np.allclose(rev.sort_index(), h)


def test_sample_thickness_repeated_depth():
    # step = median(1, 0, 1) = 1; the two samples at 1.0 share [.5, 1.5]
    h = sample_thickness(pd.Series([0., 1., 1., 2.]), 0., 2.)
    assert np.allclose(h, [.5, .5, .5, .5]) and np.isclose(h.sum(), 2.)


# ---- averages and flags -----------------------------------------------------------------------

def test_weighted_mean():
    # (.2 * 3 + .8 * 1) / 4 = .35, the NaN value and its weight are skipped
    assert np.isclose(weighted_mean([.2, np.nan, .8], [3., 5., 1.]), .35)
    # array values and Series weights pair by position, whatever the index of the weights
    assert np.isclose(weighted_mean(np.array([.2, .8]), pd.Series([3., 1.], index=[10, 11])), .35)
    assert np.isnan(weighted_mean([np.nan], [1.])) and np.isnan(weighted_mean([.3, .4], [0., 0.]))


def test_net_flags_cutoffs_and_nan():
    # cutoffs Vsh .5, PHIE .1, SW .6; a value equal to its cutoff passes, NaN fails
    vsh = pd.Series([.5, .49, .1, .1, np.nan, .1, .1, .1], index=range(10, 18))
    phie = pd.Series([.2, .2, .1, .11, .2, np.nan, .2, .2], index=range(10, 18))
    sw = pd.Series([.3, .3, .3, .3, .3, .3, .6, np.nan], index=range(10, 18))
    f = net_flags(vsh, phie, sw, .5, .1, .6)
    assert f['ROCK_NET_FLAG'].tolist() == [1, 1, 1, 1, 0, 1, 1, 1]
    assert f['RES_NET_FLAG'].tolist() == [1, 1, 1, 1, 0, 0, 1, 1]
    assert f['PAY_NET_FLAG'].tolist() == [1, 1, 1, 1, 0, 0, 1, 0]
    assert list(f.index) == list(range(10, 18))
    # a cutoff of 1 keeps samples at exactly Vsh = 1 or SW = 1
    assert net_flags(pd.Series([1.]), pd.Series([.2]), pd.Series([1.]), 1., 0., 1.).iloc[0].tolist() == [1, 1, 1]
    # strict tests: a value equal to its cutoff fails
    f = net_flags(vsh, phie, sw, .5, .1, .6, inclusive=False)
    assert f['ROCK_NET_FLAG'].tolist() == [0, 1, 1, 1, 0, 1, 1, 1]
    assert f['RES_NET_FLAG'].tolist() == [0, 1, 0, 1, 0, 0, 1, 1]
    assert f['PAY_NET_FLAG'].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert list(f.index) == list(range(10, 18))


def test_summarize_with_gap_and_nan():
    # zone 10-13 m, step .5, gap 11.25-12.25: h = [.25, .5, .5, .5, .25]
    depth = [10., 10.5, 11., 12.5, 13.]
    vsh = [.1, .2, .6, .3, 0.]
    phie = [.2, .1, .05, .3, .25]
    sw = [.5, 1., 1., np.nan, .32]
    perm = [100., 1., .1, np.nan, 300.]
    flag = [1, 1, 0, 1, 1]
    s = summarize(depth, vsh, phie, sw, flag, 10., 13., perm=perm)
    # net = .25 + .5 + .5 + .25 = 1.5 of gross 3
    assert np.isclose(s['Gross'], 3.) and np.isclose(s['Net'], 1.5) and np.isclose(s['Not Net'], 1.5)
    assert np.isclose(s['NTG'], .5)
    # Vsh: (.1*.25 + .2*.5 + .3*.5 + 0*.25) / 1.5 = .275 / 1.5
    assert np.isclose(s['Average VShale'], .275 / 1.5)
    # PHIE: (.2*.25 + .1*.5 + .3*.5 + .25*.25) / 1.5 = .3125 / 1.5 ; PHIH = .3125
    assert np.isclose(s['Average Porosity'], .3125 / 1.5) and np.isclose(s['PHIH'], .3125)
    # SW by pore volume, sample 4 (SW NaN) left out: phi*h = .05, .05, .0625
    # (.5*.05 + 1*.05 + .32*.0625) / .1625 = .095 / .1625
    assert np.isclose(s['Average Water Saturation'], .095 / .1625)
    # BVW by thickness without sample 4: (.1*.25 + .1*.5 + .08*.25) / 1.0 = .095
    assert np.isclose(s['BVW'], .095)
    # HCPV: .2*.5*.25 + .1*0*.5 + .25*.68*.25 = .025 + 0 + .0425
    assert np.isclose(s['HCPV'], .0675)
    # permeability by thickness without sample 4: (100*.25 + 1*.5 + 300*.25) / 1.0
    assert np.isclose(s['Average Permeability'], 100.5)


def test_cutoff_sensitivity_by_hand():
    # depth 0-4 m, step 1: h = [.5, 1, 1, 1, .5], gross = 4 (first and last depth by default)
    df = pd.DataFrame({'DEPTH': [0., 1., 2., 3., 4.], 'VShale': .1, 'PHIE': [.05, .1, .15, .2, .25], 'SW': .3})
    sens = cutoff_sensitivity(df, cutoff='por', values=[0., .12, .2], vsh_cutoff=.5, sw_cutoff=.6)
    # PHIE >= 0: all 4 m ; PHIE >= .12: 1 + 1 + .5 = 2.5 ; PHIE >= .2: 1 + .5 = 1.5
    assert np.allclose(sens['Net'], [4., 2.5, 1.5]) and np.allclose(sens['Gross'], 4.)
    # at .12: (.15*1 + .2*1 + .25*.5) / 2.5 = .19
    assert np.isclose(sens['Average Porosity'].iloc[1], .19)


# ---- Quanti, end to end -------------------------------------------------------------------------

def synthetic_well(scale:float=1.) -> pd.DataFrame:

    '''
    Zone A 100-104 m with a gap at 101.75-102.75, zone B 104-105 m sharing the sample at 104.0,
    and a sample above and below the zones that must not be used. RHOB = 2.65 - 1.65 PHIT, so the
    density porosity is PHIT; RT = rw / (PHIE SW)^2 with rw = .05 gives the chosen Archie SW
    '''

    phit = [.30, .25, .20, .15, .05, .125, .04, .16, .10, .02, .30]
    return pd.DataFrame({
        'DEPTH': np.array([99.5, 100., 100.5, 101., 101.5, 103., 103.5, 104., 104.5, 105., 105.5]) * scale,
        'GR':    [500., 20., 20., 70., 120., 45., 120., 30., 130., 230., 0.],
        'RT':    [1., 20., 1.953125, 2., 2., 20., 2., np.nan, 2., 2., 1.],
        'NPHI':  .25,
        'RHOB':  [2.65 - 1.65 * p for p in phit]})


def run(df, tops=(100., 104.), bots=(104., 105.), unit='m'):
    q = Quanti(df, ['A', 'B'], list(tops), list(bots), [102., 104.5])
    q.vshale(method='linear')
    q.porosity(method='density', rhob_matrix=2.65, rhob_fluid=1., fzs=.1)
    q.water_saturation(method='archie', rw=.05, a=1., m=2., n=2.)
    q.permeability(method='timur')
    zones = q.flags(vsh_cutoff=.4, por_cutoff=.12, sw_cutoff=.5, ref_unit=unit)
    return q, zones, q.paySummary('SYN').data.set_index(['Zones', 'Flag Name'])


def test_quanti_end_to_end_by_hand():
    q, zones, s = run(synthetic_well())

    # GR endpoints (linear interpolation of the sorted zone GR)
    # A: 20 20 30 45 70 120 120, n = 7: P5 at .3 -> 20, P95 at 5.7 -> 120
    # B: 30 130 230, n = 3: P5 at .1 -> 30 + .1*100 = 40, P95 at 1.9 -> 130 + .9*100 = 220
    assert np.allclose(q()['GR_Sand'], [20., 40.]) and np.allclose(q()['GR_Shale'], [120., 220.])

    a, b = zones
    assert a['DEPTH'].tolist() == [100., 100.5, 101., 101.5, 103., 103.5, 104.]
    assert b['DEPTH'].tolist() == [104., 104.5, 105.]
    # A: Vsh = (GR - 20) / 100 ; PHIE = PHIT - .1 Vsh, floored at 0
    assert np.allclose(a['VShale'], [0., 0., .5, 1., .25, 1., .1])
    assert np.allclose(a['PHIE'], [.25, .2, .1, 0., .1, 0., .15])
    # SW = sqrt(.05 / (PHIE^2 RT)): .2, .8, sqrt(2.5) -> 1, PHIE 0 -> 1, .5, 1, RT NaN -> NaN
    assert np.allclose(a['SW'], [.2, .8, 1., 1., .5, 1., np.nan], equal_nan=True)
    assert a['ROCK_NET_FLAG'].tolist() == [1, 1, 0, 0, 1, 0, 1]
    assert a['RES_NET_FLAG'].tolist() == [1, 1, 0, 0, 0, 0, 1]
    assert a['PAY_NET_FLAG'].tolist() == [1, 0, 0, 0, 0, 0, 0]
    # B: Vsh = (GR - 40) / 180 clipped: 0, .5, 1 ; PHIE .16, .05, 0
    assert np.allclose(b['VShale'], [0., .5, 1.]) and np.allclose(b['PHIE'], [.16, .05, 0.])

    k0, k1, k4 = timur(.25, .2), timur(.2, .8), timur(.1, .5)       # about 481.3, 11.27, 1.366 mD
    # A: step .5, h = [.25, .5, .5, .5, .5, .5, .25] (the gap 101.75-102.75 is not net)
    expected = {
        # ROCK: samples 0, 1, 4, 6 -> h .25 .5 .5 .25
        ('A', 'ROCK'): dict(Gross=4., Net=1.5, NTG=.375,
                            VSH=(.25*.5 + .1*.25) / 1.5,                          # .1
                            PHI=(.25*.25 + .2*.5 + .1*.5 + .15*.25) / 1.5,        # .25 / 1.5
                            SW=(.2*.0625 + .8*.1 + .5*.05) / (.0625 + .1 + .05),  # sample 6 has no SW
                            BVW=(.05*.25 + .16*.5 + .05*.5) / 1.25,
                            K=(k0*.25 + k1*.5 + k4*.5) / 1.25,
                            PHIH=.25, HCPV=.25*.8*.25 + .2*.2*.5 + .1*.5*.5),
        # RES: samples 0, 1, 6
        ('A', 'RES'): dict(Gross=4., Net=1., NTG=.25, VSH=.1*.25, PHI=.0625 + .1 + .0375,
                           SW=(.2*.0625 + .8*.1) / (.0625 + .1), BVW=(.05*.25 + .16*.5) / .75,
                           K=(k0*.25 + k1*.5) / .75, PHIH=.2, HCPV=.25*.8*.25 + .2*.2*.5),
        # PAY: sample 0
        ('A', 'PAY'): dict(Gross=4., Net=.25, NTG=.0625, VSH=0., PHI=.25, SW=.2, BVW=.05, K=k0,
                           PHIH=.0625, HCPV=.05),
        # B: h = [.25, .5, .25]; ROCK and RES are the boundary sample 104.0, with no SW
        ('B', 'ROCK'): dict(Gross=1., Net=.25, NTG=.25, VSH=0., PHI=.16, SW=np.nan, BVW=np.nan, K=np.nan,
                            PHIH=.04, HCPV=0.),
        ('B', 'RES'): dict(Gross=1., Net=.25, NTG=.25, VSH=0., PHI=.16, SW=np.nan, BVW=np.nan, K=np.nan,
                           PHIH=.04, HCPV=0.),
        ('B', 'PAY'): dict(Gross=1., Net=0., NTG=0., VSH=np.nan, PHI=np.nan, SW=np.nan, BVW=np.nan, K=np.nan,
                           PHIH=0., HCPV=0.),
    }
    columns = {'Gross': 'Gross', 'Net': 'Net', 'NTG': 'NTG', 'VSH': 'Average VShale', 'PHI': 'Average Porosity',
               'SW': 'Average Water Saturation', 'BVW': 'BVW', 'K': 'Average Permeability', 'PHIH': 'PHIH',
               'HCPV': 'HCPV'}
    for key, values in expected.items():
        for k, v in values.items():
            assert np.isclose(s.loc[key, columns[k]], v, rtol=1e-9, equal_nan=True), (key, k)
        assert np.isclose(s.loc[key, 'Not Net'], values['Gross'] - values['Net'])
    assert s['Top'].tolist() == [100.] * 3 + [104.] * 3 and s['Bottom'].tolist() == [104.] * 3 + [105.] * 3
    assert set(s['Unit']) == {'m'} and set(s['Well']) == {'SYN'}


def test_quanti_depth_in_feet():
    # the same well in feet: thicknesses, PHIH and HCPV scale by 1/.3048, averages do not change
    ft = 1 / .3048
    _, _, m = run(synthetic_well())
    _, _, f = run(synthetic_well(ft), tops=(100*ft, 104*ft), bots=(104*ft, 105*ft), unit='ft')
    for col in ('Gross', 'Net', 'Not Net', 'PHIH', 'HCPV'):
        assert np.allclose(f[col], m[col] * ft)
    for col in ('NTG', 'Average VShale', 'Average Porosity', 'Average Water Saturation', 'BVW', 'Average Permeability'):
        assert np.allclose(f[col], m[col], equal_nan=True)
    assert set(f['Unit']) == {'ft'}


def test_quanti_row_order_does_not_matter():
    df = synthetic_well()
    shuffled = df.sample(frac=1., random_state=3).reset_index(drop=True)
    _, _, s = run(df)
    _, _, t = run(shuffled)
    num = s.select_dtypes('number').columns
    assert np.allclose(t[num], s[num], equal_nan=True)


def test_quanti_keeps_the_parameters_of_each_step():
    # methods and parameters set in earlier steps are used again when flags() recomputes them
    q = Quanti(synthetic_well(), ['A'], [100.], [104.], [102.])
    q.vshale(method='stieber_1')
    q.porosity(method='density', rhob_matrix=2.65, fzs=.1)
    q.water_saturation(method='simandoux', rw=.05, rsh=4.)
    q.permeability(method='coates')
    a = q.flags(vsh_cutoff=.4, por_cutoff=.12, sw_cutoff=.5)[0]

    igr = np.array([0., 0., .5, 1., .25, 1., .1])
    vsh = igr / (2 - igr)                                                # Stieber
    phie = np.clip(np.array([.25, .2, .15, .05, .125, .04, .16]) - .1 * vsh, 0, None)
    rt = np.array([20., 1.953125, 2., 2., 20., 2., np.nan])
    # Simandoux, n = 2: phie^2 / (rw (1 - Vsh)) SW^2 + Vsh / rsh SW - 1 / RT = 0, the positive root
    with np.errstate(divide='ignore', invalid='ignore'):
        qa, qb, qc = phie**2 / (.05 * (1 - vsh)), vsh / 4., -1 / rt
        sw = (-qb + np.sqrt(qb**2 - 4 * qa * qc)) / (2 * qa)
    sw = np.where(phie == 0, 1., np.minimum(sw, 1.))
    sw[np.isnan(rt)] = np.nan
    s = np.clip(sw, .05, None)
    k = (100 * phie**2 * (1 - s) / s)**2                                  # Coates, c = 100
    assert np.allclose(a['VShale'], vsh) and np.allclose(a['PHIE'], phie)
    assert np.allclose(a['SW'], sw, equal_nan=True) and np.allclose(a['Perm'], k, equal_nan=True)
    q.paySummary('SYN')
    methods = q.report().data.iloc[:, 0]
    assert methods[['VShale', 'Porosity', 'Water Saturation', 'Permeability']].tolist() == \
        ['Stieber_1', 'Density', 'Simandoux', 'Coates']


@pytest.mark.parametrize('kw', [dict(use_mean=False), dict(use_median=False), dict(use_mean=True, use_median=True)])
def test_quanti_gr_matrix_options(kw):
    q = Quanti(synthetic_well(), ['A'], [100.], [104.], [102.], **kw)
    q.vshale()
    # mean of zone A GR: (20 + 20 + 70 + 120 + 45 + 120 + 30) / 7 = 425 / 7
    assert np.isclose(q()['GR_Matrix'].iloc[0], 425 / 7 if kw.get('use_mean') else 75.)


# ---- pay summary workbook -----------------------------------------------------------------------

def test_workbook_keeps_values_and_fits_widths_to_shown_text(tmp_path):
    openpyxl = pytest.importorskip('openpyxl')
    from petrolib.io import write_pay_summary
    summary = pd.DataFrame({'Well': ['W'], 'Zones': ['A'], 'Flag Name': ['ROCK'], 'Top': [2823.5], 'Bottom': [2986.5],
                            'Unit': ['m'], 'Gross': [163.], 'Net': [130.70000000000073], 'NTG': [0.8018404907975505],
                            'Average Permeability': [24.192733011122243], 'PHIH': [16.39071823649254]})
    logs = pd.DataFrame({'DEPTH': [2823.5, 2823.6], 'GR': [45.123456789123, 50.], 'PAY_NET_FLAG': [1, 0]})
    ws_all = openpyxl.load_workbook(write_pay_summary(summary, tmp_path / 'pay', log_data=logs))
    ws = ws_all['Pay Summary']
    cols = {c.value: c.column_letter for c in ws[1]}
    # stored values are not rounded, only the number format rounds them. openpyxl writes floats
    # with 16 significant digits ('%.16g'), so they agree to about 1e-16, not bit for bit
    net, ntg = ws[f"{cols['Net']}2"], ws[f"{cols['NTG']}2"]
    assert np.isclose(net.value, 130.70000000000073, rtol=1e-15, atol=0) and net.value != 130.7
    assert np.isclose(ntg.value, 0.8018404907975505, rtol=1e-15, atol=0)
    assert net.number_format == '0.00' and ntg.number_format == '0.000'
    assert ws[f"{cols['Average Permeability']}2"].number_format == '0.0'
    # widths: '130.70' -> 6 + 2 = 8, '0.802' -> 8 (minimum), '2823.50' -> 9 ('130.70000000000073' gave 20)
    assert ws.column_dimensions[cols['Net']].width == 8
    assert ws.column_dimensions[cols['NTG']].width == 8
    assert ws.column_dimensions[cols['Top']].width == 9
    assert ws.column_dimensions[cols['Average Permeability']].width == 22     # header text, at most 22
    log = ws_all['Log Data']
    # the flag keeps the general format (not 1.00); GR in the general format shows 10 digits: 45.12345679
    assert log['C2'].number_format == 'General' and log['B2'].value == 45.123456789123
    assert log.column_dimensions['B'].width == 13


# ---- curve names and units ----------------------------------------------------------------------

def test_computed_curves_are_not_raw_logs():
    computed = ['VSH_GR', 'PHIE_ND', 'PHIT_ND', 'IGR', 'SW_AR', 'PERM_TIM', 'VShale', 'PHIE', 'PHIT', 'PHID',
                'SW', 'SH', 'Perm', 'BVW', 'ROCK_NET_FLAG']
    assert all(log_kind(c) is None for c in computed)
    assert find_curves(computed) == {}
    assert find_curves(computed + ['GR_EDTC', 'TNPH_LS'], ['GR', 'NPHI']) == {'GR': 'GR_EDTC', 'NPHI': 'TNPH_LS'}


def test_standardize_unit_conversions_by_hand():
    df = pd.DataFrame({'DEPT': [1000., 1000.5], 'TNPH': [25., 30.], 'RHOZ': [2400., 2650.],
                       'DTCO': [328.084, 300.], 'DTSM': [656.168, 600.]})
    out = standardize(df, units={'DEPT': 'ft', 'TNPH': '%', 'RHOZ': 'kg/m3', 'DTCO': 'us/m', 'DTSM': 'US/M'})
    assert np.allclose(out['NPHI'], [.25, .30])                       # % / 100
    assert np.allclose(out['RHOB'], [2.4, 2.65])                      # kg/m3 / 1000
    # us/m * .3048 m/ft: 328.084 -> 100, 300 -> 91.44
    assert np.allclose(out['DT'], [100., 91.44]) and np.allclose(out['DTS'], [200., 182.88])
    # depth is renamed but never converted; its unit is kept
    assert out['DEPTH'].tolist() == [1000., 1000.5] and out.attrs['petrolib']['units']['DEPTH'] == 'ft'
    # micro sign and Greek mu
    for unit in ('µs/m', 'μs/m'):
        assert np.isclose(standardize(pd.DataFrame({'DT': [328.084]}), units={'DT': unit})['DT'].iloc[0], 100.)
    # no unit: percent and kg/m3 recognised from the median, fractions and g/cc kept
    guessed = standardize(pd.DataFrame({'NPHI': [25., 30.], 'RHOB': [2400., 2500.]}))
    kept = standardize(pd.DataFrame({'NPHI': [.25, .30], 'RHOB': [2.4, 2.5]}))
    assert np.allclose(guessed['NPHI'], [.25, .3]) and np.allclose(guessed['RHOB'], [2.4, 2.5])
    assert np.allclose(kept['NPHI'], [.25, .3]) and np.allclose(kept['RHOB'], [2.4, 2.5])
    # a second call does not convert again
    again = standardize(out)
    assert np.allclose(again['NPHI'], [.25, .30]) and np.allclose(again['DT'], [100., 91.44])
