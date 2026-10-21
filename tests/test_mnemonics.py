import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from petrolib import mnemonics as mn
from petrolib import plots
from petrolib.exceptions import MnemonicError
from petrolib.petrophysics import Quanti
from petrolib.processing import set_alias

VENDOR = {'DEPTH': 'TDEP', 'GR': 'GR_EDTC', 'RT': 'AT90', 'NPHI': 'TNPH', 'RHOB': 'RHOZ', 'DT': 'DTCO'}


def test_each_alias_belongs_to_one_type():
    seen = dict()
    for kind, aliases in mn.ALIASES.items():
        for a in aliases:
            assert a not in seen, f'{a} is an alias of {seen[a]} and {kind}'
            seen[a] = kind


@pytest.mark.parametrize('columns, expected', [
    (['DEPT', 'GR_EDTC', 'AT90', 'AT30', 'TNPH', 'RHOZ', 'HDRA', 'PEFZ', 'DTCO', 'DTSM', 'HCAL'],
     {'DEPTH': 'DEPT', 'GR': 'GR_EDTC', 'RT': 'AT90', 'RM': 'AT30', 'NPHI': 'TNPH', 'RHOB': 'RHOZ', 'DRHO': 'HDRA',
      'PEF': 'PEFZ', 'DT': 'DTCO', 'DTS': 'DTSM', 'CALI': 'HCAL'}),
    (['MD', 'GR:1', 'ILD', 'ILM', 'NPOR', 'DEN', 'AC'],
     {'DEPTH': 'MD', 'GR': 'GR:1', 'RT': 'ILD', 'RM': 'ILM', 'NPHI': 'NPOR', 'RHOB': 'DEN', 'DT': 'AC'}),
    (['depth', 'gr', 'rdep', 'rmed', 'nphi', 'rhob', 'dtc'],
     {'DEPTH': 'depth', 'GR': 'gr', 'RT': 'rdep', 'RM': 'rmed', 'NPHI': 'nphi', 'RHOB': 'rhob', 'DT': 'dtc'}),
    (['DEPTH', 'GR_CORR', 'RT_1', 'NPHI_LS', 'RHOB_FINAL'],
     {'DEPTH': 'DEPTH', 'GR': 'GR_CORR', 'RT': 'RT_1', 'NPHI': 'NPHI_LS', 'RHOB': 'RHOB_FINAL'}),
])
def test_find_curves(columns, expected):
    assert mn.find_curves(columns) == expected


def test_descriptions_are_used_last():
    descr = {'X1': 'Gamma Ray', 'X2': 'Deep Induction Resistivity', 'X3': 'Thermal Neutron Porosity',
             'X4': 'Bulk Density', 'X5': 'Density Correction', 'GR': 'Spectral gamma ray'}
    assert mn.find_curves(['GR', 'X1', 'X2', 'X3', 'X4', 'X5'], descriptions=descr) == \
        {'GR': 'GR', 'RT': 'X2', 'NPHI': 'X3', 'RHOB': 'X4', 'DRHO': 'X5'}


def test_curve_types():
    assert mn.log_kind('at90') == 'RT' and mn.log_kind('RHOZ:2') == 'RHOB' and mn.log_kind('XYZ') is None
    assert mn.is_resistivity('LLD') and mn.is_resistivity('MSFL') and not mn.is_resistivity('GR')
    assert mn.is_porosity('TNPH') and mn.is_porosity('PHIE') and not mn.is_porosity('RHOB')


def test_register_aliases(monkeypatch):
    monkeypatch.setattr(mn, 'ALIASES', copy.deepcopy(mn.ALIASES))
    monkeypatch.setattr(mn, '_LOOKUP', dict(mn._LOOKUP))
    mn.register_aliases('RT', ['RDEEP_XYZ'])
    assert mn.find_curves(['RDEEP_XYZ', 'ILD'], ['RT']) == {'RT': 'RDEEP_XYZ'}


def test_standardize_renames_and_fixes_units():
    df = pd.DataFrame({'TDEP': [1, 2, 3.], 'GR_EDTC': [50, 60, 70.], 'AT90': [10, 20, 30.], 'RLA3': [5, 6, 7.],
                       'TNPH': [20, 25, 30.], 'RHOZ': [2300, 2400, 2500.], 'DTCO': [300, 330, 360.], 'OTHER': [1, 2, 3]})
    out, mapping = mn.standardize(df, units={'DTCO': 'us/m'}, return_mapping=True)
    assert list(out.columns) == ['DEPTH', 'GR', 'RT', 'RM', 'NPHI', 'RHOB', 'DT', 'OTHER']
    assert np.allclose(out['NPHI'], [.2, .25, .3]) and np.allclose(out['RHOB'], [2.3, 2.4, 2.5])
    assert np.allclose(out['DT'], np.array([300, 330, 360]) / 3.28084)
    assert mapping['RT'] == 'AT90' and len(out.attrs['petrolib']['conversions']) == 3
    assert df.columns[0] == 'TDEP'                                                      # input unchanged

    fixed = mn.standardize(df, mapping={'RT': 'RLA3'})
    assert np.allclose(fixed['RT'], [5, 6, 7]) and 'RM' not in fixed and 'AT90' in fixed


def test_standardize_keeps_fractions():
    out = mn.standardize(pd.DataFrame({'NPHI': [.2, .3], 'RHOB': [2.3, 2.4]}))
    assert np.allclose(out['NPHI'], [.2, .3]) and out.attrs['petrolib']['conversions'] == []


def test_resolve(well):
    vendor = well.rename(columns=VENDOR)
    assert mn.resolve(vendor, 'RT') == 'AT90' and mn.resolve(vendor, 'GR') == 'GR_EDTC'
    with pytest.raises(MnemonicError, match='register_aliases'):
        mn.resolve(vendor.drop(columns='AT90'), 'RT')


def test_quanti_with_vendor_mnemonics(well):
    def run(df, **kw):
        q = Quanti(df, ['A'], [1010.], [1180.], [1095.], **kw)
        q.vshale(); q.porosity(); q.water_saturation(rw=.05); q.permeability()
        return q, pd.concat(q.flags(vsh_cutoff=.5, por_cutoff=.05, sw_cutoff=.6))

    vendor = well.rename(columns=VENDOR)
    q, logs = run(vendor, sonic='DT')
    assert q.curves == {'DEPTH': 'TDEP', 'GR': 'GR_EDTC', 'RT': 'AT90', 'NPHI': 'TNPH', 'RHOB': 'RHOZ', 'DT': 'DTCO'}
    _, reference = run(well)
    assert np.allclose(logs['SW'], reference['SW']) and np.allclose(logs['PHIE'], reference['PHIE'])
    _, from_index = run(vendor.set_index('TDEP'))                                       # depth as the index
    assert np.allclose(from_index['SW'], reference['SW'])
    with pytest.raises(MnemonicError):
        Quanti(vendor.drop(columns='AT90'), ['A'], [1010.], [1180.], [1095.])


def test_set_alias(well):
    vendor = well.rename(columns=VENDOR).set_index('TDEP')
    assert {'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', 'DT'} <= set(set_alias(vendor).columns)
    assert list(set_alias(well, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB').columns) == list(well.columns)


def test_log_plots_use_curve_types(well):
    vendor = well.rename(columns=VENDOR)
    plots.plotLog(vendor, 'TDEP', ['GR_EDTC', 'AT90', 'TNPH'], 1000, 1150)
    axes = plt.gcf().axes
    assert axes[0].get_xscale() == 'linear' and axes[1].get_xscale() == 'log' and axes[2].xaxis_inverted()
    plt.close('all')
