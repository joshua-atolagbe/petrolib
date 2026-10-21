'''
petrolib.automl as users run it: load wells with petrolib.io, fill missing logs with LogImputer,
then evaluate with petrolib.petrophysics. Tests that train models need AutoGluon; the others do not
'''
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import petrolib
from petrolib.automl import LogImputer, scores
from petrolib.automl.cli import build_parser, main

try:
    import autogluon.tabular                     # noqa: F401
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

needs_autogluon = pytest.mark.skipif(not HAS_AUTOGLUON, reason="needs AutoGluon: pip install 'petrolib[automl]'")

#one fast model keeps the tests short
FAST = dict(hyperparameters={'GBM': {}}, time_limit=60, verbosity=0)
FEATURES = ['GR', 'NPHI', 'RHOB']


def with_gap(wells, well='W-2', top=1100, bot=1150, curve='DT'):
    data = wells.copy()
    gap = (data['WELL'] == well) & data['DEPTH'].between(top, bot)
    truth = data.loc[gap, curve].copy()
    data.loc[gap, curve] = np.nan
    return data, gap, truth


# ---- without AutoGluon -----------------------------------------------------------------------

def test_scores():
    s = scores([1, 2, 3, np.nan], [1, 2, 4, 5])
    assert s['N'] == 3 and np.isclose(s['RMSE'], np.sqrt(1 / 3)) and np.isclose(s['MAE'], 1 / 3)
    assert scores([np.nan], [1.])['N'] == 0


def test_reachable_from_the_petrolib_namespace():
    assert petrolib.automl.LogImputer is LogImputer


def test_settings_and_default_features(wells):
    imp = LogImputer(['DT', 'NPHI'], well_col='WELL')
    assert imp.targets == ['DT', 'NPHI'] and LogImputer('DT').targets == ['DT']
    feats = imp._features_for(wells.assign(DT_IMPUTED=0), 'DT')
    assert 'DT' not in feats and 'NPHI' not in feats and 'WELL' not in feats and 'DT_IMPUTED' not in feats
    assert {'GR', 'RHOB', 'RT'} <= set(feats)
    per_target = LogImputer(['DT', 'NPHI'], features={'DT': ['GR'], 'NPHI': ['RHOB']})
    assert per_target._features_for(wells, 'NPHI') == ['RHOB']


@pytest.mark.parametrize('kwargs, fit_kwargs, error, match', [
    (dict(), dict(holdout_wells=['W-3']), AssertionError, 'well_col'),
    (dict(well_col='WELL'), dict(holdout_wells=['NOPE']), ValueError, 'not in data'),
    (dict(features=['GR', 'MISSING']), dict(), KeyError, 'MISSING'),
])
def test_input_errors_before_training(wells, kwargs, fit_kwargs, error, match):
    with pytest.raises(error, match=match):
        LogImputer('DT', **kwargs).fit(wells, **fit_kwargs)


def test_use_before_fit():
    with pytest.raises(RuntimeError, match='fit'):
        LogImputer('DT').predict(pd.DataFrame({'GR': [1.]}))


def test_clear_message_without_autogluon(monkeypatch, wells):
    monkeypatch.setitem(sys.modules, 'autogluon.tabular', None)
    with pytest.raises(ImportError, match=r"petrolib\[automl\]"):
        LogImputer('DT', features=FEATURES).fit(wells)


def test_cli_arguments():
    args = build_parser().parse_args(['wells.las', '--targets', 'DT', 'NPHI', '--well-col', 'WELL',
                                      '--holdout-wells', 'A', 'B', '--standardize', '--no-refit'])
    assert args.targets == ['DT', 'NPHI'] and args.holdout_wells == ['A', 'B'] and args.standardize and args.no_refit
    args = build_parser().parse_args(['wells.csv', '--targets', 'DT', '--well-col', 'WELL', '--per-well',
                                      '--holdout-fraction', '.2', '--min-samples', '50', '--no-fallback'])
    assert args.per_well and args.holdout_fraction == .2 and args.min_samples == 50 and args.no_fallback


def test_per_well_argument_errors(wells):
    with pytest.raises(ValueError, match='well_col'):
        LogImputer('DT', per_well=True)
    per_well = LogImputer('DT', features=FEATURES, well_col='WELL', per_well=True)
    with pytest.raises(ValueError, match='holdout_fraction'):
        per_well.fit(wells, holdout_wells=['W-3'])
    with pytest.raises(ValueError, match='between 0 and 1'):
        per_well.fit(wells, holdout_fraction=1.5)
    with pytest.raises(ValueError, match='not both'):
        LogImputer('DT', features=FEATURES, well_col='WELL').fit(wells, holdout_wells=['W-3'], holdout_fraction=.2)


# ---- LogImputer logic with a stand-in for AutoGluon ------------------------------------------

class StandInPredictor:

    '''
    Least-squares model with the TabularPredictor methods LogImputer uses. In the synthetic wells DT
    is a linear mix of GR, NPHI and RHOB, so it is recovered almost exactly
    '''

    fits = []
    saved = {}

    def __init__(self, label, problem_type, eval_metric, path, verbosity):
        self.label, self.path = label, str(path)

    def fit(self, data, presets=None, time_limit=None, **kwargs):
        self.columns = [c for c in data.columns if c != self.label]
        X = np.c_[data[self.columns].fillna(0).to_numpy(), np.ones(len(data))]
        self.coef = np.linalg.lstsq(X, data[self.label].to_numpy(), rcond=None)[0]
        StandInPredictor.fits.append({'rows': len(data), 'path': self.path, **kwargs})
        StandInPredictor.saved[self.path] = self
        return self

    @classmethod
    def load(cls, path):
        return cls.saved[str(path)]

    def predict(self, data):
        return pd.Series(np.c_[data[self.columns].fillna(0).to_numpy(), np.ones(len(data))] @ self.coef, index=data.index)

    def leaderboard(self, data=None):
        return pd.DataFrame({'model': ['least squares']})


@pytest.fixture
def stand_in(monkeypatch):
    import petrolib.automl.imputer as imputer
    StandInPredictor.fits, StandInPredictor.saved = [], {}
    monkeypatch.setattr(imputer, '_tabular_predictor', lambda: StandInPredictor)
    monkeypatch.setattr(imputer, 'find_spec', lambda name: None)              #as if torch is not installed
    return StandInPredictor


def test_holdout_scoring_refit_and_fill(tmp_path, wells, stand_in):
    data, gap, truth = with_gap(wells)
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', path=tmp_path / 'models', time_limit=5)
    filled = imp.fit_fill(data, holdout_wells=['W-3'])

    validation, final = stand_in.fits
    n_measured = data['DT'].notna()
    assert validation['rows'] == (n_measured & (data['WELL'] != 'W-3')).sum()          #held-out well not seen
    assert Path(validation['path']).parts[-2:] == ('validation', 'DT') and Path(final['path']).parts[-2:] == ('models', 'DT')
    assert final['rows'] == n_measured.sum()                                           #refit on all wells
    assert validation['excluded_model_types'] == ['NN_TORCH', 'FASTAI']               #no torch: skip NN models
    assert imp.scores_.set_index('well').loc['W-3', 'R2'] > .999
    assert np.allclose(filled.loc[gap, 'DT'], truth) and filled['DT_IMPUTED'].sum() == gap.sum()


def test_no_refit_and_user_model_choice(tmp_path, wells, stand_in):
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', path=tmp_path / 'models', hyperparameters={'GBM': {}})
    imp.fit(wells, holdout_wells='W-3', refit=False)
    (only,) = stand_in.fits
    assert Path(only['path']).parts[-2:] == ('models', 'DT') and 'excluded_model_types' not in only
    assert only['hyperparameters'] == {'GBM': {}}


def test_per_well_models_and_field_fallback(tmp_path, wells, stand_in):
    data, gap, truth = with_gap(wells)                                                 #gap in W-2
    data.loc[data['WELL'] == 'W-3', 'DT'] = np.nan                                     #DT never run in W-3
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', per_well=True, path=tmp_path / 'models')
    filled = imp.fit_fill(data)

    measured = data['DT'].notna()
    trained = {Path(f['path']).parts[-1]: f['rows'] for f in stand_in.fits}
    assert trained == {'W-1': (measured & (data['WELL'] == 'W-1')).sum(),             #each well on its own data
                       'W-2': (measured & (data['WELL'] == 'W-2')).sum(),
                       '__field__': measured.sum()}                                    #fallback on all wells
    assert filled['DT'].notna().all() and np.allclose(filled.loc[gap, 'DT'], truth)
    assert (filled.loc[gap, 'DT_MODEL'] == 'well').all()
    assert (filled.loc[data['WELL'] == 'W-3', 'DT_MODEL'] == 'field').all()
    assert filled.loc[filled['DT_IMPUTED'] == 0, 'DT_MODEL'].isna().all()


def test_per_well_scoring_on_a_mid_well_block(tmp_path, wells, stand_in):
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', per_well=True, path=tmp_path / 'models')
    block = imp._holdout_mask(wells, 'DT', None, .2)[wells['WELL'] == 'W-1'].to_numpy()
    assert block.sum() == 80 and block[160:240].all()                                 #20% in the middle of the well

    imp.fit(wells, holdout_fraction=.2)
    s = imp.scores_.set_index('well')
    for well in ['W-1', 'W-2', 'W-3']:
        assert s.loc[well, 'N'] == 80 and s.loc[well, 'model'] == 'well' and s.loc[well, 'R2'] > .999
    assert len(stand_in.fits) == 8                                                     #3 wells + field, twice
    assert set(imp.validation_leaderboards_['DT']['trained_on']) == {'W-1', 'W-2', 'W-3'}


def test_field_model_scored_on_mid_well_blocks(tmp_path, wells, stand_in):
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', path=tmp_path / 'models')
    imp.fit(wells, holdout_fraction=.25, refit=False)
    (only,) = stand_in.fits
    assert only['rows'] == len(wells) - 300 and set(imp.scores_['model']) == {'field'}


def test_per_well_min_samples_and_no_fallback(tmp_path, wells, stand_in):
    data = wells.copy()
    w1 = data.index[data['WELL'] == 'W-1']
    data.loc[w1[50:], 'DT'] = np.nan                                                   #only 50 DT values in W-1
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', per_well=True, min_samples=100, fallback=False,
                     path=tmp_path / 'models')
    filled = imp.fit_fill(data)
    assert set(imp.models_['DT']) == {'W-2', 'W-3'}
    assert filled.loc[w1[50:], 'DT'].isna().all() and filled.loc[w1, 'DT_IMPUTED'].eq(0).all()
    with pytest.raises(ValueError, match='min_samples'):
        LogImputer('DT', features=FEATURES, well_col='WELL', per_well=True, min_samples=10_000, fallback=False,
                   path=tmp_path / 'other').fit(data)


def test_per_well_save_and_load(tmp_path, wells, stand_in):
    data, _, _ = with_gap(wells)
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', per_well=True, path=tmp_path / 'models').fit(data)
    imp.save()
    again = LogImputer.load(tmp_path / 'models')
    assert again.per_well and set(again.models_['DT']) == {'W-1', 'W-2', 'W-3', '__field__'}
    assert np.allclose(again.predict(data)['DT'], imp.predict(data)['DT'])
    assert len(again.leaderboard('DT', data, well='W-2')) == 1
    with pytest.raises(ValueError, match="No DT model for well 'W-9'"):
        again.leaderboard('DT', well='W-9')
    with pytest.raises(KeyError, match='WELL'):
        again.predict(data.drop(columns='WELL'))


# ---- with AutoGluon --------------------------------------------------------------------------

@needs_autogluon
def test_fit_fill_save_load(tmp_path, wells):
    data, gap, truth = with_gap(wells)
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', path=tmp_path / 'models', **FAST)
    filled = imp.fit_fill(data, holdout_wells=['W-3'])

    held_out = imp.scores_.set_index('well').loc['all held-out']
    assert held_out['R2'] > .9 and set(imp.scores_['well']) == {'all held-out', 'W-3'}
    assert filled['DT'].notna().all() and filled.loc[gap, 'DT_IMPUTED'].eq(1).all()
    assert filled.loc[~gap, 'DT'].equals(data.loc[~gap, 'DT'])                         # measured values kept
    assert filled.loc[~gap, 'DT_IMPUTED'].eq(0).all()
    assert scores(truth, filled.loc[gap, 'DT'])['R2'] > .9
    assert not (tmp_path / 'models' / 'validation').exists()
    assert 'DT' in imp.validation_leaderboards_

    imp.save()
    again = LogImputer.load(tmp_path / 'models')
    assert np.allclose(again.predict(data)['DT'], imp.predict(data)['DT'])
    assert again.fill(data)['DT_IMPUTED'].sum() == gap.sum()
    assert imp.evaluate(data, by_well=True)['well'].tolist() == ['all', 'W-1', 'W-2', 'W-3']


@needs_autogluon
def test_several_targets_and_diagnostics(tmp_path, wells):
    data, gap, _ = with_gap(wells)
    data.loc[data.index[::9], 'NPHI'] = np.nan
    imp = LogImputer(['DT', 'NPHI'], features={'DT': FEATURES, 'NPHI': ['GR', 'RHOB']}, well_col='WELL',
                     path=tmp_path / 'models', **FAST).fit(data)
    filled = imp.fill(data, keep_prediction=True)
    assert filled[['DT', 'NPHI']].notna().all().all()
    assert {'DT_PRED', 'NPHI_PRED', 'DT_IMPUTED', 'NPHI_IMPUTED'} <= set(filled.columns)
    assert filled['NPHI_IMPUTED'].sum() == data['NPHI'].isna().sum()
    assert imp.features_ == {'DT': FEATURES, 'NPHI': ['GR', 'RHOB']}

    assert len(imp.leaderboard('DT', data)) >= 1
    importance = imp.feature_importance('DT', data[data['WELL'] == 'W-1'])
    assert set(importance.index) == set(FEATURES)
    track, cross = imp.plot_qc(data[data['WELL'] == 'W-1'], 'DT')
    assert isinstance(track, plt.Axes) and isinstance(cross, plt.Axes)
    plt.close('all')


@needs_autogluon
def test_without_refit_models_never_see_the_held_out_well(tmp_path, wells):
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', path=tmp_path / 'models', **FAST)
    imp.fit(wells, holdout_wells='W-3', refit=False)
    assert imp.scores_.loc[imp.scores_['well'] == 'W-3', 'N'].item() == (wells['WELL'] == 'W-3').sum()
    assert (tmp_path / 'models' / 'DT').exists() and not (tmp_path / 'models' / 'validation').exists()


@needs_autogluon
def test_per_well_with_autogluon(tmp_path, wells):
    data = wells.copy()
    data.loc[data['WELL'] == 'W-3', 'DT'] = np.nan                                     #filled by the field model
    imp = LogImputer('DT', features=FEATURES, well_col='WELL', per_well=True, path=tmp_path / 'models', **FAST)
    filled = imp.fit_fill(data, holdout_fraction=.2, refit=False)
    s = imp.scores_.set_index('well')
    assert set(s.index) == {'all held-out', 'W-1', 'W-2'} and (s['R2'] > .8).all()
    assert filled['DT'].notna().all() and (filled.loc[data['WELL'] == 'W-3', 'DT_MODEL'] == 'field').all()
    imp.save()
    again = LogImputer.load(tmp_path / 'models')
    assert np.allclose(again.predict(data)['DT'], imp.predict(data)['DT'])


@needs_autogluon
def test_rows_without_features_are_not_filled(tmp_path, wells):
    data = wells.copy()
    data.loc[:4, ['DT', 'GR', 'NPHI', 'RHOB']] = np.nan
    imp = LogImputer('DT', features=FEATURES, path=tmp_path / 'models', **FAST).fit(data)
    filled = imp.fill(data)
    assert filled.loc[:4, 'DT'].isna().all() and filled.loc[:4, 'DT_IMPUTED'].eq(0).all()


# ---- the petrolib workflow: LAS files -> mnemonics -> AutoML -> petrophysics ------------------

def write_las(path, well_df, name):

    '''
    LAS file with vendor mnemonics and units, as delivered by a logging company
    '''

    lasio = pytest.importorskip('lasio')
    las = lasio.LASFile()
    las.well.WELL.value = name
    for mnemonic, values, unit in [('DEPT', well_df['DEPTH'], 'm'), ('GR_EDTC', well_df['GR'], 'gAPI'),
                                   ('RESD', well_df['RT'], 'ohm.m'), ('TNPH', well_df['NPHI'] * 100, '%'),
                                   ('RHOZ', well_df['RHOB'], 'g/cm3'), ('DTCO', well_df['DT'], 'us/ft')]:
        las.append_curve(mnemonic, values.to_numpy(), unit=unit)
    las.write(str(path), version=2.0)
    return path


@needs_autogluon
def test_petrolib_workflow_from_las_files(tmp_path, wells):
    from petrolib.io import load_well
    from petrolib.petrophysics import Quanti, sonic_porosity

    data, gap, truth = with_gap(wells)
    paths = [write_las(tmp_path / f'{w}.las', part, w) for w, part in data.groupby('WELL')]
    loaded = pd.concat([load_well(p, standardize=True).assign(WELL=p.stem) for p in paths], ignore_index=True)
    assert {'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', 'DT'} <= set(loaded.columns)
    assert np.allclose(loaded['NPHI'], wells['NPHI'], atol=1e-4)                       # TNPH in % -> fraction

    imp = LogImputer('DT', features=['GR', 'RT', 'NPHI', 'RHOB'], well_col='WELL', path=tmp_path / 'models', **FAST)
    filled = imp.fit_fill(loaded, holdout_wells=['W-3'])
    assert filled['DT'].notna().all()
    assert scores(truth.to_numpy(), filled.loc[gap.to_numpy(), 'DT'])['R2'] > .9

    w2 = filled[filled['WELL'] == 'W-2'].reset_index(drop=True)
    q = Quanti(w2, ['A'], [1060.], [1190.], [1125.], sonic='DT')
    q.vshale(method='linear')
    logs = pd.concat(q.porosity(method='sonic', dt_matrix=55.5))
    in_gap = logs['DEPTH'].between(1100, 1150)
    assert logs.loc[in_gap, 'PHIT'].notna().all()
    assert np.allclose(logs['PHIT'], sonic_porosity(logs['DT']).clip(lower=0))


@needs_autogluon
def test_cli_on_las_files(tmp_path, wells, capsys):
    data, gap, _ = with_gap(wells, well='W-1', top=1010, bot=1040)
    las = write_las(tmp_path / 'W-1.las', data[data['WELL'] == 'W-1'], 'W-1')
    main([str(las), '--targets', 'DT', '--features', 'GR', 'RT', 'NPHI', 'RHOB', '--standardize',
          '--time-limit', '60', '--verbosity', '0', '--keep-prediction', '--model-dir', str(tmp_path / 'models')])
    out = pd.read_csv(tmp_path / 'W-1_filled.csv')
    assert out['DT'].notna().all() and out['DT_IMPUTED'].sum() == gap.sum() and 'DT_PRED' in out
    assert 'values filled' in capsys.readouterr().out
    assert LogImputer.load(tmp_path / 'models').targets == ['DT']


@needs_autogluon
def test_cli_with_held_out_wells(tmp_path, wells):
    data = wells.copy()
    data.loc[data.index[::7], 'DT'] = np.nan
    src = tmp_path / 'wells.csv'
    data.to_csv(src, index=False)
    main([str(src), '--targets', 'DT', '--features', 'GR', 'NPHI', 'RHOB', '--well-col', 'WELL',
          '--holdout-wells', 'W-3', '--no-refit', '--time-limit', '60', '--verbosity', '0',
          '--model-dir', str(tmp_path / 'models')])
    out = pd.read_csv(tmp_path / 'wells_filled.csv')
    assert out['DT'].notna().all() and out['DT_IMPUTED'].sum() == data['DT'].isna().sum()
    assert (tmp_path / 'models' / 'petrolib_imputer.json').exists()
