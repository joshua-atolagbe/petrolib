'''
LogImputer: trains AutoGluon models to predict log curves and fills their missing intervals,
with one model for the whole field or one model per well

'''
from __future__ import annotations
import json
import os
import re
import shutil
from importlib.util import find_spec
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_META = 'petrolib_imputer.json'

#key and folder of the model trained on all wells
_FIELD = '__field__'


def _tabular_predictor():

    '''
    AutoGluon's TabularPredictor, imported on first use
    '''

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as e:
        raise ImportError("LogImputer needs AutoGluon: pip install 'petrolib[automl]'") from e
    return TabularPredictor


def _folder(well) -> str:

    '''
    Folder name for a well, safe on any file system (well names such as 15/9-F-1A contain slashes)
    '''

    return re.sub(r'[^\w.-]', '_', str(well))


def _plain(value):
    return value.item() if hasattr(value, 'item') else value


def _kind(source:pd.Series) -> str:

    '''
    'well', 'field' or 'mixed': which models produced a set of predictions
    '''

    kinds = set(source.dropna())
    return kinds.pop() if len(kinds) == 1 else ('mixed' if kinds else '')


def scores(measured, predicted) -> dict:

    '''
    RMSE, MAE, R2 and number of samples, over the samples where both values exist
    '''

    y, p = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'N': 0}
    ss_tot = np.sum((y - y.mean())**2)
    return {'RMSE': float(np.sqrt(np.mean((y - p)**2))), 'MAE': float(np.mean(np.abs(y - p))),
            'R2': float(1 - np.sum((y - p)**2) / ss_tot) if ss_tot > 0 else np.nan, 'N': int(len(y))}


class LogImputer:

    r'''
    Predicts missing log curves from the other logs with AutoGluon and fills the gaps

    Two ways to train, for each target curve:

    * field-wide (default): one model trained on all wells. Fills wells where the curve was
      never run, and works for new wells
    * per well (`per_well=True`): one model per well, trained only on that well's data, so each
      well keeps its own log responses. Wells with fewer than `min_samples` measured values, or
      none, are filled by a field-wide fallback model (`fallback=True`)

    Validate before trusting the filled values. Logs are correlated with depth, so random sample
    splits overstate accuracy; use whole held-out wells (`holdout_wells`, field-wide only) or a
    held-out block in the middle of each well, like a logging gap (`holdout_fraction`)

    Needs AutoGluon: pip install 'petrolib[automl]'

    Parameters
    ----------
    targets : str or list of str
        Curves to predict, e.g. ['DT', 'NPHI']

    features : list of str or dict default None
        Input curves. A list is used for every target; a dict maps each target to its own list.
        Default: all numeric columns except the targets and `well_col`. Missing feature values
        are allowed, AutoGluon handles them

    well_col : str default None
        Column with the well names. Needed for `per_well`, `holdout_wells` and per-well scores

    per_well : bool default False
        Train one model per well instead of one for the whole field

    min_samples : int default 100
        With `per_well`, the fewest measured values of a target a well needs for its own model

    fallback : bool default True
        With `per_well`, also train a field-wide model for wells without their own model

    path : str default 'petrolib_imputer'
        Folder to save the models in: <path>/<target> for a field-wide model, and
        <path>/<target>/<well> plus <path>/<target>/__field__ per well

    presets : str default 'medium_quality'
        AutoGluon preset: 'medium_quality' (fast), 'good_quality', 'high_quality' or 'best_quality'

    time_limit : float default None
        Training time limit per model, in seconds. None means no limit

    eval_metric : str default 'root_mean_squared_error'
        Metric AutoGluon optimizes

    verbosity : int default 1
        AutoGluon verbosity, 0 (silent) to 4

    **fit_kwargs
        Other arguments for TabularPredictor.fit, e.g. hyperparameters, num_cpus, excluded_model_types

    Example
    -------
    >>> from petrolib.automl import LogImputer
    >>> # one model for the field, scored on a well it never saw
    >>> imp = LogImputer('DT', features=['GR', 'RT', 'NPHI', 'RHOB'], well_col='WELL', time_limit=300)
    >>> filled = imp.fit_fill(df, holdout_wells=['15_9-F-1A'])
    >>> # one model per well, scored on a 20% block blanked in the middle of each well
    >>> imp = LogImputer('DT', features=['GR', 'RT', 'NPHI', 'RHOB'], well_col='WELL', per_well=True)
    >>> filled = imp.fit_fill(df, holdout_fraction=.2)
    >>> imp.scores_                 # RMSE, MAE and R2 per well; 'model' says which model was scored
    >>> filled['DT_MODEL']          # 'well' or 'field': the model that filled each value
    >>> imp.save(); imp = LogImputer.load('petrolib_imputer')
    '''

    def __init__(self, targets, features=None, well_col:str=None, per_well:bool=False, min_samples:int=100,
                 fallback:bool=True, path:str='petrolib_imputer', presets:str='medium_quality', time_limit:float=None,
                 eval_metric:str='root_mean_squared_error', verbosity:int=1, **fit_kwargs):

        if per_well and well_col is None:
            raise ValueError('per_well=True needs well_col, the column with the well names')
        self.targets = [targets] if isinstance(targets, str) else list(targets)
        self.features = features
        self.well_col = well_col
        self.per_well = per_well
        self.min_samples = min_samples
        self.fallback = fallback
        self.path = str(path)
        self.presets = presets
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.verbosity = verbosity
        self.fit_kwargs = fit_kwargs
        self.features_ = dict()
        self.models_ = dict()
        self.scores_ = None
        self.validation_leaderboards_ = dict()

    # ---- training --------------------------------------------------------------------------

    def _features_for(self, df:pd.DataFrame, target:str) -> list:
        if isinstance(self.features, dict):
            feats = list(self.features[target])
        elif self.features is not None:
            feats = [f for f in self.features if f != target]
        else:
            feats = [c for c in df.select_dtypes('number').columns
                     if c not in self.targets and c != self.well_col and not c.endswith(('_IMPUTED', '_PRED'))]
        missing = [f for f in feats + [target] if f not in df.columns]
        if missing:
            raise KeyError(f'Columns {missing} not in data')
        return feats

    def _train(self, data:pd.DataFrame, target:str, path:str):
        TabularPredictor = _tabular_predictor()
        kwargs = dict(self.fit_kwargs)
        if 'hyperparameters' not in kwargs and 'excluded_model_types' not in kwargs and find_spec('torch') is None:
            #neural network models need torch, which petrolib[automl] does not install
            kwargs['excluded_model_types'] = ['NN_TORCH', 'FASTAI']
        predictor = TabularPredictor(label=target, problem_type='regression', eval_metric=self.eval_metric,
                                     path=path, verbosity=self.verbosity)
        rows = data[target].notna() & np.isfinite(data[target])
        return predictor.fit(data.loc[rows, self.features_[target] + [target]], presets=self.presets,
                             time_limit=self.time_limit, **kwargs)

    def _train_models(self, data:pd.DataFrame, target:str, root:str) -> dict:

        '''
        Models of one target: {well: model} per well and/or {_FIELD: model} for the field
        '''

        if not self.per_well:
            return {_FIELD: self._train(data, target, root)}

        models = dict()
        feats = self.features_[target]
        for well, part in data.groupby(self.well_col, sort=False):
            measured = part[target].notna() & np.isfinite(part[target])
            if measured.sum() >= self.min_samples and part.loc[measured, feats].notna().any().any():
                models[well] = self._train(part, target, os.path.join(root, _folder(well)))
        if self.fallback:
            models[_FIELD] = self._train(data, target, os.path.join(root, _FIELD))
        if not models:
            raise ValueError(f'No well has {self.min_samples} measured {target} values. Lower min_samples, '
                             'or keep fallback=True to train a field-wide model')
        return models

    def _check_holdout(self, df:pd.DataFrame, holdout_wells, holdout_fraction):
        if holdout_wells is not None and holdout_fraction is not None:
            raise ValueError('Use holdout_wells or holdout_fraction, not both')
        if holdout_fraction is not None and not 0 < holdout_fraction < 1:
            raise ValueError('holdout_fraction must be between 0 and 1')
        if holdout_wells is None:
            return None
        if self.per_well:
            raise ValueError('With per_well=True each well has its own model, so hold out part of each well '
                             'with holdout_fraction instead of holdout_wells')
        assert self.well_col is not None, 'Set `well_col` to use holdout_wells'
        holdout_wells = [holdout_wells] if isinstance(holdout_wells, str) else list(holdout_wells)
        unknown = set(holdout_wells) - set(df[self.well_col].unique())
        if unknown:
            raise ValueError(f'Wells {sorted(unknown)} not in data')
        return holdout_wells

    def _holdout_mask(self, df:pd.DataFrame, target:str, holdout_wells, holdout_fraction):

        '''
        Rows kept out of training to score the models, or None. With `holdout_fraction`, a block of
        that fraction of the measured values in the middle of each well (rows in depth order)
        '''

        if holdout_wells is not None:
            return df[self.well_col].isin(holdout_wells)
        if holdout_fraction is None:
            return None
        measured = df[target].notna()
        mask = pd.Series(False, index=df.index)
        groups = df.groupby(self.well_col, sort=False).groups.values() if self.well_col else [df.index]
        for idx in groups:
            rows = [i for i in idx if measured[i]]
            k = int(round(len(rows) * holdout_fraction))
            start = (len(rows) - k) // 2
            mask.loc[rows[start:start + k]] = True
        return mask

    def fit(self, df:pd.DataFrame, holdout_wells:list=None, holdout_fraction:float=None, refit:bool=True) -> LogImputer:

        r'''
        Trains the models of each target on the samples where the target is measured

        Parameters
        ----------
        df : pd.DataFrame
            Training data of one or more wells, each well in depth order

        holdout_wells : list default None
            Wells kept out of training to score a field-wide model (needs `well_col`)

        holdout_fraction : float default None
            Fraction of each well's measured values, in one block in the middle of the well, kept out
            of training to score the models, e.g. 0.2. Works for both field-wide and per-well models

        refit : bool default True
            After scoring, train the final models again on all the data. This doubles training
            time. If False, the final models are the ones scored, which did not see the held-out data

        Scores go to `scores_` (per well, with the 'model' that was scored: 'well' or 'field') and
        AutoGluon leaderboards on the held-out data to `validation_leaderboards_`

        Returns
        -------
        self
        '''

        holdout_wells = self._check_holdout(df, holdout_wells, holdout_fraction)
        self.features_ = {target: self._features_for(df, target) for target in self.targets}
        self.models_, self.validation_leaderboards_ = dict(), dict()

        rows = list()
        for target in self.targets:
            final_root = os.path.join(self.path, target)
            test = self._holdout_mask(df, target, holdout_wells, holdout_fraction)
            if test is None:
                self.models_[target] = self._train_models(df, target, final_root)
                continue

            val_root = os.path.join(self.path, 'validation', target) if refit else final_root
            models = self._train_models(df[~test], target, val_root)
            measured = df[test & df[target].notna()]
            pred, source = self._predict_target(measured, target, models)
            rows.append({'target': target, 'well': 'all held-out', 'model': _kind(source),
                         **scores(measured[target], pred)})
            if self.well_col is not None:
                for well, part in measured.groupby(self.well_col, sort=False):
                    rows.append({'target': target, 'well': well, 'model': _kind(source[part.index]),
                                 **scores(part[target], pred[part.index])})
            self.validation_leaderboards_[target] = self._leaderboards(measured, target, models)

            if refit:
                shutil.rmtree(val_root, ignore_errors=True)
                self.models_[target] = self._train_models(df, target, final_root)
            else:
                self.models_[target] = models

        shutil.rmtree(os.path.join(self.path, 'validation'), ignore_errors=True)
        self.scores_ = pd.DataFrame(rows) if rows else None
        return self

    # ---- prediction ------------------------------------------------------------------------

    def _check_fitted(self):
        if not self.models_:
            raise RuntimeError('Call fit() or LogImputer.load() first')

    def _route(self, df:pd.DataFrame, models:dict) -> pd.Series:

        '''
        Model key of each row: the row's own well model, else the field model, else None
        '''

        if not self.per_well:
            return pd.Series(_FIELD, index=df.index, dtype=object)
        if self.well_col not in df.columns:
            raise KeyError(f"per_well=True needs the '{self.well_col}' column to pick each row's model")
        fallback = _FIELD if _FIELD in models else None
        return df[self.well_col].map(lambda w: w if w in models else fallback).astype(object)

    def _predict_target(self, df:pd.DataFrame, target:str, models:dict) -> tuple[pd.Series, pd.Series]:

        '''
        Predictions of one target, and which kind of model made each ('well' or 'field').
        NaN where all the features are missing or no model applies
        '''

        feats = self.features_[target]
        key = self._route(df, models)
        usable = df[feats].notna().any(axis=1) & key.notna()
        pred = pd.Series(np.nan, index=df.index)
        source = pd.Series(None, index=df.index, dtype=object)
        for k in pd.unique(key[usable]):
            rows = usable & (key == k)
            pred[rows] = np.asarray(models[k].predict(df.loc[rows, feats]), dtype=float)
            source[rows] = 'field' if k == _FIELD else 'well'
        return pred, source

    def _leaderboards(self, measured:pd.DataFrame, target:str, models:dict) -> pd.DataFrame:

        '''
        AutoGluon leaderboard of each model on the held-out rows it predicts, with a trained_on column
        '''

        key = self._route(measured, models)
        boards = list()
        for k in pd.unique(key.dropna()):
            board = models[k].leaderboard(measured.loc[key == k, self.features_[target] + [target]])
            board.insert(0, 'trained_on', 'field' if k == _FIELD else k)
            boards.append(board)
        return pd.concat(boards, ignore_index=True) if boards else pd.DataFrame()

    def predict(self, df:pd.DataFrame) -> pd.DataFrame:

        r'''
        Predictions of every target for all rows of `df`. NaN where all the features are missing,
        or with `per_well` where a well has no model and there is no fallback

        Returns
        -------
        DataFrame with one column per target, with the index of `df`
        '''

        self._check_fitted()
        out = pd.DataFrame(index=df.index)
        for target, models in self.models_.items():
            out[target] = self._predict_target(df, target, models)[0]
        return out

    def fill(self, df:pd.DataFrame, keep_prediction:bool=False) -> pd.DataFrame:

        r'''
        Copy of `df` with the missing values of each target filled with predictions

        Adds <TARGET>_IMPUTED (1 where the value was filled, 0 otherwise); with `per_well`,
        <TARGET>_MODEL ('well' or 'field', the model that filled the value); and if `keep_prediction`,
        <TARGET>_PRED with the prediction on every row for quality control

        Example
        -------
        >>> filled = imp.fill(df, keep_prediction=True)
        '''

        self._check_fitted()
        out = df.copy()
        for target, models in self.models_.items():
            pred, source = self._predict_target(df, target, models)
            if target not in out.columns:
                out[target] = np.nan
            gap = out[target].isna() & pred.notna()
            out.loc[gap, target] = pred[gap]
            out[f'{target}_IMPUTED'] = gap.astype(int)
            if self.per_well:
                out[f'{target}_MODEL'] = source.where(gap)
            if keep_prediction:
                out[f'{target}_PRED'] = pred
        return out

    def fit_fill(self, df:pd.DataFrame, holdout_wells:list=None, holdout_fraction:float=None, refit:bool=True,
                 keep_prediction:bool=False) -> pd.DataFrame:

        r'''
        `fit` then `fill` on the same data
        '''

        self.fit(df, holdout_wells=holdout_wells, holdout_fraction=holdout_fraction, refit=refit)
        return self.fill(df, keep_prediction=keep_prediction)

    def evaluate(self, df:pd.DataFrame, by_well:bool=False) -> pd.DataFrame:

        r'''
        Scores of the predictions against the measured values in `df`. Use data that was not
        used for training, otherwise the scores are optimistic

        Returns
        -------
        DataFrame of RMSE, MAE, R2 and N per target (and per well if `by_well`)
        '''

        pred = self.predict(df)
        rows = list()
        for target in self.models_:
            rows.append({'target': target, 'well': 'all', **scores(df[target], pred[target])})
            if by_well:
                assert self.well_col is not None, 'Set `well_col` to score by well'
                for well, part in df.groupby(self.well_col):
                    rows.append({'target': target, 'well': well, **scores(part[target], pred.loc[part.index, target])})
        return pd.DataFrame(rows)

    def _model(self, target:str, well=None):
        self._check_fitted()
        models = self.models_[target]
        key = _FIELD if well is None else well
        if key not in models:
            wells = [k for k in models if k != _FIELD]
            if well is None:
                raise ValueError(f'No field-wide {target} model; pass well= one of {wells}')
            raise ValueError(f"No {target} model for well '{well}'. Wells with a model: {wells}")
        return models[key]

    def _rows(self, df:pd.DataFrame, target:str, well=None) -> pd.DataFrame:
        if well is not None and self.well_col in df.columns:
            df = df[df[self.well_col] == well]
        return df.loc[df[target].notna(), self.features_[target] + [target]]

    def leaderboard(self, target:str, df:pd.DataFrame=None, well=None) -> pd.DataFrame:

        r'''
        AutoGluon leaderboard of the `target` model of `well` (the field-wide model if None),
        scored on the rows of `df` (of that well) if given
        '''

        model = self._model(target, well)
        return model.leaderboard(None if df is None else self._rows(df, target, well))

    def feature_importance(self, target:str, df:pd.DataFrame, well=None) -> pd.DataFrame:

        r'''
        Permutation feature importance of the `target` model of `well` (the field-wide model if
        None), on the rows of `df` (of that well) where `target` is measured
        '''

        return self._model(target, well).feature_importance(self._rows(df, target, well))

    # ---- saving ----------------------------------------------------------------------------

    def save(self):

        r'''
        Saves the settings next to the AutoGluon models in `path`, so `LogImputer.load(path)` restores them
        '''

        self._check_fitted()
        meta = {'targets': self.targets, 'features_': self.features_, 'well_col': self.well_col,
                'per_well': self.per_well, 'min_samples': self.min_samples, 'fallback': self.fallback,
                'presets': self.presets, 'time_limit': self.time_limit, 'eval_metric': self.eval_metric,
                'scores_': None if self.scores_ is None else self.scores_.to_dict(orient='records'),
                'models': {t: [[_plain(k), os.path.relpath(m.path, self.path)] for k, m in models.items()]
                           for t, models in self.models_.items()}}
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, _META), 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path:str) -> LogImputer:

        r'''
        Loads a LogImputer saved with `save`
        '''

        TabularPredictor = _tabular_predictor()
        with open(os.path.join(path, _META)) as f:
            meta = json.load(f)
        imp = cls(meta['targets'], well_col=meta['well_col'], per_well=meta['per_well'], min_samples=meta['min_samples'],
                  fallback=meta['fallback'], path=path, presets=meta['presets'], time_limit=meta['time_limit'],
                  eval_metric=meta['eval_metric'])
        imp.features_ = meta['features_']
        imp.scores_ = None if meta['scores_'] is None else pd.DataFrame(meta['scores_'])
        imp.models_ = {t: {k: TabularPredictor.load(os.path.join(str(path), p)) for k, p in pairs}
                       for t, pairs in meta['models'].items()}
        return imp

    # ---- quality control -------------------------------------------------------------------

    def plot_qc(self, df:pd.DataFrame, target:str, depth:str='DEPTH', title:str=None, figsize:tuple=(10, 10)):

        r'''
        Quality control of one target in one well: measured and predicted curves against depth,
        and a cross plot of predicted against measured with the 1:1 line

        Example
        -------
        >>> imp.plot_qc(df[df['WELL'] == '15_9-F-1A'], 'DT')
        '''

        pred = self.predict(df)[target]
        fig, (track, cross) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]})
        track.plot(df[target], df[depth], color='k', linewidth=.8, label='measured')
        track.plot(pred, df[depth], color='tab:red', linewidth=.8, label='predicted')
        track.set_ylim(df[depth].max(), df[depth].min())
        track.set_xlabel(target); track.set_ylabel(depth); track.legend(loc='best', fontsize='small')
        track.grid(linestyle=':', color='lightgrey')

        s = scores(df[target], pred)
        cross.scatter(df[target], pred, s=6, alpha=.5)
        lims = [np.nanmin([df[target].min(), pred.min()]), np.nanmax([df[target].max(), pred.max()])]
        cross.plot(lims, lims, '--', color='k', linewidth=.8)
        cross.set_xlabel(f'measured {target}'); cross.set_ylabel(f'predicted {target}')
        cross.set_title(f"RMSE {s['RMSE']:.3g}   R2 {s['R2']:.3f}   N {s['N']}")
        cross.grid(linestyle=':', color='lightgrey')
        fig.suptitle(title or f'{target} prediction')
        plt.tight_layout()
        return track, cross
