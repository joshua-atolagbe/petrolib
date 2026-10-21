'''
Machine learning on log data with AutoGluon: train models and fill missing log curves

Needs AutoGluon: pip install 'petrolib[automl]'. It is imported only when a model is trained or loaded

Example
-------
>>> from petrolib.automl import LogImputer
>>> imp = LogImputer(targets=['DT'], features=['GR', 'RT', 'NPHI', 'RHOB'], well_col='WELL', time_limit=300)
>>> filled = imp.fit_fill(df, holdout_wells=['WELL-3'])

From the command line: petrolib-impute wells.csv --targets DT --well-col WELL --holdout-wells WELL-3
'''

from .imputer import LogImputer, scores

__all__ = ['LogImputer', 'scores']
