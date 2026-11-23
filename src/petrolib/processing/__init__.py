'''
Preparing log data: renaming and cleaning curves, lithofacies and electrofacies

Example
-------
>>> from petrolib.processing import process_data
>>> df = process_data(df, 'GR', 'RT', 'NPHI', 'RHOB')
'''

from .cleaning import set_alias, process_data, trim
from .facies import model_facies, electrofacies

__all__ = ['set_alias', 'process_data', 'trim', 'model_facies', 'electrofacies']
