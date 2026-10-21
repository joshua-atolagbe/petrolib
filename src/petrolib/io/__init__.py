'''
Loading of well data (LAS, DLIS and table files, well tops) and writing pay summaries to Excel

Curve mnemonics of any vendor can be recognized and renamed to standard names with
`find_curves` and `standardize` (from `petrolib.mnemonics`)

Example
-------
>>> from petrolib.io import load_well, Zonation
>>> df = load_well('15_9-19.dlis', standardize=True)      # LAS, DLIS, CSV or Excel
'''

from .las import load_las
from .tables import load_table
from .dlis import load_dlis, dlis_summary
from .wells import load_well
from .tops import Zonation
from .excel import write_pay_summary
from ..mnemonics import find_curves, standardize

__all__ = ['load_well', 'load_las', 'load_dlis', 'dlis_summary', 'load_table', 'Zonation', 'write_pay_summary',
           'find_curves', 'standardize']
