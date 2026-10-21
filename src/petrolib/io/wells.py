'''
Loading of well data from LAS, DLIS, CSV or Excel files into one DataFrame

'''
from __future__ import annotations
from pathlib import Path
import pandas as pd

from ..mnemonics import standardize as _standardize, find_curves, resolve


def load_well(file:Path|str, curves:list=None, standardize:bool=False, **kwargs) -> pd.DataFrame:

    r'''
    Loads a well from a LAS, DLIS, CSV, TXT or Excel file, with depth as a column

    Parameters
    ----------
    file : pathlib.Path or str
        .las, .dlis, .csv, .txt, .tsv, .xls or .xlsx file

    curves : list of str default None
        Curves to keep, besides depth. Names may be standard types (e.g. 'RT'), which are
        matched to the file's mnemonics (e.g. 'AT90'). Default: all curves

    standardize : bool default False
        Rename recognized curves to standard names and fix units, see `petrolib.mnemonics.standardize`

    **kwargs
        Passed to `load_dlis` (frame, logical_file, depth_unit) or to pandas for tables

    Returns
    -------
    DataFrame. Units and curve descriptions of LAS and DLIS files are in `df.attrs['petrolib']`

    Example
    -------
    >>> from petrolib.io import load_well
    >>> df = load_well('15_9-19.dlis', curves=['GR', 'RT', 'NPHI', 'RHOB'], standardize=True)
    '''

    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f'{file} path does not exists.')
    suffix = path.suffix.lower()

    if suffix == '.las':
        import lasio
        las = lasio.read(str(path), **kwargs)
        df = las.df().reset_index()
        meta = {'source': str(path), 'well': str(las.well.WELL.value) if 'WELL' in las.well else '',
                'units': {c.mnemonic: c.unit for c in las.curves},
                'descriptions': {c.mnemonic: c.descr for c in las.curves}}
    elif suffix == '.dlis':
        from .dlis import load_dlis
        df = load_dlis(path, **kwargs)
        meta = df.attrs['petrolib']
    elif suffix in ('.xls', '.xlsx'):
        df, meta = pd.read_excel(path, **kwargs), {'source': str(path)}
    elif suffix in ('.csv', '.txt', '.tsv'):
        sep = kwargs.pop('sep', ',' if suffix == '.csv' else None)
        df = pd.read_csv(path, sep=sep, engine=None if sep else 'python', **kwargs)
        meta = {'source': str(path)}
    else:
        raise ValueError(f'Unsupported file type {suffix}. Use LAS, DLIS, CSV, TXT or Excel')

    df.attrs['petrolib'] = meta
    if standardize:
        df = _standardize(df)

    if curves is not None:
        depth = find_curves(df, ['DEPTH']).get('DEPTH', df.columns[0])
        keep = [depth] + [resolve(df, c) for c in curves if c != depth]
        attrs = df.attrs
        df = df[list(dict.fromkeys(keep))]
        df.attrs = attrs
    return df
