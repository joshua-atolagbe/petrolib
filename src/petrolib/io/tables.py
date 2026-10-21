'''
Loading of CSV, TSV and Excel tables

'''
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_table(file:Path|str, curves:list[str]=None, delimiter:str=None, header:int|list[int]|str='infer', 
                    skiprows:list|int=None, sheet_name:int|str|list=None) -> pd.DataFrame:

    r"""
    Function to load a table data, either csv, tsv, or excel file

    Arguments
    ---------

    file : pathlib.Path or str
        Filename or filepath specifying the file

    curves : list or tuple, optional
        If specified, returns only dataframe containing the log curves specified
        If not, all available logs are imported

    delimiter : str, default ','
        Delimiter to use

    header : int, list of int, default 'infer'
        Row number(s) to use as the column names, and the start of the
        data.  Default behavior is to infer the column names. See official pandas doc for more..

    skiprows : list, int , optional
        Line numbers to skip (0-indexed) or number of lines to skip (int)
        at the start of the file

    sheet_name : str, int, list, default None
        Strings are used for sheet names. Integers are used in zero-indexed
        sheet positions.
        
        Available cases:

            * 0 : 1st sheet as a `DataFrame`
            * 1: 2nd sheet as a `DataFrame`
            * "Sheet1" : Load sheet with name "Sheet1"
            * [0, 1, "Sheet5"]: Load first, second and sheet named "Sheet5" as a dict of `DataFrame`
            * defaults to None: All sheets.

        See help(pd.read_excel) for more

    Example
    -------
    >>> well_path = Path(r"C:\Users\USER\Documents\petrolib\test\petrolib\petrolib\15_9-19.csv")

    >>> #loads all logs
    >>> df = load_table(well_path)
    
    >>> #loads specific
    >>> df = load_table(well_path, ['GR', 'RT'], skiprows=[1])
    >>> df

    """


    if type(file) == Path:

        if file.exists():
            pass

        else:
            raise FileNotFoundError('File Path does not exists : {file}')

    file = str(file)

    if file.endswith('.csv'):
        
        # last_four = file[-4:].lower()
        df = pd.read_csv(file, delimiter=delimiter, header=header, skiprows=skiprows)

        if curves != None:

            df = df.filter(curves, axis=1)
            return df
        
        else:
            return df

        
    elif file.endswith('.xls') or file.endswith('.xlsx'):

        df = pd.read_excel(file, sheet_name=sheet_name, skiprows=skiprows)

        if curves != None:

            df = df.filter(curves, axis=1)
            return df
        
        else:
            return df
