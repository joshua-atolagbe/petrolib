'''
A Python module for handling th importation of files into code environment

Functions
---------
load_las
load_table
'''

from pathlib import Path
from warnings import filterwarnings
filterwarnings('ignore')
import lasio
import pandas as pd

def load_las(file:Path|str, return_csv:bool=False, curves:list|tuple=None) -> lasio.las.LASFile|pd.DataFrame:

    '''
    Function to read LAS file

    Arguments
    ---------

    file : pathlib.Path or str
        Filename or filepath specifying the LAS file

    return_csv : bool default False
        If True, both dataframe and LAS object are returned. 
        If False, returns only LAS object

    curves : list or tuple, optional
        If specified, returns only dataframe containing the log curves specified.
        If not, all available logs are imported


    Returns
    -------
    Returns either LAS and/or dataframe object of the well data

    Example
    -------

    #return both dataframe containing only ['GR','RT', 'RHOB'] curves and the lasio object
    >>> df, las = load_las(well_path, return_csv=True, curves=['GR', 'RT', 'RHOB'])

    #return only LAS object
    >>> las = load_las(well_path)

    '''

    try:

        if type(file) == Path:

            exist = file.exists()      

    except:

        raise FileNotFoundError(f'{file} path does not exists.')

        # raise TypeError(f'{file} is not a LAS file')

    las = lasio.read(str(file))

    assert type(las) == lasio.las.LASFile, 'Fucntion can only read a LAS file'

    if return_csv == True:

        df = las.df()

        if curves != None:

            try:
            
                for i in curves:
                    
                    assert i in df.columns, f"'{i}' not found in data."
                    
            except:
                
                raise AttributeError('Check data. Log curve mnemonic not passed correctly.')
        
            df = df.filter(curves, axis=1)
            
            return df, las
    
        elif curves == None:
            
            return df, las

    else:
        
        return las


def load_table(file:Path|str, curves:list[str]=None, delimiter:str=None, header:int|list[int]='infer', 
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

    #loads all logs
    >>> df = load_table(well_path)
    
    #loads specific
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
