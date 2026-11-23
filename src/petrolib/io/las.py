'''
Loading of LAS files

'''
from __future__ import annotations
from pathlib import Path
import lasio
import pandas as pd
from ..exceptions import MnemonicError

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

    >>> #return both dataframe containing only ['GR','RT', 'RHOB'] curves and the lasio object
    >>> df, las = load_las(well_path, return_csv=True, curves=['GR', 'RT', 'RHOB'])

    >>> #return only LAS object
    >>> las = load_las(well_path)

    '''

    try:

        if type(file) == Path:

            assert file.exists(), 'File does not exists'      

    except:

        raise FileNotFoundError(f'{file} path does not exists.')


    las = lasio.read(str(file))

    assert type(las) == lasio.las.LASFile, 'Fucntion can only read a LAS file'

    if return_csv == True:

        df = las.df()

        if curves != None:

            try:
            
                for i in curves:
                    
                    assert i in df.columns, f"'{i}' not found in log data."
                    
            except:
                
                raise MnemonicError('Check data. A log curve mnemonic was not passed correctly.')
        
            df = df.filter(curves, axis=1)
            
            return df, las
    
        elif curves == None:
            
            return df, las

    else:
        
        return las
