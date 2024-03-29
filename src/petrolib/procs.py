'''
Python module for data processing and lithofacies modelling

'''
from __future__ import annotations

from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np
from typing import Optional


def set_alias(df:pd.DataFrame, DEPTH:str, GR:str, RT:str, NPHI:str, RHOB:str, DT:Optional[str]=None) -> pd.DataFrame:

    r'''
    Function to rename the log curves in order to maintain petrophysics conventions

    Parameters
    ---------

    df : pd.DataFrame
        dataframe object
    
    DEPTH : str
        Depth column

    GR : str
        Gamma ray column 

    RT : str
        Resistivity column

    NPHI :  str
        Neutron porosity column

    RHOB :  str
        Bulk density column

    DT :  str, default None
        Sonic column (optional)

    Returns
    -------
    Returns data of renamed log curves

    Example
    -------
    >>> df = set_alias(df, 'DEPT', 'GR','RES', 'NPHI', 'RHOB')
    >>> print(df.columns)
    >>> ['DEPTH', 'GR', 'RT', 'NPHI', 'RHOB']
    '''

    if DT != None:
        dataframe = df.rename({DEPTH:'DEPTH', GR:'GR', RT:'RT', 
                        NPHI:'NPHI', RHOB:'RHOB', DT:'DT'}, axis=1)

    else:
        dataframe = df.rename({DEPTH:'DEPTH', GR:'GR', RT:'RT', 
                        NPHI:'NPHI', RHOB:'RHOB'}, axis=1)

    return dataframe


def process_data(df:pd.DataFrame, gr:str, rt:str, nphi:str, rhob:str, dt:Optional[str]=None, trim:str='both') -> pd.DataFrame:

    r'''
    Function to preprocess data before beginning petrophysics workflow.
    This processing workflow uses conventional values for the log curves. 
    To use user-defined preprocessing method , refer to the `petrolib.data.procs.trim()`

    Arguments
    ---------

    df : pd.DataFrame
        dataframe object
    
    gr : str
        Gamma ray column 

    rt : str
        Resistivity column

    nphi :  str
        Neutron porosity column

    rhob :  str
        Bulk density column

    sonic :  str, default None
        Sonic column (optional)

    trim : str default 'both'
        Conditions for trim arbitrary values 
        * 'max' : to trim values higher than conventional maximum values 
        * 'min' : to trim values lower than conventional lower values
        * default 'both' : to trim both lower and higher values to conventional high and lower values 
    
    Returns
    -------

    A new copy of dataframe containing processed data

    Example
    -------
    >>> df = process_data(df, 'GR', 'RT', 'NPHI', 'RHOB')
    >>> df.describe()
                |DEPTH  |  GR	|   RT	|  NPHI	 |  RHOB|
                +-------+-------+-------+--------+------+
          count	| 35361	| 34671 | 34211 | 10524	 |10551 |
         -------+-------+-------+-------+--------+------+
           mean	| 1913.9| 56.97	|  1.95	|  0.17	 | 2.48 |
         -------+-------+-------+-------+--------+------+
            min	|  145.9| 0.15	|  0.2	|  0.03  | 1.98 |
          ------+-------+-------+-------+--------+------+
            max | 3681.9|  200	| 2000	|  0.45	 |  2.93|
           ----------------------------------------------
    '''
    
    df.replace(-999.00, np.nan, inplace=True)
    
    data = df.copy()
    
    if trim == 'max':
    
        data[gr] = data[gr].mask(data[gr]>150, 150)
        data[rt] = data[rt].mask(data[rt]>2000, 2000)
        data[nphi] = data[nphi].mask(data[nphi]>.45, .45)
        data[rhob] = data[rhob].mask(data[rhob]>2.95, 2.95)

        if dt != None:
            data[dt] = data[dt].mask(data[dt] > 150, 150)
        
    elif trim == 'min':
        data[gr] = data[gr].mask(data[gr]<0, 0)
        data[rt] = data[rt].mask(data[rt]<0.2, 0.2)
        data[nphi] = data[nphi].mask(data[nphi]<-.15, -.15)
        data[rhob] = data[rhob].mask(data[rhob]<1.95, 1.95)

        if dt != None:
            data[dt] = data[dt].mask(data[dt] < 40, 40)
        
    elif trim == 'both':
        
        data[gr] = data[gr].mask(data[gr]>150, 150)
        data[gr] = data[gr].mask(data[gr]<0, 0)
        data[rt] = data[rt].mask(data[rt]>2000, 2000)
        data[rt] = data[rt].mask(data[rt]<0.2, 0.2)
        data[nphi] = data[nphi].mask(data[nphi]>.45, .45)
        data[nphi] = data[nphi].mask(data[nphi]<-.15, -.15)
        data[rhob] = data[rhob].mask(data[rhob]>2.95, 2.95)
        data[rhob] = data[rhob].mask(data[rhob]<1.95, 1.95)

        if dt != None:
            data[dt] = data[dt].mask(data[dt] > 200, 200)
            data[dt] = data[dt].mask(data[dt] < 40, 40)
    
    return data

def trim(df:pd.DataFrame, col:str, lower:int|float, upper:int|float):
    
    '''
    Function to preprocess data by trimming arbitrary values 

    Parameters
    ---------
    df : pd.DataFrame
    	Dataframe 
    
    col : str
    	Log curve to trim its values
    	
    lower : int or float
    	Lower limit or minimum value
    
    upper : int or float
    	Upper limit or maximum value
    
    Returns
    -------
    Dataframe with user defined log limits
    
    Example
    -------
    >>> trim(df, 'GR', lower=0, upper=200)
    '''

    assert col in df.columns, f'{col} not in dataframe.'

    df[col] = df[col].mask(df[col]<lower, lower)
    df[col] = df[col].mask(df[col]>upper, upper)

    return df

def model_facies(df:pd.DataFrame, gr:str, env:str='SS') -> pd.DataFrame:

    '''
    Models lithofacies from Gamma ray log specific to a particular environment

    Parameters
    ----------

    df : pd.DataFrame
        dataframe 
    
    gr : str
        Gamma ray log column

    env : str 
        Environment type. Either siliciclastic or carbonate
        * 'SS' for Siliclastic (Shale and Sandstone) environment
        * 'CO' for carbonate (Anhydrite, Limestone, Dolomite, Sandstone, Shale) environment 

    Example
    -------
    >>> from petrolib.interp import model_facies
    >>> model_facies(df, gr='GR', env='SS')
    
    '''

    litho = list()

    if env == 'SS':

        for i in df[gr]:

            if i < 75.0:
                litho.append('Sandstone')
            else:
                litho.append('Shale')
    
    elif env == 'CO':

        for i in df[gr]:

            if i>= 8. and i<11.:
                litho.append('Anhydrite')
            
            elif i>=10 and i<15.:
                litho.append('Limestone')

            elif i>=15 and i<=40.:
                litho.append('Dolomite')

            elif i>40 and i<=75:
                litho.append('Sandstone')

            elif i > 75.:
                litho.append('Shale')

    df['litho'] = litho

    return df
