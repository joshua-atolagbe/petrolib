'''
Lithofacies modelling from logs

'''
from __future__ import annotations
import pandas as pd


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
    >>> from petrolib.processing import model_facies
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


def electrofacies(df:pd.DataFrame, logs:list, n_facies:int=4, seed:int=0) -> pd.Series:

    r'''
    Unsupervised electrofacies: k-means clustering of the standardized logs (k-means++ start)

    Resistivity curves (recognized from their mnemonics) are clustered on log10. Facies are numbered
    0 to n_facies-1 in increasing order of the mean of the first log, so with GR first, facies 0
    is the cleanest. Interpret each facies with cross plots and core before naming it

    Parameters
    ----------
    df : pd.DataFrame
        Log data

    logs : list of str
        Curves to cluster on, e.g. ['GR', 'RT', 'NPHI', 'RHOB']

    n_facies : int default 4
        Number of facies

    seed : int default 0
        Random seed, for repeatable facies

    Returns
    -------
    Series of facies numbers, NaN where any of the logs is missing

    Example
    -------
    >>> df['EFAC'] = electrofacies(df, ['GR', 'RT', 'NPHI', 'RHOB'], n_facies=5)
    '''

    import inspect
    import numpy as np
    from scipy.cluster.vq import kmeans2
    from ..mnemonics import is_resistivity

    X = df[logs].astype(float).copy()
    for c in logs:
        if is_resistivity(c):
            X[c] = np.log10(X[c].clip(lower=1e-3))
    ok = X.notna().all(axis=1)
    Z = ((X[ok] - X[ok].mean()) / X[ok].std(ddof=0)).to_numpy()
    #scipy renamed seed to rng in 1.15
    key = 'rng' if 'rng' in inspect.signature(kmeans2).parameters else 'seed'
    _, labels = kmeans2(Z, n_facies, minit='++', **{key: seed})
    order = X.loc[ok, logs[0]].groupby(labels).mean().sort_values().index
    rank = {label: i for i, label in enumerate(order)}
    facies = pd.Series(np.nan, index=df.index)
    facies[ok] = [rank[label] for label in labels]
    return facies
