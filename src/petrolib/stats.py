'''
Python module for handling data statistics

'''

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from seaborn import histplot

class Correlation:
    
    r"""
    A correlation class for pearson and chatterjee method of statistical significance. 
    
    Parameters
    ----------

    df : pd.DataFrame
        Takes in only the dataframe

    
    """
    def __init__(self, dataframe:pd.DataFrame):
        
        self._df = dataframe
    
    
    def _chatterjee(self, x:pd.Series, y:pd.Series) -> float:
        '''
        A private method that implements chatterjee method

        Return
        ------
        correlation between two variable
        '''
        #complete pairs only, sorted by x
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        n = df.shape[0]
        y = df['y'].to_numpy()[np.argsort(df['x'].to_numpy())]
        #r = #{y_j <= y_i}, l = #{y_j >= y_i}: Chatterjee (2021) xi_n, valid with ties
        r = pd.Series(y).rank(method='max').to_numpy()
        l = pd.Series(-y).rank(method='max').to_numpy()
        sum_term = np.abs(np.diff(r)).sum()
        chatt_corr = 1 - n * sum_term / (2 * (l * (n - l)).sum())

        return chatt_corr

    def corr(self, method:str='chatterjee'):

        r'''

        Function to calculate the linear (Pearson's) and non-linear (Chatterjee's) relationships between log curves.
        Relationship between well logs are usually non-linear.

        Parameters
        ----------

        method : str, default 'chatterjee'
              Method of correlation. {'chatterjee', 'pearsonr', 'linear', 'nonlinear'}

              * 'linear' is the same as 'pearsonr'
              * 'nonlinear' is the same as 'chatterjee'
        
        Returns
        -------
        Correlation matrix of all possible log curves combination

        Example
        -------
         >>> corr = Correlation(df)
         >>> v = corr.corr(method='chatterjee) 
        
        '''

        self._method = method
        X = self._df.columns.tolist()
        Y = X.copy()

        df = pd.DataFrame(index=X, columns=Y)
        
        for i in X:
            for j in Y:
                if method == 'chatterjee' or method == 'nonlinear':
                    corr = self._chatterjee(self._df[i], self._df[j])
                    df.loc[j, i] = corr
                elif method=='pearsonr' or method == 'linear':
                    data = self._df.dropna()
                    corr, _ = pearsonr(data[i], data[j])
                    df.loc[j, i] = corr

        #convert the columns to numeric from object                    
        for column in df.columns:
            
            df[column] = df[column].astype(np.float32)

        return df


    def plot_heatmap(self, title:str='Correlation', figsize:tuple=(7, 6), annot:bool=True, cmap=None):

        r'''
        Plots the heat map of Correlation Matrix

        Parameters
        ----------
        title : str
            Title of plot
        
        figsize : tuple
            Size of plot

        annot : bool, default True
            To annotate the coefficient in the plot

        cmap : matplotlib colormap name or object, or list of colors, optional
            The mapping from data values to color space

        Example
        -------
         >>> corr = Correlation(df)
         >>> v = corr.corr(method='chatterjee) 
         >>> corr.plot_heatmap(cmap='Reds')

        '''

        from .plots import style as st
        with st.theme():
            corr = self.corr(self._method).astype(float)
            fig, ax = plt.subplots(figsize=figsize)
            n = len(corr)
            mesh = ax.pcolormesh(corr.to_numpy(), cmap=cmap or st.DIVERGING, vmin=-1, vmax=1, edgecolors=st.SURFACE,
                                 linewidth=2)
            if annot:
                for (i, j), v in np.ndenumerate(corr.to_numpy()):
                    if np.isfinite(v):
                        ax.text(j + .5, i + .5, f'{v:.2f}', ha='center', va='center', fontsize=8,
                                color=st.SURFACE if abs(v) > .55 else st.INK)
            ax.set_xticks(np.arange(n) + .5, corr.columns)
            ax.set_yticks(np.arange(n) + .5, corr.index)
            ax.set_xlim(0, n); ax.set_ylim(n, 0)
            ax.set_aspect('equal')
            ax.tick_params(length=0)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.colorbar(mesh, ax, 'Correlation')
            method = {'chatterjee': 'Chatterjee ξ of the column curve predicting the row curve',
                      'pearsonr': 'Pearson r'}.get(self._method, self._method)
            st.title(ax, title, method)
        return ax

def displayFreq(df:pd.DataFrame, *cols:tuple[str], bins:int=12, kde:bool=True,
                 figsize:tuple=(8, 10)):
    '''
    Function to plot the frequency distribution of well log curves
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of data
        
    cols : tuple[str]
        log curves to show its distribution
    
    bins : int
        Number of bins to group the data
        
    figsize : tuple
        Size of plot
        
    Returns
    -------
    Shows a plot of the frequency distribution of well log curves
        
    Example
    -------
    >>> from petrolib.stats import displayFreq
    >>> displayFreq(df, 'GR','CALI', 'COAL', 'DT', 'DT_LOG', bins=15, figsize=(20,10))
    
    '''
    from .plots import style as st
    with st.theme():
        ncols = len(cols) - (len(cols) // 2) if len(cols) > 1 else 1
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        for ax, col in zip(axes.flat, cols):
            histplot(data=df, x=col, bins=bins, kde=kde, color=st.SERIES[0], alpha=.8, edgecolor=st.SURFACE,
                     linewidth=.8, line_kws={'color': st.INK2, 'lw': 1.2}, ax=ax)
            if kde and ax.lines:
                ax.lines[-1].set_color(st.INK2)
            st.frame(ax, 'y')
            st.title(ax, col)
            ax.set_xlabel('')
            ax.set_ylabel('Samples' if ax in axes[:, 0] else '')
        for ax in list(axes.flat)[len(cols):]:
            ax.set_visible(False)
        fig.suptitle('Frequency distributions', x=axes[0, 0].get_position().x0, ha='left')
        fig.tight_layout()
    return fig
