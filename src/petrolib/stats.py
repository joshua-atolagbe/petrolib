'''
Python module for handling data statistics

'''

from __future__ import annotations

from warnings import filterwarnings
filterwarnings('ignore')
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from itertools import cycle
from random import choice
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
        df = pd.DataFrame()
        df['x_rk'] = x.rank()
        df['y_rk'] = y.rank()
        df = df.sort_values('x_rk')
        sum_term = df['y_rk'].diff().abs().sum()
        chatt_corr = (1 - 3 * sum_term / (pow(df.shape[0], 2) - 1))

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
                    df[i][j] = corr
                elif method=='pearsonr' or method == 'linear':
                    self._df = self._df.dropna()
                    corr, _ = pearsonr(self._df[i], self._df[j])
                    df[i][j] = corr

        #convert the columns to numeric from object                    
        for column in df.columns:
            
            df[column] = df[column].astype(np.float32)

        return df


    def plot_heatmap(self, title:str='Correlation Heatmap', figsize:tuple=(12, 7), annot:bool=True, cmap=None):

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

        corr = self.corr(self._method)
        plt.rcParams['figure.figsize'] = figsize
        plt.title(title)
        sns.heatmap(corr, annot=annot, vmin=-1, vmax=1, cmap=cmap)

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
    #randomnly generated colors 
    cycol = cycle('bgrcmk')
    color = [choice(next(cycol)) for i in range(len(cols))]
    np.random.shuffle(color)

    plt.subplots(nrows=1, ncols=len(cols), figsize=figsize)
    plt.suptitle(f'Frequency Distribution', fontsize=20)

    for i, col in enumerate(cols):
        plt.subplot(2, len(cols)-(len(cols)//2), i+1)
        histplot(data=df, x=col, bins=bins, kde=kde, color=color[i])
        plt.grid(which='major', linestyle=':', linewidth='1', color='lightgray')
        plt.title('Histogram of ' + col)
        plt.ylabel('Frequency')# Set text for y axis
        plt.xlabel(col.upper()) #set text for x axis