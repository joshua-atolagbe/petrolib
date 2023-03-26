'''
Python module for reservoir interpretation from plots

Function
--------
crossPlot
picketPlot
'''

import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle
from random import choice
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def crossPlot(df:pd.DataFrame, column_x:str, column_y:str, hue:str=None,
               color_code:str=None, figsize:slice=(20,7), rhob_fluid:float=1., res_name:str=None, cmap='viridis'):  

    r'''
    Plots the cross plot relationship of density against porosity on compatible scales
    to facilitate in identification of reservoir type and its fluid type.

    Reference
    ---------
    This code was initially written by Yohanes Nuwara but was modified to give
    the resulting plot a more classic and pretty view.

    Argument
    --------
    df : pd.DataFrame
        Dataframe of well

    column_x : str
        Porosity column 
    
    column_y : str
        Density column

    hue : str 
        Column to color code the scatter plot by
        
    color_code : str default None
        Color code typing. If 'num', arg `hue` must be a continuous column.
        If 'cat', argument `hue` must be a categorical column

    figsize : slice
        Size of plot

    rhob_fluid : float, default 1.0
        Fluid density
    
    res_name : str
        Reservoir name

    cmap : str
        color map 

    Returns
    -------
    A plot showing the neutron-density cross plot 

    Example
    -------
    >>> from petrolib.interp import crossPlot
    >>> crossPlot(df=df, column_x='NPHI', column_y='RHOB', res_name='RES_A', color_code='num', hue='GR') 
    
    '''

    plt.figure(figsize=figsize)
    # plt.style.use('classic')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major', linestyle=':', linewidth='1', color='black')

    lsX = np.arange(0, 0.55, 0.05)

    ssSnpX = np.empty((np.size(lsX),0), float)
    dolSnpX = np.empty((np.size(lsX),0), float)
    ssCnlX = np.empty((np.size(lsX),0), float)
    dolCnlX = np.empty((np.size(lsX),0), float)

    for i in np.nditer(lsX):
        ssSnpX = np.append(ssSnpX, np.roots([0.222, 1.021, 0.024 - i])[1])
        dolSnpX = np.append(dolSnpX, np.roots([0.6, 0.749, -0.00434 - i])[1])
        ssCnlX = np.append(ssCnlX, np.roots([0.222, 1.021, 0.039 - i])[1])
        dolCnlX = np.append(dolCnlX, np.roots([1.40, 0.389, -0.01259 - i])[1])

    densma_Ls = 2.71; densma_Ss = 2.65; densma_Dol = 2.87 #densma: density matrix

    denLs = (rhob_fluid - densma_Ls) * lsX + densma_Ls
    denSs = (rhob_fluid - densma_Ss) * lsX + densma_Ss
    denDol = (rhob_fluid - densma_Dol) * lsX + densma_Dol

    # plot the sand, limestone, and dolomite line 
    plt.plot(ssCnlX, denSs, '.-', color='blue', markersize=10, label = 'Sandstone',linewidth=2.)
    plt.plot(lsX, denLs, '.-', color='green', markersize=10, label = 'Limestone',linewidth=2.)
    plt.plot(dolCnlX, denDol, '.-', color='red', markersize=10, label = 'Dolomite',linewidth=2.)
    plt.plot([ssCnlX, lsX, dolCnlX], [denSs, denLs, denDol], '--', color='black', linewidth=2.)

    #ticks added to the opposite side of the plot
    plt.tick_params(bottom=True, top=True,
            left=True, right=True, labelbottom=True,
            labeltop=True, labelleft=True, labelright=True)

    if color_code == 'num':

        # plot data with color of the continuous variable defined (depth, vsh, etc.)

        plt.scatter(df[column_x], df[column_y], c=df[hue], cmap=cmap)

        plt.colorbar(label=hue, orientation='vertical', fraction=.057, pad=0.05)


    elif color_code == 'cat':

        # plot data with color of each unique value in the catgorical column
        sns.scatterplot(data=df, x=column_x, y=column_y, hue=hue, ec=None)

    elif color_code == None:

        #plot with no coloring
        cycol = cycle('bgrcmk')
        plt.scatter(df[column_x], df[column_y], c=choice(next(cycol)))

    plt.legend(loc='upper left')
    plt.title(f'Neutron-Density Cross Plot of RES {res_name}', size=20, pad=20)
    plt.xlim(-0.15, .60)
    plt.ylim(3, 1.3)
    plt.xlabel('$\phi (m3/m3) $',fontsize=18); plt.ylabel(r'$\rho (g/cm3)$', fontsize=18)
    plt.show()



def picketPlot(df:pd.DataFrame, rt:str='RT', por:str='NPHI', rwa:float=0.018,
                a:float=1., m:float=1.8, n:float=2., res_name:str=None, figsize:slice=(20, 8),
               hue:str=None, color_code:str=None, cmap=None):
    
    r'''

    Plot Pickett plot based on a pattern recognition approach to solving Archie’s equation. The resistivity
    and porosity logs are plotted on a logarithmic scales to evaluate formation characteristics of conventional, granular reservoirs.
    Read more here: https://wiki.seg.org/wiki/Pickett_plot

    Argument
    --------
    df : pd.DataFrame
        Dataframe

    rt : str 
        Resistivity column

    rwa : float default 0.03
        Formation water resisitivity

    a : float default, 1. 
        Turtuosity factor

    m : float default 2.
        Cementation factor

    n : float default 2.
        Saturation exponent

    res_name : str 
        Reservoir/Zone name

    figsize : slice
        Size of plot

    hue : str 
        Column to color code the scatter plot by.
        
    color_code : str default None
        Color code typing. If 'num', arg `hue` must be a continuous column
        If 'cat', argument `hue` must be a categorical column
        if None, there is no color coding 
    
    cmap : str
        Color map option 

    Example
    -------
    >>> import petrolib.interp.picketPlot
    >>> picketPlot(df, color_code='num', hue='GR', cmap='rainbow')

    >>> from petrolib.interp import picketPlot
    >>> picketPlot(df, rt='RT', por='NPHI') 
    
    '''
   
    plt.figure(figsize=figsize)
    # plt.style.use('classic')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='1.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='1', color='black')    
    plt.title(f'Pickett Plot of RES {res_name}', size=20, pad=17)
   
    if color_code == None:
        assert hue == None and cmap ==None, 'Set hue and cmap to None'
        plt.loglog(df[rt], df[por], 'bo')
        
    elif color_code=='num':
        #for continuous or numerical column
        assert hue != None and cmap !=None, 'Set hue and cmap values'
        plt.scatter(df[rt], df[por], c=df[hue], cmap=cmap)
        plt.colorbar(label=hue, orientation='vertical', fraction=.057, pad=0.05)
        plt.yscale('log'); plt.xscale('log')
        
    elif color_code=='cat':
        # for catgorical column
        assert hue != None, 'Set hue value'
        sns.scatterplot(data=df, x=rt, y=por, hue=hue, ec=None)
        plt.yscale('log'); plt.xscale('log')
        plt.legend(loc='best')
        
    plt.xlim(0.1,1000)
    plt.ylim(0.01,1.)
    plt.ylabel('$ \phi Porosity $', fontsize=18); plt.xlabel('$ \Omega m (R_{t}) $', fontsize=18)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    #ticks added to the opposite side of the plot
    plt.tick_params(bottom=True, top=True,
            left=True, right=True, labelbottom=True,
            labeltop=True, labelleft=True, labelright=True)

    
    #saturation lines
    sw = (.8,0.6,0.4,0.2, 0.1)
    phi = (0.01,1)

    rt = np.zeros((len(sw), len(phi)))

    for i in range (0, len(sw)):
        for j in range (0,len(phi)):
            rt_result = ((a*rwa)/(sw[i]**n)/(phi[j]**m))
            rt[i,j] = rt_result     

    for i in range(0,len(sw)):
        plt.plot(rt[i], phi, label=str(int(sw[i]*100))+'%')
        plt.legend(loc='best')
    
    plt.show()