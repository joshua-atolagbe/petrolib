'''
Python module for petrophysics

Classes
-------
Quanti

Methods
-------
vshale
porosity
water_saturation
permeability
flags
paySummary
report
save

'''

import pandas as pd
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle
from random import choice
import warnings
warnings.filterwarnings('ignore')


class Quanti(object):

    r"""
    Class to petrophysics workflow to evaluate any number of reservoirs of interest.
    Computes IGR/VSH, Total and Effective Porosities, Water and Hydrocarbon Saturation, Permeability 

    Attributes
    -----------
    df : pd.DataFrame
        Dataframe of data 

    zonename : list
        List of zonenames. Will be accessed from the zonation file passed into `Zonation` class

    ztop : list 
        Tops of the reservoirs. Will be accessed from the zonation file passed into `Zonation` class

    zbot : list
        Bottoms of the reservoirs. Will be accessed from the zonation file passed into `Zonation` class

    f_mids :  list
        Formation mids to help place the `zonename` in the plots. Will be accessed from the zonation file passed into `Zonation` class

    depth: str
        Depth column 

    gr : str
        Gamma ray column 

    rt : str
        Resistivity column

    nphi :  str
        Neutron porosity column

    rhob :  str
        Bulk density column

    sonic :  str default None
        Sonic column (optional)

    use_mean : bool default None
        For cutoff. Whether to use mean of GR in IGR/VSH computation or not. If None, uses either median or average value

    use_median; bool default None
        For cutoff. Whether to use median of GR in IGR/VSH computation or not. If None, uses either mean or average value

    Returns
    -------
    Displays Pay Summary results as final output

    Example
    --------
    #loading libraries/packages
    >>> from petrolib.workflow import Quanti
    >>> from petrolib.plot import Zonation
    >>> from petrolib.file_reader import load_las
    >>> from pathlib import Path

    #loading well file and zonation/tops file
    >>> well_path = Path(r"./15_9-F-1A.LAS")
    >>> contact_path = Path(r"./well tops.csv")

    >>> las, df= load_las(well_path, curves=['GR', 'RT', 'NPHI', 'RHOB'], return_las=True)

    #creating zonation class to extra info
    >>> zones = Zonation(df, path=contact_path)
    >>> ztop, zbot, zn, fm = zones()

    #creating quanti class
    >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', use_mean=True)
    """


    def __init__(self, df:pd.DataFrame, zonename:list, ztop:list, zbot:list,  f_mids:list,
                 depth:str, gr:str, rt:str, nphi:str, rhob:str, sonic:str=None,
                use_mean:bool=None,  use_median:bool=None):
        
        self._df = df
        self._depth = depth
        self._use_mean = use_mean
        self._use_median = use_median
        self._gr = gr
        self._rt = rt
        self._nphi = nphi
        self._rhob = rhob
        self._sonic = sonic
        self._ztop = ztop
        self._zbot = zbot
        self._zonename=zonename
        self._f_mids = f_mids
        
    def __call__(self):

        '''
        Returns
        -------

        method to return the GR_matrix, GR_Shale, GR_Sand and Zone_Names.
        Can be invoked if the instance of the Quanti object is called

        Example
        -------

        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', use_mean=True)
        >>> x = pp()

        '''

        return self._results

    def _filter(self):

        '''
        
        A  private method to filter/restrict the data to the zones of interest
        This also determines GR_matrix, GR_Shale and GR_Sand of the reservoirs

        Returns
        -------
        Return two objects. Dataframe containing the parameters for VSH computation and filtered Dataframe 
        
        '''

        # columns=self._df.columns.tolist()
        data = list()
        
        no = range(0,len(self._ztop)+1)
        for i, z in zip(no, self._zonename):
            d= self._df[(self._df[self._depth] >= self._ztop[i]) & (self._df[self._depth] <= self._zbot[i])]
#             d['Zone'] = z
            data.append(d)
            
            
        #get the parameters for each zone    
        self._grMatrix = list()
        self._grSand = list()
        self._grShale = list()
        self._zone = list()

        for d, z in zip(data, self._zonename):
            if self._use_mean==None and self._use_median==None:
                self._grMatrix.append(75.)
            elif self._use_mean == True and self._use_median==None:
                self._grMatrix.append(d[self._gr].mean())
            elif self._use_mean == None and self._use_median==True:
                self._grMatrix.append(d[self._gr].median())
            
            self._grShale.append(d[self._gr].max())
            self._grSand.append(d[self._gr].min())
            self._zone.append(z)
        
        #store parameter info in dataframe  
        df = {'GR_Matrix': self._grMatrix,
              'GR_Shale': self._grShale,
              'GR_Sand' : self._grSand,
              'Zone' : self._zone}

        self._results = pd.DataFrame.from_records(df)

        #return result

        return self._results, data

    def vshale(self, method:str='linear', show_plot:bool=False, palette_op:str=None, figsize:tuple=None):
        
        '''

        Computes the Volume of Shale

        Parameters
        ----------
        method : str default 'linear'
            Volume of Shale method. {'linear', 'clavier', 'larionov_ter', 'larionov_older', 'stieber_1', 'stieber_2, 'stieber_m_pliocene'}
            *Linear = Gamma Ray Index (IGR)
            *Larionov Tertiary
            *Larionov Older Rocks
            *Stieber (Miocene/Pliocene)

        show_plot : bool default False
            Display plot if True.. Plots GR, VSH and Zone track

        palette_op: str default None
            Palette option for to color code vshale plot. Check https://matplotlib.org/stable/tutorials/colors/colormaps.html

        figsize: tuple default None
            Size of plot

        Returns
        ------
        Either/Both Dataframe containing the VSH and the plot if show_plot=True  
       

        Example
        -------
        # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        # display plot only
        >>> pp.vshale(method='clavier', show_plot=True, palette_op='cubehelix', figsize=(9,12))

        # display data only 
        >>> x = pp.vshale(method='clavier')
        >>> x = pd.concat(x)
        >>> print(x)

        '''


        self._palette = palette_op
        self._v_method = method
        self._fig = figsize

        results, data = self._filter()
        new_data = list()
        
        for idx, d in enumerate(data):
            
            d['VShale'] = [(i - (results['GR_Sand'])[idx])/(results['GR_Shale'][idx]-results['GR_Sand'][idx]) for i in d[self._gr]]
    
                
            if method == 'clavier':
                d['VShale'] = 1.7 - np.sqrt((3.38 - (d['VShale'] + .7)**2))
                new_data.append(d)
                
            elif method == 'larionov_ter':
                d['VShale'] = .083*((2 ** (3.7*d['VShale'])) - 1)
                new_data.append(d)
            
            elif method == 'larionov_older':
                d['VShale'] = .33 *((2 ** (2*d['VShale'])) - 1)
                new_data.append(d)
                
            elif method == 'stieber_1':
                d['VShale'] = d['VShale'] / (2 - d['VShale'])
                new_data.append(d)
                
            elif method == 'stieber_2':
                d['VShale'] = d['VShale'] / (4 - (3 * d['VShale']))
                new_data.append(d)
            
            elif method == 'stieber_m_pliocene':
                d['VShale'] = d['VShale'] / (3 - (2 * d['VShale']))
                new_data.append(d)
            
            elif method=='linear':
                new_data.append(d)
                
        # PLOT
        data = pd.concat(new_data).reindex(np.arange(0, self._df.shape[0]))

        if show_plot == True:

            assert palette_op != None and figsize != None, f'Supply value for palette option and figsize'

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
            fig.suptitle(f'Volume of Shale', size=15, y=1.)

            span = 1
            cmap=plt.get_cmap(palette_op)
            color_index = np.arange(0, 1, span/10)

            logs = [self._gr, 'VShale']
            gr_base = 75.#(data[self._gr].max() - data[self._gr].min())/2
            for i in range(2):
                
                ax[i].plot(data[logs[i]], data[self._depth], color='black', linewidth=0.5)

                if i == 1:
                    for index in sorted(color_index):
                        index_value = (index-0.)/span
                        palette = cmap(index_value)
                        ax[i].fill_betweenx(data[self._depth], 0., data['VShale'], where=data['VShale']>=index, color=palette)
                    # ax[i].set_xlim(0, 1)

                elif i == 0:
                    ax[i].fill_betweenx(data[self._depth], gr_base, data[self._gr], where=data[self._gr]<=gr_base, facecolor='yellow', linewidth=0)
                    ax[i].fill_betweenx(data[self._depth],data[self._gr], gr_base, where=data[self._gr]>=gr_base, facecolor='brown', linewidth=0)

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(data[self._depth].min(), data[self._depth].max()); ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].tick_params(axis='x')
                ax[i].spines['top'].set_edgecolor('black')
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
                ax[i].hlines([t for t in self._ztop], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
                ax[i].hlines([b for b in self._zbot], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
            
            #formation subplot
            ax[-1].set_ylim(data[self._depth].min(), data[self._depth].max()); ax[-1].invert_yaxis()
            ax[-1].set_title('Zones', pad=45)
            ax[-1].set_xticks([])
            # ax[-1].set_yticklabels([])
            ax[-1].set_xticklabels([])
            ax[-1].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            ax[-1].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            formations = ax[-1]

            #delineating zones
            cycol = cycle('bgrycmk')
            color = [choice(next(cycol)) for i in range(len(self._zonename))]
            np.random.shuffle(color)
            for i in ax:
                for t,b, c in zip(self._ztop, self._zbot, color):
                    i.axhspan(t, b, color=c, alpha=0.3)

            #adding zone names
            for label, fm_mids in zip(self._zonename, self._f_mids):
                formations.text(0.5, fm_mids, label, rotation=0,
                        verticalalignment='center', fontweight='bold',
                        fontsize='large')
        #    
            plt.tight_layout(h_pad=1)
            fig.subplots_adjust(wspace = 0.01)
            # plt.show()
            
            return new_data

        elif show_plot==False:
            
#             assert palette_op == None and figsize == None, f'show_plot is set to {show_plot}. Set palette_op and figsize as None'

            return new_data

    def porosity(self, method:str='density', rhob_shale:float=2.4, rhob_fluid:float=1.,
                     rhob_matrix:float=2.65, fzs:float=None, show_plot:bool=False, figsize:tuple=None):

        '''
        Computes the effective and total porosities using the 'density' and Wyllie's 'sonic' method. 
        To use, must have called the `vshale` method 

        Parameters
        ----------
        method : str default 'density'
            Porosity method. {'density', 'sonic'}

        rhob_shale : float default 2.4
            Shale matrix

        rhob_fluid : float default 1.0
            Fluid density

        rhob_matrix : float default 2.65
            Matrix density

        fzs: float default None
            Flushed zone saturation for PHIE. If None, it is calculated from rhob_fluid, rhob_shale and rhob_matrix

        show_plot : bool default False
            Display plot if True.. Plots RHOB, VSH, PHIE/PHIT and Zone track

        figsize: tuple default None
            Size of plot

        Returns
        -------
        Either/Both Dataframe containing the PHIE/PHIT and the plot if show_plot=True  
       

        Example
        -------
        # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        # display plot only
        >>> pp.porosity(method='density', show_plot=True, figsize=(10, 12))

        # display data only 
        >>> y = pp.porosity(method='density')
        >>> result = pd.concat(y)
        >>> print(result)

        '''

        # self._show_plot == show_plot
        self._p_method = method 

        #calls the vshale method and concate all the dataframe representing each zones
        new_data = self.vshale(method=self._v_method, palette_op=self._palette, figsize=self._fig)

        for d in new_data:
            
            #equations from Techlog manual
            if method == 'density':
                d['PHIT'] = (rhob_matrix - d[self._rhob])/ (rhob_matrix - rhob_fluid) #total porosity
                d['PHIT'] = d['PHIT'].mask(d['PHIT']<0, 0)#mask area where phit is less than 0
                if fzs != None:
                    phi_fzs = fzs
                else:
                    phi_fzs = (rhob_matrix - rhob_shale)/ (rhob_matrix - rhob_fluid)  #flushed zone saturation
                d['PHIE'] = d['PHIT'] - (phi_fzs * d['VShale']) #effective porosity
                d['PHIE'] = d['PHIE'].mask(d['PHIE']<0, 0)#mask areas where phie is less than 0
             
            elif method == 'sonic':
                '''
                only supports Wyllie equation
                '''
                d['PHIT'] = (d[self._sonic] - 47.) / (189. - 47.)
                d['PHIT'] = d['PHIT'].mask(d['PHIT']<0, 0)#mask areas where phie is less than 0

                d['PHIE'] = 0

        data = pd.concat(new_data).reindex(np.arange(0, self._df.shape[0]))

        if show_plot == True:

            assert figsize != None, f'Supply figsize value'

            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=figsize, sharey=True)
            fig.suptitle(f'Porosity', size=15, y=1.)

            span = 1
            cmap=plt.get_cmap(self._palette) #uses palette from vshale
            color_index = np.arange(0, 1, span/10)

            #for RHOB and VSH
            logs = [self._rhob, 'VShale']
            for i in range(2):
                if i==0:
                    ax[i].plot(data[logs[i]], data[self._depth], color='blue', linewidth=1.)

                if i == 1:
                    ax[i].plot(data[logs[i]], data[self._depth], color='black', linewidth=0.3)
                    for index in sorted(color_index):
                        index_value = (index-0.)/span
                        palette = cmap(index_value)
                        ax[i].fill_betweenx(data[self._depth], 0., data['VShale'], where=data['VShale']>=index, color=palette)

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(data[self._depth].min(), data[self._depth].max()); ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].tick_params(axis='x')
                ax[i].spines['top'].set_edgecolor('black')
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
                ax[i].hlines([t for t in self._ztop], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
                ax[i].hlines([b for b in self._zbot], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)

            #for PHIT
            ax[2].minorticks_on()
            ax[2].set_xticklabels([]);ax[2].set_xticks([])
            ax[2].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
            ax[2].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
            ax[2].hlines([t for t in self._ztop], xmin=0, xmax=data["PHIT"].max(), colors='black', linestyles='solid', linewidth=1.)
            ax[2].hlines([b for b in self._zbot], xmin=0, xmax=data["PHIT"].max(), colors='black', linestyles='solid', linewidth=1.)
            nphi_ = ax[2].twiny()
            nphi_.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
            nphi_.plot(data['PHIT'], data[self._depth], color='blue', linewidth=0.5)
            nphi_.fill_betweenx(data[self._depth], data['PHIE'].max(), data['PHIT'], color='slategray', linewidth=1.)
            nphi_.set_xlim(data['PHIT'].min(), data['PHIT'].max())
            nphi_.set_ylim(data[self._depth].min(), data[self._depth].max())
            nphi_.invert_yaxis()
            nphi_.invert_xaxis()
            nphi_.tick_params(axis='x', colors='blue')
            nphi_.spines['top'].set_edgecolor('blue')
            nphi_.set_xlabel('PHIT_'+method[0].upper(), color='blue')
            nphi_.spines["top"].set_position(("axes", 1.02))
            nphi_.xaxis.set_ticks_position("top")
            nphi_.xaxis.set_label_position("top")
            nphi_.set_xticks(list(np.linspace(data['PHIT'].min(), data['PHIT'].max(), num=4)))
            
            
            #for PHIE
            phi = ax[2].twiny()
            phi.plot(data['PHIE'], data[self._depth], color='red', linewidth=0.5)
            phi.fill_betweenx(data[self._depth], data["PHIE"], data['PHIE'].max(), color='lightblue')
            phi.set_xlim(data['PHIE'].min(), data['PHIE'].max())
            phi.set_ylim(data[self._depth].min(), data[self._depth].max())
            phi.invert_yaxis()
            phi.invert_xaxis()
            phi.xaxis.label.set_color('red')
            phi.tick_params(axis='x', colors='red')
            phi.spines['top'].set_edgecolor('red')
            phi.set_xlabel('PHIE_'+method[0].upper(), color='red')
            phi.spines["top"].set_position(("axes", 1.05))
            phi.xaxis.set_ticks_position("top")
            phi.xaxis.set_label_position("top")
            phi.set_xticks(list(np.linspace(data['PHIE'].min(), data['PHIE'].max(), num=4)))
            

            #formation subplot
            ax[-1].set_ylim(data[self._depth].min(), data[self._depth].max()); ax[-1].invert_yaxis()
            ax[-1].set_title('Zones', pad=45)
            ax[-1].set_xticks([])
            # ax[-1].set_yticklabels([])
            ax[-1].set_xticklabels([])
            ax[-1].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            ax[-1].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            formations = ax[-1]

            #delineating zones
            cycol = cycle('bgrycmk')
            color = [choice(next(cycol)) for i in range(len(self._zonename))]
            np.random.shuffle(color)
            for i in ax:
                for t,b, c in zip(self._ztop, self._zbot, color):
                    i.axhspan(t, b, color=c, alpha=0.3)

            #adding zone names
            for label, fm_mids in zip(self._zonename, self._f_mids):
                formations.text(0.5, fm_mids, label, rotation=0,
                        verticalalignment='center', fontweight='bold',
                        fontsize='large')
         
            plt.tight_layout(h_pad=1)
            fig.subplots_adjust(wspace = 0.01)
            # plt.show()

            return new_data
        
        elif show_plot==False:
            
#             assert figsize == None, f'show_plot is set to {show_plot}. Set figsize as None'

            return new_data

    def water_saturation(self, method:str='archie', rw:float=0.03, a:float=1., m:float=2., n:float=2.,
                                 show_plot:bool=False, figsize:tuple=None):

                
        '''

        Computes water and hydrocarbon saturation
        To use, must have called both `vshale` and `porosity` methods

        Parameters
        ----------
        method : str default 'archie'
            Water Saturation method. {'archie', 'simmandoux'}

        rw : float default 0.03
            Formation water resisitivity

        a : float default, 1. 
            Turtuosity factor

        m : float default 2.
            Cementation factor

        n : float default 2.
            Saturation exponent

        show_plot : bool default False
            Display plot if True.. Plots RT, SW, PHIE/PHIT and Zone track

        figsize: tuple default None
            Size of plot

        Returns
        ------
        Either/Both Dataframe containing the SW, SH and the plot if show_plot=True  
       

        Example
        -------
        # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        # display plot only
        >>> pp.water_saturation(method='archie', show_plot=True, figsize=(10, 12))


        # display data only 
        >>> z = pp.water_saturation(method='archie')
        >>> result = pd.concat(z)
        >>> print(result)

        '''
        self._sw_method = method
        new_data = self.porosity(method=self._p_method)

        for d in new_data:

            # inspired from https://github.com/andymcdgeo/Petrophysics-Python-Series/blob/master/05%20-%20Petrophysical%20Calculations.ipynb

            if method == 'archie':

                d['SW'] = ((a/(d['PHIE']**m)) * (rw/d['RT']))**(a/n)
                
                #mask value greater than one to 1
                d['SW'] = d['SW'].mask(d['SW']>1, 1)
                # d['SW'] = d['SW'].mask(d['SW']<0, 0)
                d['SH'] = 1 - d['SW']
            
            elif method == 'simmandoux':
                A = (1 - d['VShale']) * a * rw / (d['PHIE'] ** m)
                B = A * d['VShale'] / (2 * 2)
                C = A / d['RT']
                
                d['SW'] = ((B **2 + C)**0.5 - B) **(2 / n)
                
                #mask value greater than one to 1
                d['SW'] = d['SW'].mask(d['SW']>1, 1)
                # d['SW'] = d['SW'].mask(d['SW']<0, 0)
                d['SH'] = 1 - d['SW']


        data = pd.concat(new_data).reindex(np.arange(0, self._df.shape[0]))

        if show_plot == True:

            assert figsize != None, f'Supply figsize value'

            logs = [self._rt, 'SW']

            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=figsize, sharey=True)
            fig.suptitle(f'Saturations', size=15, y=1.)

            for i in range(2):
                
                # for resistivity plot
                if i==0:
                    ax[i].semilogx(data[logs[i]], data[self._depth], color='red', linewidth=1, linestyle='--')
                    ax[i].hlines([t for t in self._ztop], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
                    ax[i].hlines([b for b in self._zbot], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
                
                #for non-resistivity plot
                if i==1:
                    ax[i].plot(data[logs[i]], data[self._depth], color='blue', linewidth=0.5)
                    ax[i].invert_xaxis()
                    ax[i].set_xlim(1., 0.)
                    ax[i].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid')
                    ax[i].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid')
                    ax[i].fill_betweenx(data[self._depth], data[logs[i]].max(), data[logs[i]], where=data[logs[i]]<=data[logs[i]].max(), facecolor='lightblue', interpolate=True, linewidth=0)

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(data[self._depth].min(), data[self._depth].max())
                ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].tick_params(axis='x')
                ax[i].spines['top'].set_edgecolor('black')
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
                
                
            #for PHIT
            ax[2].minorticks_on()
            ax[2].set_xticklabels([]);ax[2].set_xticks([])
            ax[2].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
            ax[2].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
            ax[2].hlines([t for t in self._ztop], xmin=data["PHIT"].min(), xmax=data["PHIT"].max(), colors='black', linestyles='solid', linewidth=1.)
            ax[2].hlines([b for b in self._zbot], xmin=data["PHIT"].min(), xmax=data["PHIT"].max(), colors='black', linestyles='solid', linewidth=1.)
            nphi_ = ax[2].twiny()
            nphi_.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
            nphi_.plot(data['PHIT'], data[self._depth], color='blue', linewidth=0.5)
            nphi_.fill_betweenx(data[self._depth], data['PHIE'].max(), data['PHIT'], color='slategray', linewidth=1.)
            nphi_.set_xlim(data['PHIT'].min(), data['PHIT'].max())
            nphi_.set_ylim(data[self._depth].min(), data[self._depth].max())
            nphi_.invert_yaxis()
            nphi_.invert_xaxis()
            nphi_.tick_params(axis='x', colors='blue')
            nphi_.spines['top'].set_edgecolor('blue')
            nphi_.set_xlabel('PHIT_'+method[0].upper(), color='blue')
            nphi_.spines["top"].set_position(("axes", 1.02))
            nphi_.xaxis.set_ticks_position("top")
            nphi_.xaxis.set_label_position("top")
            nphi_.set_xticks(list(np.linspace(data['PHIT'].min(), data['PHIT'].max(), num=5)))
            
            #for PHIE
            phi = ax[2].twiny()
            phi.plot(data['PHIE'], data[self._depth], color='red', linewidth=0.5)
            phi.fill_betweenx(data[self._depth], data["PHIE"], data['PHIE'].max(), color='lightblue')
            phi.set_xlim(data['PHIE'].min(), data['PHIE'].max())
            phi.set_ylim(data[self._depth].min(), data[self._depth].max())
            phi.invert_yaxis()
            phi.invert_xaxis()
            phi.xaxis.label.set_color('red')
            phi.tick_params(axis='x', colors='red')
            phi.spines['top'].set_edgecolor('red')
            phi.set_xlabel('PHIE_'+method[0].upper(), color='red')
            phi.spines["top"].set_position(("axes", 1.05))
            phi.xaxis.set_ticks_position("top")
            phi.xaxis.set_label_position("top")
            phi.set_xticks(list(np.linspace(data['PHIE'].min(), data['PHIE'].max(), num=5)))

            #formation subplot
            ax[-1].set_ylim(data[self._depth].min(), data[self._depth].max()); ax[-1].invert_yaxis()
            ax[-1].set_title('Zones', pad=45)
            ax[-1].set_xticks([])
            # ax[-1].set_yticklabels([])
            ax[-1].set_xticklabels([])
            ax[-1].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            ax[-1].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            formations = ax[-1]

            #delineating zones
            cycol = cycle('bgrcmk')
            color = [choice(next(cycol)) for i in range(len(self._zonename))]
            np.random.shuffle(color)
            for i in ax:
                for t,b, c in zip(self._ztop, self._zbot, color):
                    i.axhspan(t, b, color=c, alpha=0.3)

            #adding zone names
            for label, fm_mids in zip(self._zonename, self._f_mids):
                formations.text(0.5, fm_mids, label, rotation=0,
                        verticalalignment='center', fontweight='bold',
                        fontsize='large')
        #    
            plt.tight_layout(h_pad=1)
            fig.subplots_adjust(wspace = 0.01)
            # plt.show()

            return new_data

        elif show_plot==False:
            
            
#             assert figsize == None, f'show_plot is set to {show_plot}. Set figsize as None'

            return new_data

    def permeability(self, show_plot:bool=False, figsize:tuple=None):

        '''

        Computes the permeability
        To use, must have called `vshale` and `porosity` and `water_saturation` methods

        Parameters
        ----------
        show_plot : bool default False
            Display plot if True.. Plots PHIE, Permeability and Zone track

        figsize: tuple default None
            Size of plot

        Returns
        ------
        Either/Both Dataframe containing the Perm and the plot if show_plot=True  
       

        Example
        -------
        # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        # display plot only
        >>> pp.vshale(method='clavier')
        >>> pp.porosity(method='density')
        >>> pp.water_saturation(method='archie')
        >>> pp.permeability(show_plot=True, figsize=(9, 10))


        # display data only 
        >>> x = pp.vshale(method='clavier')
        >>> y = pp.porosity(method='density')
        >>> z = pp.water_saturation(method='archie')
        >>> a = pp.permeability()
        >>> result = pd.concat(a)
        >>> print(result)

        '''

        new_data = self.water_saturation(method=self._sw_method)

        for d in new_data:
            d['Perm'] = 307. + (26552*pow(d['PHIE'], 2)) - (34540 * pow(d['PHIE'] * d['SW'], 2))
            d['Perm'] = d['Perm'].mask(d['Perm']<0, 0)

        data = pd.concat(new_data).reindex(np.arange(0, self._df.shape[0]))

        if show_plot == True:

            assert figsize != None, f'Supply figsize value'

            logs = ['PHIE', 'Perm']

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
            fig.suptitle(f'Permeability', size=15, y=1.)

            for i in range(2):

                # for non-resistivity plot
                if i==1:

                    ax[i].semilogx(data[logs[i]], data[self._depth], color='blue', linewidth=1.0)
                    # ax[0].set_xlim(0.1, 1000)
                    
                if i==0:
                    ax[i].plot(data[logs[i]], data[self._depth], color='blue', linewidth=0.5)
                    ax[i].invert_xaxis()
                    ax[i].fill_betweenx(data[self._depth], data[logs[i]].max(), data[logs[i]], where=data[logs[i]]<=data[logs[i]].max(), facecolor='lightblue', interpolate=True, linewidth=0)

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(data[self._depth].min(), data[self._depth].max())
                ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].tick_params(axis='x')
                ax[i].spines['top'].set_edgecolor('black')
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
                ax[i].hlines([t for t in self._ztop], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
                ax[i].hlines([b for b in self._zbot], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
                
            #formation subplot
            ax[-1].set_ylim(data[self._depth].min(), data[self._depth].max()); ax[-1].invert_yaxis()
            ax[-1].set_title('Zones', pad=45)
            ax[-1].set_xticks([])
            # ax[-1].set_yticklabels([])
            ax[-1].set_xticklabels([])
            ax[-1].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            ax[-1].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            formations = ax[-1]

            #delineating zones
            cycol = cycle('bgrycmk')
            color = [choice(next(cycol)) for i in range(len(self._zonename))]
            np.random.shuffle(color)
            for i in ax:
                for t,b, c in zip(self._ztop, self._zbot, color):
                    i.axhspan(t, b, color=c, alpha=0.3)

            #adding zone names
            for label, fm_mids in zip(self._zonename, self._f_mids):
                formations.text(0.5, fm_mids, label, rotation=0,
                        verticalalignment='center', fontweight='bold',
                        fontsize='large')
          
            plt.tight_layout(h_pad=1)
            fig.subplots_adjust(wspace = 0.01)
            # plt.show()

            return new_data

        elif show_plot==False:
            
            
#             assert figsize == None, f'show_plot is set to {show_plot}. Set figsize as None'

            return new_data

    def flags(self, vsh_cutoff:float, por_cutoff:float, sw_cutoff:float, 
                        ref_unit:str='m', show_plot:bool=False, palette_op:str=None, figsize:tuple=None):

        '''

        Create the {ROCK, RES, PAY} flags

        To use, must have called `vshale`, `porosity`, `water_saturation` and `permeability` methods


        Parameters
        ----------

       vsh_method : float
            Volume of Shale cutoff. Applied only to ['ROCK'] flag

       por_cutoff : float
            Porosity cutoff. Applied only to the ['ROCK', 'RES'] flags

        sw_cutoff : float
            Water Saturation cutoff. Applied only to the ['ROCK', 'RES', 'PAY] flags

        ref_unit : str default 'm'
            Reference unit for measured depth. Defaults to metres
        
        show_plot : bool default False
            Display plot if True.. Plots GR, RT, VSH, SW, Perm, NPHI/RHOB, PHIE/PHIT, ['ROCK', 'RES', 'PAY] flags and Zonation track

        palette_op : str default None
             palette option for VSH coloring. Check https://matplotlib.org/stable/tutorials/colors/colormaps.html for availabel palette options

        figsize: tuple default None
            Size of plot

        Returns
        ------
        Either/Both Dataframe containing the flags and the plot if show_plot=True  
       

        Example
        -------
        # Create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        # Display plot only
        >>> pp.flags(por_cutoff=.12, vsh_cutoff=.5, sw_cutoff=0.8, show_plot=True, palette_op='cubehelix', figsize=(20, 15))

        # Display data only 
        >>> y = pp.flags(por_cutoff=.12, vsh_cutoff=.5, sw_cutoff=0.8)
        >>> result = pd.concat(y)
        >>> print(result)

        '''
    
        self._vsh_cutoff = vsh_cutoff
        self._sw_cutoff = sw_cutoff
        self._por_cutoff = por_cutoff
        self._pale = palette_op
        self._ref = ref_unit
        self.show = show_plot
        
        new_data = self.permeability()
        
        #creating  flags
        # 1 for net and 0 for gross
        for d in new_data:
            d['ROCK_NET_FLAG'] = [1 if (j < vsh_cutoff) else 0 for j in d['VShale']] 
            d['RES_NET_FLAG'] = [1 if (i > por_cutoff) or (j < vsh_cutoff) else 0 for i, j in zip(d['PHIE'], d['VShale'])] 
            d['PAY_NET_FLAG'] = [1 if (i > por_cutoff) or (j < vsh_cutoff) or (k < sw_cutoff) else 0 for i, j, k in zip(d['PHIE'], d['VShale'], d['SW'])] 

        data = pd.concat(new_data).reindex(np.arange(0, self._df.shape[0]))
        data2 = pd.concat(new_data)

        #creating plots
        if show_plot == True:

            assert palette_op != None and figsize != None, f'Supply palette and figsize values'

            fig, ax = plt.subplots(nrows=1, ncols=11, figsize=figsize)
            fig.suptitle(f'Well Layout', size=15, y=1.)

            #creating netpay flag
            facies_colors = ['#F4D03F', '#004347']; facies_labels = ['Gross Pay', 'Net Pay']

            facies_colormap = {}
            for ind, label in enumerate(facies_labels):
                facies_colormap[label] = facies_colors[ind]

            cmap_facies = colors.ListedColormap(
                    facies_colors[0 : 2], 'indexed'
                    )
            
            # data = data.set_index('DEPTH')
            #.loc[(data.DEPTH >=self._ztop[0])&(data.DEPTH <=self._zbot[-1])]
            # .loc[(data.DEPTH >= data.DEPTH.min())&(data.DEPTH <=data.DEPTH.max())]
            cluster1 = np.repeat(np.expand_dims(data['ROCK_NET_FLAG'].values, 1), 100, 1)
            cluster1 = pd.DataFrame(cluster1)
            cluster1 = cluster1.loc[(cluster1.index>=(data2[self._depth].index.min())) & (cluster1.index<=(data2[self._depth].index.max()))]
            cluster2 = np.repeat(np.expand_dims(data['RES_NET_FLAG'].values, 1), 100, 1)
            cluster2 = pd.DataFrame(cluster2)
            cluster2 = cluster2.loc[(cluster2.index>=(data2[self._depth].index.min())) & (cluster2.index<=(data2[self._depth].index.max()))]
            cluster3 = np.repeat(np.expand_dims(data['PAY_NET_FLAG'].values, 1), 100, 1)
            cluster3 = pd.DataFrame(cluster3)
            cluster3 = cluster3.loc[(cluster3.index>=(data2[self._depth].index.min())) & (cluster3.index<=(data2[self._depth].index.max()))]
          
            #logs to plot
            logs = [self._gr, self._rt, 'VShale', 'SW', 'Perm']

            #generate random colors
            cycol = cycle('bgrycmk')

            span = 1
            cmap=plt.get_cmap(self._pale)
            color_index = np.arange(0, 1, span/10)

            gr_base = 75.#(data[self._gr].max() - data[self._gr].min())/2

            #numeric plots
            for i in range(len(logs)):

                #resistivity
                if i == 1 or i ==4:
                    ax[i].semilogx(data[logs[i]], data[self._depth], color=next(cycol), linestyle='--')
                #non-resistivity
                else:
                    ax[i].plot(data[logs[i]], data[self._depth], color=next(cycol), linewidth=0.5)

                if i == 2:
                    for index in sorted(color_index):
                        index_value = (index-0.)/span
                        palette = cmap(index_value)
                        ax[i].fill_betweenx(data[self._depth], 0., data['VShale'], where=data['VShale']>=index, color=palette)
                elif i == 0:
                    ax[i].set_xlim(data[logs[i]].min(), data[logs[i]].max())
                    ax[i].fill_betweenx(data[self._depth], gr_base, data[self._gr], where=data[self._gr]<=gr_base, facecolor='yellow', linewidth=0)
                    ax[i].fill_betweenx(data[self._depth],data[self._gr], gr_base, where=data[self._gr]>=gr_base, facecolor='brown', linewidth=0)

                if i > 0:
                    ax[i].set_yticklabels([])

                if i ==3:
                    ax[i].invert_xaxis()
                    ax[i].fill_betweenx(data[self._depth], data[logs[i]].max(), data[logs[i]], where=data[logs[i]]<=data[logs[i]].max(), facecolor='lightblue', interpolate=True, linewidth=0)

                #facies = data['ROCK_NET_FLAG']
                # F = np.vstack((facies,facies)).T
                # ax[i].imshow(F, aspect='auto', extent=[0,1,max(data.DEPTH), min(data.DEPTH)])

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(data[self._depth].min(), data[self._depth].max()); ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].xaxis.label.set_color(next(cycol))
                ax[i].tick_params(axis='x', colors=next(cycol))
                ax[i].spines['top'].set_edgecolor(next(cycol))
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
                ax[i].hlines([t for t in self._ztop], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
                ax[i].hlines([b for b in self._zbot], xmin=data[logs[i]].min(), xmax=data[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)


            #for rhob
            ax[5].minorticks_on()
            ax[5].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
            ax[5].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
            ax[5].set_yticklabels([])
            ax[5].set_xticklabels([]);ax[5].set_xticks([])
            nphi_ = ax[5].twiny()
            nphi_.grid(which='major', linestyle='-', color='darkgrey')
            nphi_.plot(data[self._rhob], data[self._depth], color='red', linestyle='-', linewidth=0.5)
            nphi_.set_xlim(data[self._rhob].min(), data[self._rhob].max())
            nphi_.set_ylim(data[self._depth].min(), data[self._depth].max())
            nphi_.invert_yaxis()
            nphi_.xaxis.label.set_color('red')
            nphi_.tick_params(axis='x', colors='red')
            nphi_.spines['top'].set_edgecolor('red')
            nphi_.set_xlabel(self._rhob, color='red')
            nphi_.spines["top"].set_position(("axes", 1.02))
            nphi_.xaxis.set_ticks_position("top")
            nphi_.xaxis.set_label_position("top")
            nphi_.set_xticks(list(np.linspace(data[self._rhob].min(), data[self._rhob].max(), num=3)))
            
            #for nphi
            rhob_ = ax[5].twiny()
            rhob_.plot(data[self._nphi], data[self._depth], 'b--', linewidth=0.5)
            rhob_.invert_xaxis()
            rhob_.set_xlim(data[self._nphi].max(), data[self._nphi].min())
            rhob_.set_ylim(data[self._depth].min(), data[self._depth].max())
            rhob_.invert_yaxis()
            rhob_.xaxis.label.set_color('blue')
            rhob_.tick_params(axis='x', colors='blue')
            rhob_.spines['top'].set_edgecolor('blue')
            rhob_.set_xlabel(self._nphi, color='blue')
            rhob_.spines["top"].set_position(("axes", 1.05))
            rhob_.xaxis.set_ticks_position("top")
            rhob_.xaxis.set_label_position("top")
            rhob_.set_xticks(list(np.linspace(data[self._nphi].min(), data[self._nphi].max(), num=3)))
            
            #setting up the nphi and rhob fill
            #inspired from 
            x2=data[self._rhob]
            x1=data[self._nphi]
            
            x = np.array(rhob_.get_xlim())
            z = np.array(nphi_.get_xlim())

            nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)
            #shows both porous and non-porous zones
            rhob_.fill_betweenx(data[self._depth], x1, nz, where=x1<=nz, interpolate=True, color='yellow', linewidth=0)
            rhob_.fill_betweenx(data[self._depth], x1, nz, where=x1>=nz, interpolate=True, color='brown', linewidth=0)

            #for PHIT
            ax[6].minorticks_on()
            ax[6].set_yticklabels([])
            ax[6].set_xticklabels([]);ax[6].set_xticks([])
            ax[6].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
            ax[6].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
            ax[6].hlines([t for t in self._ztop], xmin=data["PHIT"].min(), xmax=data["PHIT"].max(), colors='black', linestyles='solid')
            ax[6].hlines([b for b in self._zbot], xmin=data["PHIT"].min(), xmax=data["PHIT"].max(), colors='black', linestyles='solid')
            phi_ = ax[6].twiny()
            phi_.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
            phi_.plot(data['PHIT'], data[self._depth], color='blue', linewidth=0.5)
            phi_.fill_betweenx(data[self._depth], data['PHIE'].max(), data['PHIT'], color='slategray', linewidth=1.)          
            phi_.set_xlim(data['PHIT'].min(), data['PHIT'].max())
            phi_.set_ylim(data[self._depth].min(), data[self._depth].max())
            phi_.invert_yaxis()
            phi_.invert_xaxis()
            phi_.tick_params(axis='x', colors='blue')
            phi_.spines['top'].set_edgecolor('blue')
            phi_.set_xlabel('PHIT', color='blue')
            phi_.spines["top"].set_position(("axes", 1.02))
            phi_.xaxis.set_ticks_position("top")
            phi_.xaxis.set_label_position("top")
            phi_.set_xticks(list(np.linspace(data['PHIT'].min(), data['PHIT'].max(), num=3)))
            
            
            #for PHIE
            phi = ax[6].twiny()
            phi.plot(data['PHIE'], data[self._depth], color='red', linewidth=0.5)
            phi.fill_betweenx(data[self._depth], data["PHIE"], data['PHIE'].max(), color='lightblue')
            phi.set_xlim(data['PHIE'].min(), data['PHIE'].max())
            phi.set_ylim(data[self._depth].min(), data[self._depth].max())
            phi.invert_yaxis()
            phi.invert_xaxis()
            phi.xaxis.label.set_color('red')
            phi.tick_params(axis='x', colors='red')
            phi.spines['top'].set_edgecolor('red')
            phi.set_xlabel('PHIE', color='red')
            phi.spines["top"].set_position(("axes", 1.05))
            phi.xaxis.set_ticks_position("top")
            phi.xaxis.set_label_position("top")
            phi.set_xticks(list(np.linspace(data['PHIE'].min(), data['PHIE'].max(), num=3)))

            #formation subplot
            ax[7].set_ylim(data[self._depth].min(), data[self._depth].max());
            ax[7].invert_yaxis()
            ax[7].set_title('Zones', pad=45)
            ax[7].set_xticks([])
            ax[7].set_yticklabels([])
            ax[7].set_xticklabels([])
            ax[7].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            ax[7].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
            formations = ax[7]


            # #delineating zones
            cycol = cycle('bgrcmk')
            color = [choice(next(cycol)) for i in range(len(self._zonename))]
            np.random.shuffle(color)
            for i in ax:
                for t,b, c in zip(self._ztop, self._zbot, color):
                    i.axhspan(t, b, color=c, alpha=0.2)
                    # pass


            #adding zone names
            for label, fm_mids in zip(self._zonename, self._f_mids):
                formations.text(0.5, fm_mids, label, rotation=0,
                        verticalalignment='center', fontweight='bold',
                        fontsize='large')
            
            #for flags
            im=ax[8].imshow(cluster1, interpolation='none', aspect='auto',
                            cmap=cmap_facies,vmin=0,vmax=1)
            im=ax[9].imshow(cluster2, interpolation='none', aspect='auto',
                            cmap=cmap_facies,vmin=0,vmax=1)
            im=ax[10].imshow(cluster3, interpolation='none', aspect='auto',
                            cmap=cmap_facies,vmin=0,vmax=1)
            
            divider = make_axes_locatable(ax[10])
            cax = divider.append_axes("right", size="20%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label((17*' ').join([
                'Gross Pay', 'Net Pay'
            ]))
            cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

            x = [8, 9, 10]
            flag_name = ['ROCK', 'RES', 'PAY']
            for i, f in zip(x, flag_name):
                ax[i].set_title(f, pad=45)
                ax[i].set_xticks([])
                ax[i].set_yticklabels([])
                ax[i].set_xticklabels([])
                # ax[i].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid')
                # ax[i].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid')
            
            plt.tight_layout(h_pad=1)
            fig.subplots_adjust(wspace = 0.01)


            return new_data
        
        elif show_plot==False:
            
            
#             assert figsize == None, f"show_plot is set to {show_plot}. Set figsize as None"


            return new_data


    def paySummary(self, name:str):

        '''

        Computes the
                *net, grossand not net thicknesses
                *net-to-gross 
                *average volume of shale
                *average porosity value
                *bulk volume of water
                *water saturation
        for each of the three flags {ROCK, RES, PAY}
        
        Parameter
        ---------
        name: str 
            Name of the well

        Return
        ------
        Displays the Pay Summary Report table

        Example
        -------
        pp.paySummary(name='15-9_F1A')

        '''
        self._name = name
        new_data = self.flags(self._por_cutoff, self._vsh_cutoff, self._sw_cutoff, show_plot=False, palette_op=self._pale)

        # attributes
        flag_name= list()  
        net = list()
        gross = list()
        top = list()
        bot = list()
        unit = list()
        zone_name = list()
        net_to_gross = list()
        wellname = list()
        avg_bvw = list()
        # por_th = list()
        # hcpor_th = list()
        avg_shale = list()
        avg_por = list()
        avg_sw = list()
        not_net = list()

        np.random.seed(2)

        for d, i in zip(new_data, self._zonename):
            # np.random.seed(2)
            # top and bottom and unit
            top_ = d[self._depth].min()
            bot_ = d[self._depth].max()
            top.append([top_, top_, top_])
            bot.append([bot_, bot_, bot_])
            unit.append([self._ref, self._ref, self._ref])

            #ntg
            ntg_rock = d['ROCK_NET_FLAG'].sum()/d.shape[0]
            ntg_res = d['RES_NET_FLAG'].sum()/d.shape[0]
            ntg_pay = d['PAY_NET_FLAG'].sum()/d.shape[0]
            net_to_gross.append([ntg_rock, ntg_res, ntg_pay])

            #gross
            gross_ = d[self._depth].max() - d[self._depth].min()
            gross.append([gross_, gross_, gross_])

            #net
            net_rock = ntg_rock * gross_
            net_res = ntg_res * gross_
            net_pay = ntg_pay * gross_
            net.append([net_rock, net_res, net_pay])

            #not net
            not_net.append([gross_-net_rock, gross_-net_res, gross_-net_pay])

            #zonename and flag name
            flag_name.append(['ROCK', 'RES', 'PAY'])
            zone_name.append([i, i, i])


            #average volume of shale, water saturationa and porosity
            rock_filter = d[d['ROCK_NET_FLAG'] == 1]
            res_filter = d[d['RES_NET_FLAG'] == 1]
            pay_filter = d[d['PAY_NET_FLAG'] == 1]
            
            #avg shale
            avg_shale_rock = rock_filter['VShale'].mean() 
            avg_shale_res = res_filter['VShale'].mean()
            avg_shale_pay = pay_filter['VShale'].mean()
            avg_shale.append([avg_shale_rock, avg_shale_res, avg_shale_pay])

            #avg porosity
            avg_por_rock = rock_filter['PHIE'].mean() 
            avg_por_res = res_filter['PHIE'].mean()
            avg_por_pay = pay_filter['PHIE'].mean()
            avg_por.append([avg_por_rock, avg_por_res, avg_por_pay])

            #avg water saturation
            avg_water_rock = rock_filter['SW'].mean() 
            avg_water_res = res_filter['SW'].mean()
            avg_water_pay = pay_filter['SW'].mean()
            avg_sw.append([avg_water_rock, avg_water_res, avg_water_pay])

            #avg bulk volume of water
            avg_bulk_rock = (rock_filter['SW'] * rock_filter['PHIE']).mean()
            avg_bulk_res = (res_filter['SW'] * res_filter['PHIE']).mean()
            avg_bulk_pay = (pay_filter['SW'] * pay_filter['PHIE']).mean()
            avg_bvw.append([avg_bulk_rock, avg_bulk_res, avg_bulk_pay])

            #wellname
            wellname.append([name, name, name])
           
        #store info in dataframe  
        df = {  
                'Well': wellname,
                'Zones' : zone_name,
                'Flag Name': flag_name,
                'Top': top,
                'Bottom': bot,
                'Unit': unit,
                'Gross': gross,
                'Net': net,
                'Not Net': not_net,
                'NTG': net_to_gross,
                'BVW': avg_bvw,
                'Average VShale': avg_shale,
                'Average Porosity': avg_por,
                'Average Water Saturation': avg_sw
            }

        summary = pd.DataFrame(df).apply(pd.Series.explode).reset_index(drop=True)

        n = len(summary.columns)

        def highlight(x):

            '''
            highlighting cell colors by flag type
            '''
            if x['Flag Name']== 'ROCK':
                return ["background-color: yellow"]*n
            elif x['Flag Name'] == 'RES':
                return ["background-color: green"]*n
            else:
                return ["background-color: red"]*n


       # ref https://stackoverflow.com/questions/57958432/how-to-add-table-title-in-python-preferably-with-pandas
       
        self._styles = [dict(selector="caption",
                       props=[("text-align", "center"),
                              ("font-size", "150%"),
                              ("color", 'black')])]
        summary = summary.style.apply(highlight, axis = 1).set_caption(f"Workflow Table Result MD for {name.upper()}").set_table_styles(self._styles)


        return summary
        

    def report(self):

        '''

        Displays the methods used in each parameter estimations and cutoff used for flagging

        '''

        df = {
            'VShale': self._v_method.title(),
            'Porosity': self._p_method.title(),
            'Water Saturation': self._sw_method.title(),
            'VSH Cutoff':self._vsh_cutoff,
            'SH Cutoff': self._sw_cutoff,
            'PHI Cutoff': self._por_cutoff
        }

        df = pd.DataFrame.from_records(df, index=np.arange(0, 1)).T.rename({0:''}, axis=1).style.set_caption('Methods and Cutoffs').set_table_styles(self._styles)

        return df



    def save(self, file_name:str):

        '''
        A method to save the pay summary results into a excel file

        Argument
        --------
        file_name : str
            name of to save the pay summary report as
        
        '''

        c = self.paySummary(self._name)

        c.to_excel(file_name+'.xlsx', index=False)
