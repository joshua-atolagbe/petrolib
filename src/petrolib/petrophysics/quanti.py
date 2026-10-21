'''
Quanti: zone-by-zone petrophysics workflow with log plots and pay summary

'''
import pandas as pd
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle
from random import choice

from .shale import gamma_ray_index, vshale as _vshale
from .porosity import (density_porosity, neutron_density_porosity, shale_corrected_nd_porosity,
                       sonic_porosity, raymer_porosity, effective_porosity)
from .saturation import archie, simandoux, indonesia
from .permeability import permeability as _permeability
from .netpay import net_flags, summarize
from ..mnemonics import resolve, log_kind


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

    depth: str default 'DEPTH'
        Depth column

    gr : str default 'GR'
        Gamma ray column

    rt : str default 'RT'
        Resistivity column

    nphi :  str default 'NPHI'
        Neutron porosity column

    rhob :  str default 'RHOB'
        Bulk density column

    sonic :  str default None
        Sonic column (optional)

    A curve name that is not a column of `df` is matched to a curve of the same type from its
    mnemonic (e.g. 'RT' to AT90, ILD or LLD), see `petrolib.mnemonics`. The columns used are in
    the `curves` attribute. If depth is the index of `df` it is used as the depth column

    use_mean : bool default None
        For cutoff. Whether to use mean of GR in IGR/VSH computation or not. If None, uses either median or average value

    use_median; bool default None
        For cutoff. Whether to use median of GR in IGR/VSH computation or not. If None, uses either mean or average value


    Example
    --------
    >>> #loading libraries/packages
    >>> from petrolib.petrophysics import Quanti
    >>> from petrolib.io import Zonation, load_las
    >>> from pathlib import Path

    >>> #loading well file and zonation/tops file
    >>> well_path = Path(r"./15_9-F-1A.LAS")
    >>> contact_path = Path(r"./well tops.csv")

    >>> df, las = load_las(well_path, return_csv=True, curves=['GR', 'RT', 'NPHI', 'RHOB'])

    >>> #creating zonation class to extra info
    >>> zones = Zonation(df, path=contact_path)
    >>> ztop, zbot, zn, fm = zones()

    >>> #creating quanti class
    >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', use_mean=True)
    >>> pp = Quanti(df, zn, ztop, zbot, fm)          # curves found from their mnemonics
    """


    def __init__(self, df:pd.DataFrame, zonename:list, ztop:list, zbot:list,  f_mids:list,
                 depth:str='DEPTH', gr:str='GR', rt:str='RT', nphi:str='NPHI', rhob:str='RHOB', sonic:str=None,
                use_mean:bool=None,  use_median:bool=None):

        #depth as the index (e.g. from load_las) becomes a column
        if depth not in df.columns and df.index.name is not None and log_kind(df.index.name) == 'DEPTH':
            df = df.reset_index()

        self._df = df
        self._depth = resolve(df, depth, 'DEPTH')
        self._use_mean = use_mean
        self._use_median = use_median
        self._gr = resolve(df, gr, 'GR')
        self._rt = resolve(df, rt, 'RT')
        self._nphi = resolve(df, nphi, 'NPHI')
        self._rhob = resolve(df, rhob, 'RHOB')
        self._sonic = resolve(df, sonic, 'DT') if sonic != None else None
        self.curves = {'DEPTH': self._depth, 'GR': self._gr, 'RT': self._rt, 'NPHI': self._nphi,
                       'RHOB': self._rhob, 'DT': self._sonic}
        self._ztop = ztop
        self._zbot = zbot
        self._zonename=zonename
        self._f_mids = f_mids

        #methods and parameters of each step. Later steps recompute the earlier ones
        #with these, so the values a user passed are not replaced by the defaults
        self._v_method = 'linear'
        self._p_method = 'density'
        self._sw_method = 'archie'
        self._k_method = 'timur'
        self._p_params = dict()
        self._sw_params = dict()
        self._palette = None
        self._fig = None
        self._ref = 'm'

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

    def _merged(self, new_data:list) -> pd.DataFrame:

        '''
        The zones joined back on the rows of `df`, for plotting. A sample on the boundary of two
        adjacent zones belongs to both; it is kept once
        '''

        data = pd.concat(new_data)
        return data[~data.index.duplicated()].reindex(np.arange(0, self._df.shape[0]))

    def _filter(self):

        '''
        
        A  private method to filter/restrict the data to the zones of interest
        This also determines GR_matrix, GR_Shale and GR_Sand of the reservoirs

        GR_Sand and GR_Shale are the 5th and 95th percentiles of GR in each zone.
        The min/max are set by spikes and make VSH too low.

        Returns
        -------
        Return two objects. Dataframe containing the parameters for VSH computation and filtered Dataframe

        '''

        # columns=self._df.columns.tolist()
        data = list()

        for i, z in enumerate(self._zonename):
            d= self._df[(self._df[self._depth] >= self._ztop[i]) & (self._df[self._depth] <= self._zbot[i])].copy()
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
            
            self._grShale.append(d[self._gr].quantile(0.95))
            self._grSand.append(d[self._gr].quantile(0.05))
            self._zone.append(z)
        
        #store parameter info in dataframe  
        df = {'GR_Matrix': self._grMatrix,
              'GR_Shale': self._grShale,
              'GR_Sand' : self._grSand,
              'Zone' : self._zone}

        self._results = pd.DataFrame.from_records(df)

        #return result

        return self._results, data

    def vshale(self, method:str='linear', show_plot:bool=False, palette_op:str=None, figsize:tuple=None, gr_base:float=None):
        
        '''

        Computes the Volume of Shale

        Parameters
        ----------
        method : str default 'linear'
            Volume of Shale method. {'linear', 'clavier', 'larionov_ter', 'larionov_older', 'stieber_1', 'stieber_2, 'stieber_m_pliocene'}
            * Linear = Gamma Ray Index (IGR)
            * Larionov Tertiary
            * Larionov Older Rocks
            * Stieber (Miocene/Pliocene)

            IGR = (GR - GR_Sand) / (GR_Shale - GR_Sand), with GR_Sand and GR_Shale
            the 5th and 95th GR percentiles of each zone

        show_plot : bool default False
            Display plot if True.. Plots GR, VSH and Zone track

        palette_op: str default None
            Palette option for to color code vshale plot. Check https://matplotlib.org/stable/tutorials/colors/colormaps.html

        figsize: tuple default None
            Size of plot

        gr_base: float default None
            Gamma Ray base value

        Returns
        ------
        Either/Both Dataframe containing the VSH and the plot if show_plot=True  
       

        Example
        -------
        >>> # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        >>> # display plot only
        >>> pp.vshale(method='clavier', show_plot=True, palette_op='cubehelix', figsize=(9,12))

        >>> # display data only 
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

            #gamma ray index (IGR), clipped to [0, 1] since some samples lie outside the P5/P95 endpoints
            igr = gamma_ray_index(d[self._gr], results['GR_Sand'][idx], results['GR_Shale'][idx])
            d['VShale'] = _vshale(igr, method)
            new_data.append(d)

        # PLOT
        data = self._merged(new_data)

        if show_plot == True:

            assert palette_op != None and figsize != None, f'Supply value for palette option and figsize'

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
            fig.suptitle(f'Volume of Shale', size=15, y=1.)

            span = 1
            cmap=plt.get_cmap(palette_op)
            color_index = np.arange(0, 1, span/10)

            logs = [self._gr, 'VShale']

            if gr_base == None:
                gr_base = 75. #(data[self._gr].max() - data[self._gr].min())/2
            else:
                gr_base = gr_base

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
                     rhob_matrix:float=2.65, fzs:float=None, show_plot:bool=False, figsize:tuple=None,
                     nphi_shale:float=.33, dt_matrix:float=55.5, dt_fluid:float=189.):

        '''
        Computes the effective and total porosities using the 'density', 'neutron_density',
        Wyllie's 'sonic' and Raymer-Hunt-Gardner 'raymer' methods.
        To use, must have called the `vshale` method

        * density, sonic, raymer : PHIE = PHIT - fzs * VShale
        * neutron_density : each log is corrected for shale first, then combined
            PHIN_c = PHIN - VShale * nphi_shale ; PHID_c = PHID - VShale * PHID_shale
            PHIE = combine(PHIN_c, PHID_c) ; PHIT = combine(PHIN, PHID)
            combine = sqrt((N^2 + D^2)/2) where N < D (gas crossover), else (N + D)/2

        See `petrolib.petrophysics.porosity` for the equations

        Parameters
        ----------
        method : str default 'density'
            Porosity method. {'density', 'neutron_density', 'sonic', 'raymer'}

        rhob_shale : float default 2.4
            Bulk density of shale (g/cc). Pick it in a shale interval

        rhob_fluid : float default 1.0
            Fluid density

        rhob_matrix : float default 2.65
            Matrix density

        fzs: float default None
            Shale porosity for PHIE = PHIT - fzs * VShale. If None, it is the density porosity of shale,
            calculated from rhob_fluid, rhob_shale and rhob_matrix. For 'neutron_density', setting it
            replaces the per-log shale correction

        show_plot : bool default False
            Display plot if True.. Plots RHOB, VSH, PHIE/PHIT and Zone track

        figsize: tuple default None
            Size of plot

        nphi_shale : float default 0.33
            Neutron porosity of shale (fraction), for 'neutron_density'. Pick it in a shale interval

        dt_matrix : float default 55.5
            Matrix slowness (us/ft), for 'sonic' and 'raymer'. Sandstone 55.5, limestone 47.5, dolomite 43.5

        dt_fluid : float default 189.
            Fluid slowness (us/ft), for 'sonic' and 'raymer'

        Returns
        -------
        Either/Both Dataframe containing the PHIE/PHIT and the plot if show_plot=True
       

        Example
        -------
        >>> # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        >>> # display plot only
        >>> pp.porosity(method='density', show_plot=True, figsize=(10, 12))

        >>> # display data only 
        >>> y = pp.porosity(method='density')
        >>> result = pd.concat(y)
        >>> print(result)

        '''

        # self._show_plot == show_plot
        self._p_method = method
        self._p_params = dict(rhob_shale=rhob_shale, rhob_fluid=rhob_fluid, rhob_matrix=rhob_matrix,
                              fzs=fzs, nphi_shale=nphi_shale, dt_matrix=dt_matrix, dt_fluid=dt_fluid)

        #calls the vshale method and concate all the dataframe representing each zones
        new_data = self.vshale(method=self._v_method, palette_op=self._palette, figsize=self._fig)

        #density porosity of shale
        phid_sh = density_porosity(rhob_shale, rhob_matrix, rhob_fluid)
        phi_fzs = fzs if fzs != None else phid_sh

        for d in new_data:

            #equations from Techlog manual
            if method == 'density':
                d['PHIT'] = density_porosity(d[self._rhob], rhob_matrix, rhob_fluid).clip(lower=0) #total porosity
                d['PHIE'] = effective_porosity(d['PHIT'], d['VShale'], phi_fzs)

            elif method == 'neutron_density':
                phid = density_porosity(d[self._rhob], rhob_matrix, rhob_fluid).clip(lower=0)
                phin = d[self._nphi].clip(lower=0)
                d['PHIT'] = neutron_density_porosity(phin, phid)
                if fzs != None:
                    d['PHIE'] = effective_porosity(d['PHIT'], d['VShale'], fzs)
                else:
                    d['PHIE'] = shale_corrected_nd_porosity(phin, phid, d['VShale'], nphi_shale, phid_sh)

            elif method in ('sonic', 'raymer'):
                assert self._sonic != None, 'Pass the sonic column to Quanti for sonic porosity'
                transform = sonic_porosity if method == 'sonic' else raymer_porosity
                d['PHIT'] = transform(d[self._sonic], dt_matrix, dt_fluid).clip(lower=0)
                d['PHIE'] = effective_porosity(d['PHIT'], d['VShale'], phi_fzs)

            else:
                raise ValueError(f"Unknown porosity method '{method}'")

        data = self._merged(new_data)

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
                                 show_plot:bool=False, figsize:tuple=None, rsh:float=2.):


        '''

        Computes water and hydrocarbon saturation
        To use, must have called both `vshale` and `porosity` methods

        * archie : SW = (a * rw / (PHIE^m * RT))^(1/n)
        * simandoux, indonesia : shaly sand models, using VShale and `rsh`
        * SW = 1 where PHIE = 0

        See `petrolib.petrophysics.saturation` for the equations, and for the Waxman-Smits
        and dual water models

        Parameters
        ----------
        method : str default 'archie'
            Water Saturation method. {'archie', 'simandoux', 'indonesia'}. 'simmandoux' is also accepted

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

        rsh : float default 2.
            Shale resistivity (ohm-m), for 'simandoux' and 'indonesia'

        Returns
        ------
        Either/Both Dataframe containing the SW, SH and the plot if show_plot=True
       

        Example
        -------
        >>> # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        >>> # display plot only
        >>> pp.water_saturation(method='archie', show_plot=True, figsize=(10, 12))


        >>> # display data only 
        >>> z = pp.water_saturation(method='archie')
        >>> result = pd.concat(z)
        >>> print(result)

        '''
        self._sw_method = method
        self._sw_params = dict(rw=rw, a=a, m=m, n=n, rsh=rsh)
        new_data = self.porosity(method=self._p_method, **self._p_params)

        for d in new_data:

            if method == 'archie':
                d['SW'] = archie(d['PHIE'], d[self._rt], rw, a, m, n)

            elif method in ('simandoux', 'simmandoux'):
                d['SW'] = simandoux(d['PHIE'], d[self._rt], d['VShale'], rw, rsh, a, m, n)

            elif method == 'indonesia':
                d['SW'] = indonesia(d['PHIE'], d[self._rt], d['VShale'], rw, rsh, a, m, n)

            else:
                raise ValueError(f"Unknown water saturation method '{method}'")

            d['SH'] = 1 - d['SW']


        data = self._merged(new_data)

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
                    ax[i].fill_betweenx(data[self._depth], 1, data[logs[i]], where=data[logs[i]]<=data[logs[i]].max(), facecolor='lightblue', interpolate=True, linewidth=0)

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

    def permeability(self, show_plot:bool=False, figsize:tuple=None, method:str='timur'):

        '''

        Computes the permeability (mD), PHIE and SW as fractions. Default Timur (1968)::

            Perm = 8581 * PHIE^4.4 / SW^2

        SW stands in for the irreducible water saturation, so the estimate only holds above the
        transition zone. SW is floored at 0.05.
        To use, must have called `vshale` and `porosity` and `water_saturation` methods

        See `petrolib.petrophysics.permeability` for the equations

        Parameters
        ----------
        show_plot : bool default False
            Display plot if True.. Plots PHIE, Permeability and Zone track

        figsize: tuple default None
            Size of plot

        method : str default 'timur'
            Permeability method. {'timur', 'coates', 'morris_biggs'}

        Returns
        ------
        Either/Both Dataframe containing the Perm and the plot if show_plot=True  
       

        Example
        -------
        >>> # create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        >>> # display plot only
        >>> pp.vshale(method='clavier')
        >>> pp.porosity(method='density')
        >>> pp.water_saturation(method='archie')
        >>> pp.permeability(show_plot=True, figsize=(9, 10))


        >>> # display data only 
        >>> x = pp.vshale(method='clavier')
        >>> y = pp.porosity(method='density')
        >>> z = pp.water_saturation(method='archie')
        >>> a = pp.permeability()
        >>> result = pd.concat(a)
        >>> print(result)

        '''

        self._k_method = method
        new_data = self.water_saturation(method=self._sw_method, **self._sw_params)

        for d in new_data:
            d['Perm'] = _permeability(d['PHIE'], d['SW'], method)

        data = self._merged(new_data)

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
                        ref_unit:str='m', show_plot:bool=False, palette_op:str=None, figsize:tuple=None, gr_base:float=None):

        '''

        Create the {ROCK, RES, PAY} flags. Each flag is a subset of the one before:
            ROCK : VShale < vsh_cutoff
            RES  : ROCK and PHIE > por_cutoff
            PAY  : RES  and SW < sw_cutoff

        To use, must have called `vshale`, `porosity`, `water_saturation` and `permeability` methods


        Parameters
        ----------

        vsh_cutoff : float
                Volume of Shale cutoff. Applied to the ['ROCK', 'RES', 'PAY'] flags

        por_cutoff : float
                Porosity cutoff. Applied to the ['RES', 'PAY'] flags

        sw_cutoff : float
            Water Saturation cutoff. Applied only to the ['PAY'] flag

        ref_unit : str default 'm'
            Reference unit for measured depth. Defaults to metres
        
        show_plot : bool default False
            Display plot if True.. Plots GR, RT, VSH, SW, Perm, NPHI/RHOB, PHIE/PHIT, ['ROCK', 'RES', 'PAY] flags and Zonation track

        palette_op : str default None
            palette option for VSH coloring. Check https://matplotlib.org/stable/tutorials/colors/colormaps.html for availabel palette options

        figsize: tuple default None
            Size of plot

        gr_base: float default None
            Gamma Ray base value

        Returns
        ------
        Either/Both Dataframe containing the flags and the plot if show_plot=True  
    

        Example
        -------
        >>> # Create Quanti class
        >>> pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB')

        >>> # Display plot only
        >>> pp.flags(por_cutoff=.12, vsh_cutoff=.5, sw_cutoff=0.8, show_plot=True, palette_op='cubehelix', figsize=(20, 15))

        >>> # Display data only 
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
        
        new_data = self.permeability(method=self._k_method)

        #creating  flags
        # 1 for net and 0 for gross
        for d in new_data:
            flags = net_flags(d['VShale'], d['PHIE'], d['SW'], vsh_cutoff, por_cutoff, sw_cutoff)
            for col in flags:
                d[col] = flags[col]

        data = self._merged(new_data)
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

            if gr_base == None:
                gr_base = 75. #(data[self._gr].max() - data[self._gr].min())/2
            else:
                gr_base = gr_base

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
            cycol = cycle('bgyrcmk')
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
                'Gross Flag', 'Net Flag'
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
                * net, grossand not net thicknesses
                * net-to-gross
                * average volume of shale
                * average porosity value
                * bulk volume of water
                * water saturation
                * permeability
                * porosity thickness (PHIH) and hydrocarbon pore thickness (HCPV)
        for each of the three flags {ROCK, RES, PAY}

        Gross is the zone thickness from the tops. Net is the thickness of the flagged samples,
        so gaps in the data count as not net. Water saturation is averaged by pore volume
        (PHIE * thickness), the other properties by thickness. See `petrolib.petrophysics.netpay`

        Parameters
        ----------
        name: str 
            Name of the well

        Returns
        -------
        Displays the Pay Summary Report table

        Example
        -------
        >>> pp.paySummary(name='15-9_F1A')

        '''
        self._name = name
        new_data = self.flags(vsh_cutoff=self._vsh_cutoff, por_cutoff=self._por_cutoff, sw_cutoff=self._sw_cutoff,
                              ref_unit=self._ref, show_plot=False, palette_op=self._pale)

        rows = list()
        for idx, (d, zone) in enumerate(zip(new_data, self._zonename)):
            # top and bottom from the zone tops
            top_ = self._ztop[idx]
            bot_ = self._zbot[idx]
            for flag in ('ROCK', 'RES', 'PAY'):
                rows.append({'Well': name, 'Zones': zone, 'Flag Name': flag, 'Top': top_, 'Bottom': bot_, 'Unit': self._ref,
                             **summarize(d[self._depth], d['VShale'], d['PHIE'], d['SW'], d[f'{flag}_NET_FLAG'],
                                         top_, bot_, perm=d['Perm'])})

        #store info in dataframe
        summary = pd.DataFrame(rows)

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
            'Permeability': self._k_method.title(),
            'VSH Cutoff':self._vsh_cutoff,
            'SW Cutoff': self._sw_cutoff,
            'PHI Cutoff': self._por_cutoff
        }

        df = pd.DataFrame.from_records(df, index=np.arange(0, 1)).T.rename({0:''}, axis=1).style.set_caption('Methods and Cutoffs').set_table_styles(self._styles)

        return df



    def save(self, file_name:str, log_data:bool=True):

        '''
        A method to save the pay summary results into a excel file: a dark blue header, ROCK rows in
        yellow, RES in green and PAY in red, thin borders and fitted columns (see `petrolib.io.write_pay_summary`)

        Parameters
        ----------
        file_name : str
            name of to save the pay summary report as; '.xlsx' is added if missing

        log_data : bool default True
            Also write the curves and flags of the zones to a 'Log Data' sheet

        Returns
        -------
        Path of the workbook

        '''

        from ..io.excel import write_pay_summary

        summary = self.paySummary(self._name).data
        logs = None
        if log_data:
            zones = self.flags(vsh_cutoff=self._vsh_cutoff, por_cutoff=self._por_cutoff, sw_cutoff=self._sw_cutoff,
                               ref_unit=self._ref, show_plot=False, palette_op=self._pale)
            logs = pd.concat([z.assign(Zone=name) for z, name in zip(zones, self._zonename)], ignore_index=True)
        return write_pay_summary(summary, file_name, log_data=logs)
