'''
A Python module for displaying log plots

Class
-----
Zonation

Function
--------
plotLoc
tripleCombo
plotLog
plotZoneCombo
plotLogFacies
plotLogs
'''


import numpy as np
import lasio
import csv
import pandas as pd
import contextily as ctx
import geopandas as gpd 
import os
from itertools import cycle
from pathlib import Path
from random import choice
from matplotlib import pyplot as plt
from typing import *
from random import choice
from matplotlib.patches import Patch 

def plotLoc(data:list[lasio.las.LASFile], shape_file:Path=None, area:str=None, figsize:tuple=(7, 7),
             label:list=None, withmap:bool=False, return_table:bool=False) -> pd.DataFrame|None:
    
    '''
    Plots location of wells
    The longitude and latitude must be available in th LAS fils
    
    Arguments
    ---------

    data : list
         List of lasio.las.LASFile objects
    
    shape_file : str, default None
        A .shp, .shx or .dbf file containing the shape file of the area where the well is located
        If not supplied, `area` must be passed

    area : str default, None
        Well location. Must inlcude area from `gpd.datasets.get_path('naturalearth_lowres')`

    figsize : tuple
        Size of plot

    label : list of str
        Well name(s). Must be a list of equal length with `data`

    withmap : bool default False  
        Plots location of wells on a map if True
        Plots the location on a scatter plot if False
    
    return_table : bool default False
        Return well information such as the long, lat and name

    Example
    -------
    >>> plotLoc(data=[las], area='Norway', label=['15_9-F-1A'], withmap=True, figsize=(30, 10))
    
    
    '''
    
    #getting well names        
    if label!=None:
        assert len(label) == len(data), 'label and data not of same length'
        well_name = label

    #getting longitude and latitude from LAS
    latitude = list()
    longitude = list()
    
    try:

        for idx, d in enumerate(data):
            assert type(d) == lasio.las.LASFile, f'Expected a lasio data object'
            # well_name.append(str(idx))
            longitude.append(d.well.LONG.value)
            latitude.append(d.well.LATI.value)

    except:

        raise AttributeError('Latitude and Longitude not in LAS file')

    #save coordinate location in dataframe
    latitude = [float(i[:3]) for i in latitude]
    longitude = [float(i[:3]) for i in longitude]

    well_df = pd.DataFrame({'WELL_NAME':well_name, 'LAT':latitude,'LONG':longitude})
    a, b = np.array(longitude), np.array(latitude)
    cycol = cycle('bgrcmk')
    color = [choice(next(cycol)) for i in range(len(well_name))]

    #plot well coordinates on map
    if withmap == True:
        
        if shape_file == None:
            
            assert area != None, f'Must supply area if shape_file is None'

            shape = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                            
            shape = shape[shape['name'] == area]  
                
            #plotting
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(f'WELL LOCATION - {area}')
        
            shape.geometry.plot(ax=ax, figsize=figsize, color='#e3bccf', edgecolor='k', alpha=0.5, zorder=1)
            a, b = np.array(longitude), np.array(latitude)
            plt.plot(a, b, 'b*', markersize=15)
            
            for i in range(0,well_df.shape[0]):
                plt.text(a[i], b[i], well_name[i], ha='right', size='large', color=color[i], va='bottom')

            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            
        elif shape_file != None:

            assert area == None, f'Area must be set to None'
            
            assert shape_file.endswidth('.shx') or shape_file.endswidth('.shp') or shape_file.endswidth('.dbf') or shape_file.endswidth('.prj'), f'Supply a valid shape file'

            shape = gpd.read_file(shape_file)
            
            #Plotting locations            
            fig, ax = plt.subplots(1, figsize=figsize)
            plt.title('WELL LOCATION')
            shape.geometry.plot(ax=ax, color='#e3bccf', edgecolor='k', alpha=0.5, zorder=1)
            plt.plot(a, b, 'b*', markersize=15)

            for i in range(0,well_df.shape[0]):
                plt.text(a[i], b[i], well_name[i], ha='right', size='large', color=color[i], va='bottom')

            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    else:
        
        assert area == None, 'Set area to None'
        
        fig=plt.figure(figsize=figsize)
        plt.title('WELL LOCATION')
        plt.grid(which='major', linestyle=':')

        plt.plot(a, b, 'b*', markersize=15)

        for i in range(0,well_df.shape[0]):
            plt.text(a[i], b[i], well_name[i], ha='right', size='large', color=color[i], va='bottom')


    if return_table:

        return well_df

    else:

        pass



def tripleCombo(data:pd.DataFrame, depth:str, gr:str, res:str, nphi:str, rhob:str, ztop:float, zbot:float, 
                res_thres:float=10.0, fill:str=None, palette_op:str=None,
                limit:str=None, figsize:tuple=(9, 10), title:str='Three Combo Log Plot') -> None:
    
    r'''
    Plots a three combo log of well data

    Arguments
    --------

    data : pd.DataFrame
        Dataframe of data 

    depth : str
        Depth column 

    gr : str
        Gamma ray column 

    res : str
        Resistivity column

    nphi :  str
        Neutron porosity column

    rhob :  str
        Bulk density column

    ztop : list 
        Top or minimum depth to zoom on. 

    zbot : list
        Bottom or maximum depth to zoom on.

    res_thres : float
        Resistivity threshold to use in the identification on hydrocarbon bearing zone

    fill : str default None
        To show either of the porous and/or non porous zones in the neutron-density crossover.
        Can either be ['left', 'right', 'both']

        * default None - show neither of the porous nor non-porous zones
        * 'left' - shows only porous zones
        * 'right' - shows only non-porous zones
        * 'both' - shows both porous and non-porous zones

    palette_op : str optional 
        Palette option to fill gamma ray log

    limit : str default None
        Tells which side to fill the gamma ray track, ['left', 'right']
        lf None, it's filled in both sides delineating shale-sand region

    figsize : tuple
        Size of plot
    
    title : str
        Title of plot

    Example
    -------
    >>> import matplotlib inline
    >>> from petrolib.plots import tripleCombo
    >>> # %matplotlib inline
    >>> tripleCombo(df, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', ztop=3300,
                         zbot=3450, res_thres=10, fill='right', palette_op='rainbow', limit='left')
    
    '''
        
    #getting logs from dataframe
    depth_log = data[depth]
    gr_log = data[gr]
    res_log = data[res]
    nphi_log = data[nphi]
    rhob_log = data[rhob]

    # color-fill options
    span = abs(gr_log.min()-gr_log.max())
    if palette_op != None:
        cmap=plt.get_cmap(palette_op)
    else:
        pass
    color_index = np.arange(gr_log.min(), gr_log.max(), span/100)

    # create the subplots; ncols equals the number of logs
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize,sharey=True)
    fig.suptitle(f'{title}', size=15, y=1.)

    #for GR track
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[0].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[0].plot(gr_log, depth_log, color='black', linewidth=1.0)
    ax[0].set_xlim(gr_log.min(), gr_log.max())
    ax[0].set_ylim(ztop, zbot)
    ax[0].invert_yaxis()
    ax[0].xaxis.label.set_color('black')
    ax[0].tick_params(axis='x', colors='black')
    ax[0].spines['top'].set_edgecolor('black')
    ax[0].set_xlabel('Gamma ray\nGR (gAPI)', color='black', labelpad=15)
    ax[0].spines["top"].set_position(("axes", 1.02))
    ax[0].xaxis.set_ticks_position("top")
    ax[0].xaxis.set_label_position("top")
    
    if palette_op == None:
        assert limit == None, 'Set limit to None'
        gr_base = (gr_log.max() - gr_log.min())/2
        ax[0].fill_betweenx(depth_log, gr_base, gr_log, where=gr_log<=gr_base, facecolor='yellow', linewidth=0)
        ax[0].fill_betweenx(depth_log, gr_log, gr_base, where=gr_log>=gr_base, facecolor='brown', linewidth=0)

    elif palette_op != None:
        assert limit != None, 'Set limit value. Can\'t be None'
        if limit == 'left':
            for index in sorted(color_index):
                index_value = (index-gr_log.min())/span
                palette = cmap(index_value)
                ax[0].fill_betweenx(depth_log, gr_log.min(), gr_log, where=gr_log>=index, color=palette)

        elif limit == 'right':
            for index in sorted(color_index):
                index_value = (index-gr_log.min())/span
                palette = cmap(index_value)
                ax[0].fill_betweenx(depth_log, gr_log.max(), gr_log, where=gr_log>=index, color=palette)


    #for resitivity
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
    ax[1].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[1].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[1].semilogx(res_log, depth_log, color='red', linewidth=1.0, linestyle='--')
    ax[1].set_xlim(res_log.min(), res_log.max())
    ax[1].set_ylim(ztop, zbot)
    ax[1].invert_yaxis()
    ax[1].xaxis.label.set_color('red')
    ax[1].tick_params(axis='x', colors='red')
    ax[1].spines['top'].set_edgecolor('red')
    ax[1].set_xlabel('Resistivity\nILD (ohm.m)', labelpad=15)
    ax[1].spines["top"].set_position(("axes", 1.02))
    ax[1].xaxis.set_ticks_position("top")
    ax[1].xaxis.set_label_position("top")
    ax[1].fill_betweenx(depth_log, res_thres, res_log, where=res_log >= res_thres, interpolate=True, color='red', linewidth=0)
    
    #for nphi
    ax[2].minorticks_on()
    ax[2].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[2].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[2].set_xticklabels([]);ax[2].set_xticks([])
    rhob_ = ax[2].twiny()
    rhob_.plot(rhob_log, depth_log, color='red', linewidth=1.)
    rhob_.set_xlim(rhob_log.min(), rhob_log.max())
    rhob_.set_ylim(ztop, zbot)
    rhob_.invert_yaxis()
    rhob_.xaxis.label.set_color('red')
    rhob_.tick_params(axis='x', colors='red')
    rhob_.spines['top'].set_edgecolor('red')
    rhob_.set_xlabel('Bulk Density\nRHOB (g/cm3)', color='red')
    rhob_.spines["top"].set_position(("axes", 1.02))
    rhob_.xaxis.set_ticks_position("top")
    rhob_.xaxis.set_label_position("top")
    rhob_.set_xticks(list(np.linspace(rhob_log.min(), rhob_log.max(), num=5, dtype='float32')))
    
    nphi_ = ax[2].twiny()
    nphi_.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
    nphi_.plot(nphi_log, depth_log, color='blue', linewidth=1.0, linestyle='--')
    nphi_.set_xlim(nphi_log.max(), nphi_log.min())
    nphi_.set_ylim(ztop, zbot)
    nphi_.invert_yaxis()
    # nphi_.invert_xaxis()
    nphi_.xaxis.label.set_color('blue')
    nphi_.tick_params(axis='x', colors='blue')
    nphi_.spines['top'].set_edgecolor('blue')
    nphi_.set_xlabel('Neutron Porosity\nNPHI (m3/m3)', color='blue')
    nphi_.spines["top"].set_position(("axes", 1.05))
    nphi_.xaxis.set_ticks_position("top")
    nphi_.xaxis.set_label_position("top")
    nphi_.set_xticks(list(np.linspace(nphi_log.min(), nphi_log.max(), num=5, dtype='float32')))
    
    #setting up the nphi and rhob fill
    #inspired from 
    x1=rhob_log
    x2=nphi_log
    
    x = np.array(rhob_.get_xlim())
    z = np.array(nphi_.get_xlim())

    nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)
    
    
    if fill == 'left':
        #shows only porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1<=nz, interpolate=True, hatch='..', facecolor='yellow', linewidth=0)
    elif fill == 'right':
        #shows only non-porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1>=nz, interpolate=True, hatch='---', facecolor='slategray', linewidth=0)
    elif fill == 'both':
        #shows both porous and non-porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1<=nz, interpolate=True, hatch='..', facecolor='yellow', linewidth=0)
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1>=nz, interpolate=True, hatch='---', facecolor='slategray', linewidth=0)

    plt.tight_layout(h_pad=1.2)
    fig.subplots_adjust(wspace = 0.0)

    plt.show()


class Zonation:

    r'''
    This is a zonation class that extract zone/reservoir information from the file.
    This information may include top, bottom and zone/reservoir name. This information
    can be accessed when an instance of `Zonation object` is called. 

    Attributes
    ----------
    df : pd.DataFrame
        Dataframe of well data

    zones : list of dictionaries, default None
        A optional attribute containing list of dictionaries that holds zone information
        If None, `path` must be supplied

    path : Path or str, default None
        A csv file containing reservoir info. 
        To be able to use this, the information in the file should be entered in the following format
            top, bottom, zonename
            100, 200, RES_A
            210, 220, RES_B
        If None, `zone` must be passed

    Example
    -------
    >>> from petrolib.plots import Zonation
    >>> zones = Zonation(df, path='./well tops.csv')
    >>> zones = Zonation(df, zones = [{'RES_A':[3000, 3100]}, {'RES_B':[3300, 3400]}])
    
    *get reservoir information by calling the Zonation object*

    ztop = top ; zbot = base; zn = zonename ; fm = formation mids to place zone name in plot
    >>> ztop, zbot, zn, fm = zones()
    
    '''
    
    def __init__(self, df:pd.DataFrame, zones:List[Dict[str, List[float]]]=None, path:Path=None):

        self._df = df 
        self._zones = zones
        self._path = path
        self._ztop = []
        self._zbot = []
        self._zonename = []
        
        if self._path != None and self._zones==None:
            with open(self._path, mode='r') as tops:
                items = list(csv.DictReader(tops)) #reads csv content as list of dictionaries
  
            for item in items:
                self._ztop.append(float(item.get('top')))
                self._zbot.append(float(item.get('bottom')))
                self._zonename.append(str(item.get('zonename')))
            
            
        elif self._path==None and self._zones!=None:
            for zone in self._zones:
                self._ztop.append(float(list(zone.values())[0][0]))
                self._zbot.append(float(list(zone.values())[0][1]))
                self._zonename.append(list(zone.keys())[0])
        
#         elif path!=None and zones!=None:
#             pass
        
        self._fm_mid = []
            
        for t, b in zip(self._ztop, self._zbot):
            self._fm_mid.append((t+(b-t)/2))

    def __call__(self):

        '''
        When the zonation object is called, this method is returned 

        Returns
        -------
        In the following sequence, Tops, Bottoms, Zone name and Formation mids

        Example
        -------
        >>> ztop, zbot, zn, fm = zones()
        
        '''
        return self._ztop, self._zbot, self._zonename, self._fm_mid

    def plotZone(self, depth:str, logs:List[str], top:float, bottom:float, title:str='Log Plot', figsize:tuple=(8, 12)):

        r'''
        Plots log curves with zonation track.

        Attributes
        ----------

        depth : str
            Depth column

        logs : list of str
            A list of logs curves to include in plot
        
        top : float
            Minimum depth to zoom on

        bottom : float
            Maximum depth to zoom on

        title : str
            Plot title

        figsize : tuple
            Size of plot

        Example
        -------
        >>> Zonation.plotZone('DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, 'Volve')
        
        '''
    
        # create the subplots; ncols equals the number of logs
        fig, ax = plt.subplots(nrows=1, ncols=len(logs)+1, figsize=figsize, sharey=True)
        fig.suptitle(f'{title}', size=15, y=1.)

        cycol = cycle('bgrcmyk')
        color = [choice(next(cycol)) for i in range(len(logs))]
        np.random.shuffle(color)

        for i in range(len(logs)):
            
            if logs[i] == 'RT' or logs[i] == 'ILD':
                # for resistivity, semilog plot
                ax[i].semilogx(self._df[logs[i]], self._df[depth], color=color[i], linewidth=1.)
            else:
                # for non-resistivity, normal plot
                ax[i].plot(self._df[logs[i]], self._df[depth], color=color[i], linewidth=1.)

            if logs[i] == 'NPHI':
                ax[i].invert_xaxis()

            ax[i].set_title(logs[i], pad=15)
            ax[i].minorticks_on()
            ax[i].set_ylim(top, bottom); ax[i].invert_yaxis()
            ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
            ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
            ax[i].xaxis.label.set_color(color[i])
            ax[i].tick_params(axis='x', colors=color[i])
            ax[i].spines['top'].set_edgecolor(color[i])
            ax[i].spines["top"].set_position(("axes", 1.01)); ax[i].xaxis.set_ticks_position("top")
            ax[i].xaxis.set_label_position("top")
            ax[i].hlines([t for t in self._ztop], xmin=self._df[logs[i]].min(), xmax=self._df[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
            ax[i].hlines([b for b in self._zbot], xmin=self._df[logs[i]].min(), xmax=self._df[logs[i]].max(), colors='black', linestyles='solid', linewidth=1.)
        

        #formation subplot
        ax[-1].set_ylim(top, bottom); ax[-1].invert_yaxis()
        ax[-1].set_title('Zones', pad=45)
        ax[-1].set_xticks([])
        # ax[-1].set_yticklabels([])
        ax[-1].set_xticklabels([])
        ax[-1].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
        ax[-1].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
        formations = ax[-1]

        #delineating zones
        colors = [choice(next(cycol)) for i in range(len(self._zonename))]
        np.random.shuffle(colors)
        for i in ax:
            for t,b, c in zip(self._ztop, self._zbot, colors):
                i.axhspan(t, b, color=c, alpha=0.3)

        #adding zone names
        for label, fm_mids in zip(self._zonename, self._fm_mid):
            formations.text(0.5, fm_mids, label, rotation=0,
                    verticalalignment='center', fontweight='bold',
                    fontsize='large')
    #    
        plt.tight_layout(h_pad=1)
        fig.subplots_adjust(wspace = 0.04)
        plt.show()


def plotLog(df:pd.DataFrame, depth:str, logs:List[str], top:float, bottom:float, title:str='Log Plot', figsize:tuple=(8, 12)):

        r'''
        Plots log curves singly. To plot overlay plots use `plots.plotLogs_` or `plots.plotLogFacies`

        Argument
        --------
        df : pd.DataFrame
            Dataframe

        depth : str
            Depth column

        logs : list of str
            A list of logs curves to include in plot
        
        top : float
            Minimum depth to zoom on

        bottom : float
            Maximum depth to zoom on

        title : str
            Plot title

        figsize : tuple
            Size of plot

        Example
        -------
        >>> import petrolib.plots.plotLog
        >>> plotLog('DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, 'Volve')
        >>> plotLog(df,'DEPTH', ['NPHI'], 2751, 2834.58, 'Fre-1')
        
        '''
    
        # create the subplots; ncols equals the number of logs
        fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=figsize, sharey=True)
        fig.suptitle(f'{title}', size=15, y=1.)

        cycol = cycle('bgrcmyk')
        color = [choice(next(cycol)) for i in range(len(logs))]
        np.random.shuffle(color)

        if len(logs) > 1:
            for i in range(len(logs)):
                np.random.shuffle(color)
                if logs[i] == 'RT' or logs[i] == 'ILD':
                    # for resistivity, semilog plot
                    ax[i].semilogx(df[logs[i]], df[depth], color=color[i], linewidth=1.)
                else:
                    # for non-resistivity, normal plot
                    ax[i].plot(df[logs[i]], df[depth], color=color[i], linewidth=1.)

                if logs[i] == 'NPHI' or logs[i] == 'PHIE':
                    ax[i].invert_xaxis()

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(top, bottom); ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].xaxis.label.set_color(color[i])
                ax[i].tick_params(axis='x', colors=color[i])
                ax[i].spines['top'].set_edgecolor(color[i])
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
        
        elif len(logs) == 1:
            
            for i in logs:
                np.random.shuffle(color)
                if i == 'RT' or i == 'ILD':
                    # for resistivity, semilog plot
                    plt.semilogx(df[i], df[depth], color=next(cycol), linewidth=1.)
                else:
                        # for non-resistivity, normal plot
                    plt.plot(df[i], df[depth], color=next(cycol), linewidth=1.)

                if i == 'NPHI' or i == 'PHIE':
                    plt.xlim(df[i].max(), df[i].min())
                    
                plt.title(i, pad=15)
                plt.minorticks_on()
                plt.ylim(bottom, top);
                plt.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                plt.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                plt.tick_params(axis='x', colors=next(cycol))
        plt.tight_layout(h_pad=1)
        fig.subplots_adjust(wspace = 0.02)
        plt.show()


    
def plotZoneCombo(data:pd.DataFrame, depth:str, gr:str, res:str, nphi:str, rhob:str, ztop:float, zbot:float,
                   ztops:List[float], zbots:List[float], zonename:List[str], limit:str, res_thres:float=10., 
                   palette_op:str=None, fill:str=None, title:str='Log plot', figsize:tuple=(10, 20)) -> None:
    
    r'''
    Function for plotting three combo logs alongside the zonation/reservoir track

    Arguments
    --------

    df : pd.DataFrame
        Dataframe of data 

    depth : str
        Depth column 

    gr : str
        Gamma ray column 

    res : str
        Resistivity column

    nphi :  str
        Neutron porosity column

    rhob :  str
        Bulk density column

    ztop : float 
        Top or minimum depth value

    zbot : float
        Bottom or maximum depth value

    ztops : list of float
        Tops (depth) of each reservoir/zones

    zbots : list of float
        Bottom (depth) of each reservoir/zones

    zonename : list of str
        Name of each zones

    limit : str default None
        Tells which side to fill the gamma ray track, ['left', 'right'].
        lf None, it's filled in both sides delineating shale-sand region

    res_thres : float
        Resistivity threshold to use in the identification on hydrocarbon bearing zone

    palette_op : str optional 
        Palette option to fill gamma ray log. If None, `fill must be provided

    fill : str default None
        To show either of the porous and/or non porous zones in the neutron-density crossover.
        Can either be ['left', 'right', 'both']

        * default None - show neither of the porous nor non-porous zones
        * 'left' - shows only porous zones
        * 'right' - shows only non-porous zones
        * 'both' - shows both porous and non-porous zones

    figsize : tuple
        Size of plot
        
    Example
    -------
    >>> from petrolib.plots import plotZoneCombo
    >>> plotZoneCombo(well11, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', min(ztop), max(zbot),
               ztop, zbot, zn, fill='both', limit=None, figsize=(13, 30), title='ATAGA-11')
    '''
    #getting logs from dataframe
    depth_log = data[depth]
    gr_log = data[gr]
    res_log = data[res]
    nphi_log = data[nphi]
    rhob_log = data[rhob]

    # color-fill options
    span = abs(gr_log.min()-gr_log.max())
    if palette_op != None:
        cmap=plt.get_cmap(palette_op)
    else:
        pass
    color_index = np.arange(gr_log.min(), gr_log.max(), span/100)

    # create the subplots; ncols equals the number of logs
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=figsize, sharey=True)
    fig.suptitle(f'{title}', size=15, y=1.)

    #for GR track
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[0].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[0].plot(gr_log, depth_log, color='black', linewidth=1.0)
    ax[0].set_xlim(gr_log.min(), gr_log.max())
    ax[0].set_ylim(ztop, zbot)
    ax[0].invert_yaxis()
    ax[0].xaxis.label.set_color('black')
    ax[0].tick_params(axis='x', colors='black')
    ax[0].spines['top'].set_edgecolor('black')
    ax[0].set_xlabel('Gamma ray\nGR (gAPI)', color='black', labelpad=15)
    ax[0].spines["top"].set_position(("axes", 1.01))
    ax[0].xaxis.set_ticks_position("top")
    ax[0].xaxis.set_label_position("top")
    ax[0].hlines([t for t in ztops], xmin=gr_log.min(), xmax=gr_log.max(), colors='black', linestyles='solid',linewidth=1.)
    ax[0].hlines([b for b in zbots], xmin=gr_log.min(), xmax=gr_log.max(), colors='black', linestyles='solid', linewidth=1.)
    
    if palette_op == None:
        assert limit == None, 'Set limit to None'
        gr_base = (gr_log.max() - gr_log.min())/2
        ax[0].fill_betweenx(depth_log, gr_base, gr_log, where=gr_log<=gr_base, facecolor='yellow', linewidth=0)
        ax[0].fill_betweenx(depth_log, gr_log, gr_base, where=gr_log>=gr_base, facecolor='brown', linewidth=0)

    elif palette_op != None:
        assert limit != None, 'Set limit value. Can\'t be None'
        if limit == 'left':
            for index in sorted(color_index):
                index_value = (index-gr_log.min())/span
                palette = cmap(index_value)
                ax[0].fill_betweenx(depth_log, gr_log.min(), gr_log, where=gr_log>=index, color=palette)

        elif limit == 'right':
            for index in sorted(color_index):
                index_value = (index-gr_log.min())/span
                palette = cmap(index_value)
                ax[0].fill_betweenx(depth_log, gr_log.max(), gr_log, where=gr_log>=index, color=palette)


    #for resitivity
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
    ax[1].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[1].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[1].semilogx(res_log, depth_log, color='red', linewidth=1.0, linestyle='--')
    ax[1].set_xlim(res_log.min(), res_log.max())
    ax[1].set_ylim(ztop, zbot)
    ax[1].invert_yaxis()
    ax[1].xaxis.label.set_color('red')
    ax[1].tick_params(axis='x', colors='red')
    ax[1].spines['top'].set_edgecolor('red')
    ax[1].set_xlabel('Resistivity\nILD (ohm.m)', labelpad=15)
    ax[1].spines["top"].set_position(("axes", 1.01))
    ax[1].xaxis.set_ticks_position("top")
    ax[1].xaxis.set_label_position("top")
    ax[1].fill_betweenx(depth_log, res_thres, res_log, where=res_log >= res_thres, interpolate=True, color='red', linewidth=0)
    ax[1].hlines([t for t in ztops], xmin=res_log.min(), xmax=res_log.max(), colors='black', linestyles='solid',linewidth=1.)
    ax[1].hlines([b for b in zbots], xmin=res_log.min(), xmax=res_log.max(), colors='black', linestyles='solid', linewidth=1.)
    
    #for nphi
    ax[2].minorticks_on()
    ax[2].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[2].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
    ax[2].set_xticklabels([]);ax[2].set_xticks([])
    rhob_ = ax[2].twiny()
    rhob_.plot(rhob_log, depth_log, color='red', linewidth=1.)
    rhob_.set_xlim(rhob_log.min(), rhob_log.max())
    rhob_.set_ylim(ztop, zbot)
    rhob_.invert_yaxis()
    rhob_.xaxis.label.set_color('red')
    rhob_.tick_params(axis='x', colors='red')
    rhob_.spines['top'].set_edgecolor('red')
    rhob_.set_xlabel('Bulk Density\nRHOB (g/cm3)', color='red')
    rhob_.spines["top"].set_position(("axes", 1.01))
    rhob_.xaxis.set_ticks_position("top")
    rhob_.xaxis.set_label_position("top")
    rhob_.set_xticks(list(np.linspace(rhob_log.min(), rhob_log.max(), num=4)))
    rhob_.hlines([t for t in ztops], xmin=rhob_log.min(), xmax=rhob_log.max(), colors='black', linestyles='solid',linewidth=1.)
    rhob_.hlines([b for b in zbots], xmin=rhob_log.min(), xmax=rhob_log.max(), colors='black', linestyles='solid', linewidth=1.)

    nphi_ = ax[2].twiny()
    nphi_.grid(which='major', linestyle='-', linewidth=0.5, color='darkgrey')
    nphi_.plot(nphi_log, depth_log, color='blue', linewidth=1.0, linestyle='--')
    nphi_.set_xlim(nphi_log.max(), nphi_log.min())
    nphi_.set_ylim(ztop, zbot)
    nphi_.invert_yaxis()
    # nphi_.invert_xaxis()
    nphi_.xaxis.label.set_color('blue')
    nphi_.tick_params(axis='x', colors='blue')
    nphi_.spines['top'].set_edgecolor('blue')
    nphi_.set_xlabel('Neutron Porosity\nNPHI (m3/m3)', color='blue')
    nphi_.spines["top"].set_position(("axes", 1.05))
    nphi_.xaxis.set_ticks_position("top")
    nphi_.xaxis.set_label_position("top")
    nphi_.set_xticks(list(np.linspace(nphi_log.min(), nphi_log.max(), num=4)))
    nphi_.hlines([t for t in ztops], xmin=nphi_log.min(), xmax=nphi_log.max(), colors='black', linestyles='solid',linewidth=1.)
    nphi_.hlines([b for b in zbots], xmin=nphi_log.min(), xmax=nphi_log.max(), colors='black', linestyles='solid', linewidth=1.)
    #setting up the nphi and rhob fill
    #inspired from 
    x1=rhob_log
    x2=nphi_log
    
    x = np.array(rhob_.get_xlim())
    z = np.array(nphi_.get_xlim())

    nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)
    
    
    if fill == 'left':
        #shows only porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1<=nz, interpolate=True, hatch='..', facecolor='yellow', linewidth=0)
    elif fill == 'right':
        #shows only non-porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1>=nz, interpolate=True, hatch='---', facecolor='slategray', linewidth=0)
    elif fill == 'both':
        #shows both porous and non-porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1<=nz, interpolate=True, hatch='..', facecolor='yellow', linewidth=0)
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1>=nz, interpolate=True, hatch='---', facecolor='slategray', linewidth=0)

    #formation subplot
    ax[-1].set_ylim(ztop, zbot); ax[-1].invert_yaxis()
    ax[-1].set_title('Zones', pad=45)
    ax[-1].set_xticks([])
    # ax[-1].set_yticklabels([])
    ax[-1].set_xticklabels([])
    ax[-1].hlines([t for t in ztops], xmin=0, xmax=1, colors='black', linestyles='solid',linewidth=1.)
    ax[-1].hlines([b for b in zbots], xmin=0, xmax=1, colors='black', linestyles='solid', linewidth=1.)
    formations = ax[-1]

    #delineating zones
    # np.random.seed(2)
    cycol = cycle('bgrcymk')
    color = [choice(next(cycol)) for i in range(len(ztops))]
    np.random.shuffle(color)
    for i in ax:
        for t,b, c in zip(ztops, zbots, color):
            i.axhspan(t, b, color=c, alpha=.3)

    #adding zone names
    fm_mid = []
            
    for t, b in zip(ztops, zbots):
            fm_mid.append((t+(b-t)/2))
    
    for label, fm_mids in zip(zonename, fm_mid):
        formations.text(0.5, fm_mids, label, rotation=0,
                verticalalignment='center', fontweight='bold',
                fontsize='large')
            
    plt.tight_layout(h_pad=1.2)
    fig.subplots_adjust(wspace = 0.0)


def plotLogFacies(df:pd.DataFrame, depth:str, logs:List[list], top:float, bottom:float, facies:str=None,  
                  title:str='Log Plot', figsize:tuple=(8, 12)):

    r'''
    
    Plots overlayed/super-imposed log curves along with the facies. 

    Argument
    --------
    df : pd.DataFrame
        Dataframe

    depth : str
        Depth column

    logs : list of lists/str
        A list of logs curves to include in plot
    
    top : float
        Minimum depth to zoom on

    bottom : float
        Maximum depth to zoom on

    facies : str
        Facies columns. Supported facies include :
            ['Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',
            'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement']

    title : str
        Plot title

    figsize : tuple
        Size of plot

    Example
    -------
    >>> from petrolib.plots import *
    >>> plotLogFacies(well11, 'DEPTH', ['GR', 'RT', ['RHOB', 'NPHI']], facies='litho', top=well11.DEPTH.min(),
                            bottom=well11.DEPTH.max(), figsize=(9, 12), title='15-9-F1A')

    '''    

    df = df.copy()
    # file_dir, file_name = os.path.split(__file__)
    
    df_litho = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'litho_info.csv'))
    
    # create the subplots; ncols equals the number of logs
    if facies is None:
        fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=figsize, sharey=True)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(logs)+1, figsize=figsize, sharey=True)

        
    fig.suptitle(f'{title}', size=15, y=1.)
    
    #generating random number of colors equal to total number of logs
    cycol = cycle('bgyrcmk')
    tot_logs = 0
    for i in logs:
        if isinstance(i, list):
            tot_logs += len(i)
        else:
            tot_logs += 1
    color = [choice(next(cycol)) for i in range(tot_logs)]
    np.random.shuffle(color)
    
    #filtering facies in lithofacies info dataframe that the number of identified lithology in `df`
    if facies != None:
        
        facies_label = df[facies].unique()

        df_litho = df_litho[df_litho.lith.isin(facies_label)]

        try:
            if df[facies].dtype == 'object':
                lithology = {}
                for li, li_num in zip(df_litho.lith, df_litho.lith_num):
                    lithology[li] = li_num 
                df[facies] = [lithology[i] for i in df[facies]]
            elif df[facies].dtype == 'int':
                pass

        except:
            raise ValueError(f'Unidentified facies in \'{facies}\'')
    
    #generating log tracks
    if len(logs) > 1:
        for i, j in enumerate(logs):

            if isinstance(j, list):

                assert len(j) > 1, 'Error. list of lists of curves must be greater than 1.'
                np.random.shuffle(color)
                for ii in range(len(j)):
                    if logs[i][ii] == 'RT' or logs[i][ii] == 'ILD':
                        # for resistivity, semilog plot
                        ax[i].semilogx(df[logs[i][ii]], df[depth], color=color[-i], linewidth=1.)

                    if ii == 0:# and logs[i][ii] != 'RT':
                        # for non-resistivity, normal plot
                        ax[i].plot(df[logs[i][ii]], df[depth], color=color[ii], linewidth=1.)
                        
                        # ax[ii].set_xticklabels([]);  ax[ii].set_xticks([])
                        ax[i].set_xticklabels([]);  ax[i].set_xticks([])
                    if ii >= 1:# or logs[i][ii] != 'RT':
                        ax[i].twiny().plot(df[logs[i][ii]], df[depth], color=color[ii], linewidth=1.)    
                        ax[i].set_xticklabels([]);  ax[i].set_xticks([])                    
                    
                    
                    ax[i].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
                    ax[i].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                    
                tracks = []

                for _ in range(len(j)):
                    tracks.append(ax[i].twiny())
                
                additive = 1.02
                for _, axes in enumerate(tracks):

                    axes.set_xlabel(logs[i][_])
                    # axes.set_ylim(top, bottom)
                    if logs[i][_] == 'NPHI' or logs[i][_] == 'PHIE':
                        axes.set_xlim(df[logs[i][_]].max(), df[logs[i][_]].min())
                        axes.set_xticks(list(np.linspace(df[logs[i][_]].max(), df[logs[i][_]].min(), num=4)))
                    else:
                        axes.set_xlim(df[logs[i][_]].min(), df[logs[i][_]].max())
                        axes.set_xticks(list(np.linspace(df[logs[i][_]].min(), df[logs[i][_]].max(), num=4)))
                        
                    axes.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                    axes.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                    axes.xaxis.label.set_color(color[_])
                    axes.tick_params(axis='x', colors=color[_])
                    axes.spines['top'].set_edgecolor(color[_])
                    axes.spines["top"].set_position(("axes", additive))
                    axes.xaxis.set_ticks_position("top")
                    axes.xaxis.set_label_position("top")
                    axes.set_frame_on(True)
                    axes.patch.set_visible(False)
                    axes.invert_yaxis()
                    additive += 0.06
  
            else:
                np.random.shuffle(color)
                if logs[i] == 'RT' or logs[i] == 'ILD':
                    # for resistivity, semilog plot
                    ax[i].semilogx(df[logs[i]], df[depth], color=color[i], linewidth=1.)
                    
                else:
                    # for non-resistivity, normal plot
                    ax[i].plot(df[logs[i]], df[depth], color=color[i], linewidth=1.)

                if logs[i] == 'NPHI' or logs[i] == 'PHIE':
                    ax[i].invert_xaxis()

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(top, bottom); ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].xaxis.label.set_color(color[i])
                ax[i].tick_params(axis='x', colors=color[i])
                ax[i].spines['top'].set_edgecolor(color[i])
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
                ax[i].set_frame_on(True)
                ax[i].patch.set_visible(False)
                
    elif len(logs) == 1 and facies == None:
            
        for idxi, i in enumerate(logs):
            if not isinstance(i, list):
                np.random.shuffle(color)
                if i == 'RT' or i == 'ILD':
                    # for resistivity, semilog plot
                    plt.semilogx(df[i], df[depth], color=color[idxi], linewidth=1.)
                else:
                        # for non-resistivity, normal plot
                    plt.plot(df[i], df[depth], color=color[idxi], linewidth=1.)

                if i == 'NPHI' or i == 'PHIE':
                    plt.xlim(df[i].max(), df[i].min())

                plt.title(i, pad=15)
                plt.minorticks_on()
                plt.ylim(bottom, top);
                plt.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                plt.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                plt.gca().tick_params(axis='x', colors=color[idxi])
                plt.gca().spines['top'].set_edgecolor(color[idxi])
                plt.gca().spines["top"].set_position(("axes", 1.02)); 
                plt.gca().xaxis.set_ticks_position("top")
                plt.gca().xaxis.set_label_position("top")
                
            else:
                np.random.shuffle(color)
                assert len(i) > 1, 'List of list of curves must be greater than one.'
                for idxii, ii in enumerate(range(len(i))):
                    if logs[idxi][idxii] == 'RT' or logs[idxi][idxii] == 'ILD':
                        # for resistivity, semilog plot
                        ax.semilogx(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)
                        # ax.set_xticklabels([]);ax.set_xticks([])
                    if idxii == 0:
                        # for non-resistivity, normal plot
                        ax.plot(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)
                        ax.set_xticklabels([]); ax.set_xticks([])

                    if idxii >= 1:
                        ax.twiny().plot(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)    
                        ax.set_xticklabels([]);  ax.set_xticks([])

                    ax.yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
                    ax.yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')

                             
                tracks = []
                
                np.random.shuffle(color)
                for x in range(len(i)):
                    tracks.append(ax.twiny())

                additive = 1.02
                for _, axes in enumerate(tracks):

                    axes.set_xlabel(logs[idxi][_])
                    # axes.minorticks_on()
                    axes.set_ylim(top, bottom)

                    if logs[idxi][_] == 'NPHI' or logs[idxi][_] == 'PHIE':
                        axes.set_xlim(df[logs[idxi][_]].max(), df[logs[idxi][_]].min())
                        axes.set_xticks(list(np.linspace(df[logs[idxi][_]].max(), df[logs[idxi][_]].min(), num=4)))
                    else:
                        axes.set_xlim(df[logs[idxi][_]].min(), df[logs[idxi][_]].max())
                        axes.set_xticks(list(np.linspace(df[logs[idxi][_]].min(), df[logs[idxi][_]].max(), num=4)))

                    axes.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                    axes.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                    axes.xaxis.label.set_color(color[_])
                    axes.tick_params(axis='x', colors=color[_])
                    axes.spines['top'].set_edgecolor(color[_])
                    axes.spines["top"].set_position(("axes", additive))
                    axes.xaxis.set_ticks_position("top")
                    axes.xaxis.set_label_position("top")
                    axes.set_frame_on(True)
                    axes.patch.set_visible(False)
                    axes.invert_yaxis()
                    additive += 0.06

    elif len(logs) == 1 and facies != None:            
        for idxi, i in enumerate(logs):
            if not isinstance(i, list):
                np.random.shuffle(color)
                if i == 'RT' or i == 'ILD':
                    # for resistivity, semilog plot
                    ax[idxi].semilogx(df[i], df[depth], color=color[idxi], linewidth=1.)
                else:
                        # for non-resistivity, normal plot
                    ax[idxi].plot(df[i], df[depth], color=color[idxi], linewidth=1.)

                if i == 'NPHI' or i == 'PHIE':
                    ax[idxi].xlim(df[i].max(), df[i].min())

                ax[idxi].set_title(i, pad=15)
                ax[idxi].minorticks_on()
                ax[idxi].set_ylim(bottom, top);
                ax[idxi].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[idxi].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[idxi].tick_params(axis='x', colors=color[idxi])
                ax[idxi].spines['top'].set_edgecolor(color[idxi])
                ax[idxi].spines["top"].set_position(("axes", 1.02)); ax[idxi].xaxis.set_ticks_position("top")
                ax[idxi].xaxis.set_label_position("top")
                
            else:
                assert len(i) > 1, 'List of list of curves must be greater than one.'
                for idxii, ii in enumerate(range(len(i))):

                    if logs[idxi][idxii] == 'RT' or logs[idxi][idxii] == 'ILD':
                        # for resistivity, semilog plot
                        ax[idxi].semilogx(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)
                        # ax[idxi].set_xticklabels([]);ax[idxi].set_xticks([])
                    if idxii == 0:
                        # for non-resistivity, normal plot
                        ax[idxi].plot(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)
                        ax[idxi].set_xticklabels([]); ax[idxi].set_xticks([])
                    if idxii >= 1:
                        ax[idxi].twiny().plot(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)  
                        ax[idxii].twiny().set_xticklabels([]);  ax[idxii].twiny().set_xticks([])  

                    ax[idxi].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
                    ax[idxi].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')

                tracks = []

                for x in range(len(i)):
                    tracks.append(ax[idxi].twiny())
#                     np.random.shuffle(color)
                additive = 1.02
                for _, axes in enumerate(tracks):

                    axes.set_xlabel(logs[idxi][_])
                    # axes.minorticks_on()
                    axes.set_ylim(top, bottom)

                    if logs[idxi][_] == 'NPHI' or logs[idxi][_] == 'PHIE':
                        axes.set_xlim(df[logs[idxi][_]].max(), df[logs[idxi][_]].min())
                        axes.set_xticks(list(np.linspace(df[logs[idxi][_]].max(), df[logs[idxi][_]].min(), num=4)))
                    else:
                        axes.set_xlim(df[logs[idxi][_]].min(), df[logs[idxi][_]].max())
                        axes.set_xticks(list(np.linspace(df[logs[idxi][_]].min(), df[logs[idxi][_]].max(), num=4)))
                        

                    axes.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                    axes.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                    axes.xaxis.label.set_color(color[_])
                    axes.tick_params(axis='x', colors=color[_])
                    axes.spines['top'].set_edgecolor(color[_])
                    axes.spines["top"].set_position(("axes", additive))
                    axes.xaxis.set_ticks_position("top")
                    axes.xaxis.set_label_position("top")
                    axes.set_frame_on(True)
                    axes.patch.set_visible(False)
                    axes.invert_yaxis()
                    additive += 0.06

    #Plot lithology track 
    if facies != None:       
        ax[-1].plot(df[facies], df[depth], color='black', linewidth=0.5)
        ax[-1].set_xlabel("Lithology")
        ax[-1].set_xlim(0, 1)
        ax[-1].xaxis.label.set_color("black")
        ax[-1].tick_params(axis='x', colors="black")
        ax[-1].spines["top"].set_edgecolor("black")
        ax[-1].xaxis.set_ticks_position("top")
        ax[-1].xaxis.set_label_position("top")
        ax[-1].spines["top"].set_position(("axes", 1.02))

        #adding legend of lithofacies
        legend = []
        for key, name, color, hatch in zip(df_litho.lith_num, df_litho.lith, df_litho.color, df_litho.hatch):
            ax[-1].fill_betweenx(df[depth], 0, df[facies], where=(df[facies]==key),
                         facecolor=color, hatch=hatch)
            legend.append(Patch(facecolor=color, hatch=hatch, label=name))
        ax[-1].set_xticks([0, 1])
        ax[-1].legend(handles=legend, bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.5)            
   
    else:
        pass
    
    plt.tight_layout(h_pad=1)
    fig.subplots_adjust(wspace = 0.04)

    plt.show()


def plotLogs(df:pd.DataFrame, depth:str, logs:List[List[str]], top:float, bottom:float,  
                  title:str='Log Plot', figsize:tuple=(8, 12)):

    r'''
    
    Plots overlayed/super-imposed log curves. 

    Argument
    --------
    df : pd.DataFrame
        Dataframe

    depth : str
        Depth column

    logs : list of lists/str
        A list of logs curves to include in plot
    
    top : float
        Minimum depth to zoom on

    bottom : float
        Maximum depth to zoom on

    title : str
        Plot title

    figsize : tuple
        Size of plot

    Example
    -------
    >>> plotLogs_(well11, 'DEPTH', ['GR', 'RT', ['RHOB', 'NPHI']], top=well11.DEPTH.min(),
                            bottom=well11.DEPTH.max(), figsize=(9, 12), title='15-9-F1A')

    '''    
    df = df.copy()
    
    # create the subplots; ncols equals the number of logs
    fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=figsize, sharey=True)
        
    fig.suptitle(f'{title}', size=15, y=1.)
    
    #generating random number of colors equal to total number of logs
    cycol = cycle('bgyrcmk')
    tot_logs = 0
    for i in logs:
        if isinstance(i, list):
            tot_logs += len(i)
        else:
            tot_logs += 1
    color = [choice(next(cycol)) for i in range(tot_logs)]
    np.random.shuffle(color)
    
    
    #generating log tracks
    if len(logs) > 1:
        for i, j in enumerate(logs):

            if isinstance(j, list):

                assert len(j) > 1, 'Error. list of lists of curves must be greater than 1.'
                np.random.shuffle(color)
                for ii in range(len(j)):
                    if logs[i][ii] == 'RT' or logs[i][ii] == 'ILD':
                        # for resistivity, semilog plot
                        ax[i].semilogx(df[logs[i][ii]], df[depth], color=color[-i], linewidth=1.)

                    if ii == 0:# and logs[i][ii] != 'RT':
                        # for non-resistivity, normal plot
                        ax[i].plot(df[logs[i][ii]], df[depth], color=color[ii], linewidth=1.)
                        
                        # ax[ii].set_xticklabels([]);  ax[ii].set_xticks([])
                        ax[i].set_xticklabels([]);  ax[i].set_xticks([])
                    if ii >= 1:# or logs[i][ii] != 'RT':
                        ax[i].twiny().plot(df[logs[i][ii]], df[depth], color=color[ii], linewidth=1.)    
                        ax[i].set_xticklabels([]);  ax[i].set_xticks([])        
                    
                    
                    ax[i].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
                    ax[i].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                    
                tracks = []

                for _ in range(len(j)):
                    tracks.append(ax[i].twiny())
                
                additive = 1.02
                for _, axes in enumerate(tracks):

                    axes.set_xlabel(logs[i][_])
                    # axes.set_ylim(top, bottom)
                    if logs[i][_] == 'NPHI' or logs[i][_] == 'PHIE':
                        axes.set_xlim(df[logs[i][_]].max(), df[logs[i][_]].min())
                        axes.set_xticks(list(np.linspace(df[logs[i][_]].min(), df[logs[i][_]].max(), num=4)))
                    else:
                        axes.set_xlim(df[logs[i][_]].min(), df[logs[i][_]].max())
                        axes.set_xticks(list(np.linspace(df[logs[i][_]].min(), df[logs[i][_]].max(), num=4)))
                        
                    axes.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                    axes.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                    axes.xaxis.label.set_color(color[_])
                    axes.tick_params(axis='x', colors=color[_])
                    axes.spines['top'].set_edgecolor(color[_])
                    axes.spines["top"].set_position(("axes", additive))
                    axes.xaxis.set_ticks_position("top")
                    axes.xaxis.set_label_position("top")
                    axes.set_frame_on(True)
                    axes.patch.set_visible(False)
                    axes.invert_yaxis()
                    additive += 0.06
  
            else:
                np.random.shuffle(color)
                if logs[i] == 'RT' or logs[i] == 'ILD':
                    # for resistivity, semilog plot
                    ax[i].semilogx(df[logs[i]], df[depth], color=color[i], linewidth=1.)
                    
                else:
                    # for non-resistivity, normal plot
                    ax[i].plot(df[logs[i]], df[depth], color=color[i], linewidth=1.)

                if logs[i] == 'NPHI' or logs[i] == 'PHIE':
                    ax[i].invert_xaxis()

                ax[i].set_title(logs[i], pad=15)
                ax[i].minorticks_on()
                ax[i].set_ylim(top, bottom); ax[i].invert_yaxis()
                ax[i].grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                ax[i].grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                ax[i].xaxis.label.set_color(color[i])
                ax[i].tick_params(axis='x', colors=color[i])
                ax[i].spines['top'].set_edgecolor(color[i])
                ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
                ax[i].xaxis.set_label_position("top")
                ax[i].set_frame_on(True)
                ax[i].patch.set_visible(False)
                
    elif len(logs) == 1:
            
        for idxi, i in enumerate(logs):
            if not isinstance(i, list):
                np.random.shuffle(color)
                if i == 'RT' or i == 'ILD':
                    # for resistivity, semilog plot
                    plt.semilogx(df[i], df[depth], color=color[idxi], linewidth=1.)
                else:
                        # for non-resistivity, normal plot
                    plt.plot(df[i], df[depth], color=color[idxi], linewidth=1.)

                if i == 'NPHI' or i == 'PHIE':
                    plt.xlim(df[i].max(), df[i].min())

                plt.title(i, pad=15)
                plt.minorticks_on()
                plt.ylim(bottom, top);
                plt.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                plt.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                plt.gca().tick_params(axis='x', colors=color[idxi])
                plt.gca().spines['top'].set_edgecolor(color[idxi])
                plt.gca().spines["top"].set_position(("axes", 1.02)); 
                plt.gca().xaxis.set_ticks_position("top")
                plt.gca().xaxis.set_label_position("top")
                plt.gca().xaxis.label.set_color(color[idxi])
                
            else:
                np.random.shuffle(color)
                assert len(i) > 1, 'List of list of curves must be greater than one.'
                for idxii, ii in enumerate(range(len(i))):
                    if logs[idxi][idxii] == 'RT' or logs[idxi][idxii] == 'ILD':
                        # for resistivity, semilog plot
                        ax.semilogx(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)
                        # ax.set_xticklabels([]);ax.set_xticks([])
                    if idxii == 0:
                        # for non-resistivity, normal plot
                        ax.plot(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)
                        ax.set_xticklabels([]); ax.set_xticks([])

                    if idxii >= 1:
                        ax.twiny().plot(df[logs[idxi][idxii]], df[depth], color=color[idxii], linewidth=1.)    
                        ax.set_xticklabels([]);  ax.set_xticks([])

                    ax.yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
                    ax.yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')

                             
                tracks = []
                
                np.random.shuffle(color)
                for x in range(len(i)):
                    tracks.append(ax.twiny())

                additive = 1.02
                for _, axes in enumerate(tracks):

                    axes.set_xlabel(logs[idxi][_])
                    # axes.minorticks_on()
                    axes.set_ylim(top, bottom)

                    if logs[idxi][_] == 'NPHI' or logs[idxi][_] == 'PHIE':
                        axes.set_xlim(df[logs[idxi][_]].max(), df[logs[idxi][_]].min())
                        axes.set_xticks(list(np.linspace(df[logs[idxi][_]].max(), df[logs[idxi][_]].min(), num=4)))
                    else:
                        axes.set_xlim(df[logs[idxi][_]].min(), df[logs[idxi][_]].max())
                        axes.set_xticks(list(np.linspace(df[logs[idxi][_]].min(), df[logs[idxi][_]].max(), num=4)))
                        

                    axes.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                    axes.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                    axes.xaxis.label.set_color(color[_])
                    axes.tick_params(axis='x', colors=color[_])
                    axes.spines['top'].set_edgecolor(color[_])
                    axes.spines["top"].set_position(("axes", additive))
                    axes.xaxis.set_ticks_position("top")
                    axes.xaxis.set_label_position("top")
                    axes.set_frame_on(True)
                    axes.patch.set_visible(False)
                    axes.invert_yaxis()
                    additive += 0.06

    plt.tight_layout(h_pad=1)
    fig.subplots_adjust(wspace = 0.04)

    plt.show()
