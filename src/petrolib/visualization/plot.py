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
'''

from pathlib import Path
import numpy as np
import lasio
import csv
import pandas as pd
import contextily as ctx
import geopandas as gpd 
from itertools import cycle
from random import choice
from matplotlib import pyplot as plt


def plotLoc(data:'list[lasio.las.LASFile]', shape_file:Path=None, area:str=None, figsize:slice=(7, 7),
             label:'list'=None, withmap:bool=False, return_table:bool=False) -> 'pd.DataFrame|None':
    
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

    figsize : slice
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



def tripleCombo(data:'pd.DataFrame', depth:'str', gr:'str', res:'str', nphi:'str', rhob:'str', ztop:float, zbot:float, 
                res_thres:float=2.0, fill:str=None, palette_op:'str'=None,
                limit:str=None, figsize:slice=(9, 10)) -> None:
    
    r'''
    Plots a three combo log of well data

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

    ztop : list 
        Top or minimum depth to zoom on. 

    zbot : list
        Bottom or maximum depth to zoom on.

    res_thres : float
        Resistivity threshold to use in the identification on hydrocarbon bearing zone

    fill : str default None
        To show either of the porous and/or non porous zones in the neutron-density crossover.
        Can either be ['left', 'right', 'both']

        ... default None - show neither of the porous nor non-porous zones
        ... 'left' - shows only porous zones
        ... 'right' - shows only non-porous zones
        ... 'both' - shows both porous and non-porous zones

    palette_op : str optional 
        Palette option to fill gamma ray log

    limit : str default None
        Tells which side to fill the gamma ray track, ['left', 'right']
        lf None, it's filled in both sides delineating shale-sand region

    figsize : slice
        Size of plot

    Example
    -------
    >>> import matplotlib inline
    >>> from petrolib.visualization.plot import tripleCombo
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
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    fig.suptitle(f'Three Combo Log Plot', size=15, y=1.)

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
    ax[0].set_xlabel('Gamma Ray (API)', color='black', labelpad=15)
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
    ax[1].set_xlabel(r'Resistivity Ohm-m', labelpad=15)
    ax[1].spines["top"].set_position(("axes", 1.02))
    ax[1].xaxis.set_ticks_position("top")
    ax[1].xaxis.set_label_position("top")
    ax[1].fill_betweenx(depth_log, res_thres, res_log, where=res_log >= res_thres, interpolate=True, color='red', linewidth=0)
    
    #for nphi
    ax[2].minorticks_on()
    ax[2].yaxis.grid(which='major', linestyle='-', linewidth=1, color='darkgrey')
    ax[2].yaxis.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
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
    nphi_.set_xlabel('Neutron Porosity (v/v)', color='blue')
    nphi_.spines["top"].set_position(("axes", 1.02))
    nphi_.xaxis.set_ticks_position("top")
    nphi_.xaxis.set_label_position("top")
    nphi_.set_xticks(list(np.linspace(nphi_log.min(), nphi_log.max(), num=5, dtype='float32')))
    
    #for rhob
    rhob_ = ax[2].twiny()
    rhob_.plot(rhob_log, depth_log, color='red', linewidth=1.)
    rhob_.set_xlim(rhob_log.min(), rhob_log.max())
    rhob_.set_ylim(ztop, zbot)
    rhob_.invert_yaxis()
    rhob_.xaxis.label.set_color('red')
    rhob_.tick_params(axis='x', colors='red')
    rhob_.spines['top'].set_edgecolor('red')
    rhob_.set_xlabel('Bulk Density (g/cc)', color='red')
    rhob_.spines["top"].set_position(("axes", 1.08))
    rhob_.xaxis.set_ticks_position("top")
    rhob_.xaxis.set_label_position("top")
    rhob_.set_xticks(list(np.linspace(rhob_log.min(), rhob_log.max(), num=5, dtype='float32')))
    
    #setting up the nphi and rhob fill
    #inspired from 
    x1=rhob_log
    x2=nphi_log
    
    x = np.array(rhob_.get_xlim())
    z = np.array(nphi_.get_xlim())

    nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)
    
    
    if fill == 'left':
        #shows only porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1<=nz, interpolate=True, color='lawngreen', linewidth=0)
    elif fill == 'right':
        #shows only non-porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1>=nz, interpolate=True, color='slategray', linewidth=0)
    elif fill == 'both':
        #shows both porous and non-porous zones
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1<=nz, interpolate=True, color='lawngreen', linewidth=0)
        rhob_.fill_betweenx(depth_log, x1, nz, where=x1>=nz, interpolate=True, color='slategray', linewidth=0)

    plt.tight_layout(h_pad=1.2)
    fig.subplots_adjust(wspace = 0.04)

    plt.show()



class Zonation:

    r'''
    This is a zonation class that extract zone/reservoir information from the file.
    This information may include top, bottom and zone/reservoir name. This information
    can be accessed when an instance of Zonation object is called. 

    Attributes
    ----------
    df : pd.DataFrame
        Dataframe of well data

    zones : list,, default None
        A optional attrribute containing list of dictionaries that holds zone information
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
    >>> from petrolib.visualization.plot import Zonation
    >>> zones = Zonation(df, path='./well tops.csv')
    >>> zones = Zonation(df, zones = [{'RES_A':[3000, 3100]}, {'RES_B':[3300, 3400]}])
    
    #get reservoir information by calling the Zonaion object
    #ztop = top ; zbot = base; zn = zonename ; fm = formation mids to place zone name in plot
    >>> ztop, zbot, zn, fm = zones()
    
    '''
    
    def __init__(self, df:'pd.DataFrame', zones:'list'=None, path:'Path'=None):

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

    def plotZone(self, depth:str, logs:list, top:float, bottom:float, title:str='Log Plot', figsize:slice=(8, 12)):

        r'''
        Plots log curves with zonation track. For 

        depth : str
            Depth column

        logs : lsit
            A list of logs curves to include in plot
        
        top : float
            Minimum depth to zoom on

        bottom : float
            Maximum depth to zoom on

        title : str
            Plot title

        figsize : slice
            Size of plot

        Example
        -------
        >>> zones.plotZone('DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, 'Volve')
        
        '''
    
        # create the subplots; ncols equals the number of logs
        fig, ax = plt.subplots(nrows=1, ncols=len(logs)+1, figsize=figsize)
        fig.suptitle(f'{title}', size=15, y=1.)

        cycol = cycle('bgrcmk')
        color = [choice(next(cycol)) for i in range(len(logs))]


        for i in range(len(logs)):
            
            if logs[i] == 'RT' or logs[i] == 'ILD':
                # for resistivity, semilog plot
                ax[i].semilogx(self._df[logs[i]], self._df[depth], color=color[i])
            else:
                # for non-resistivity, normal plot
                ax[i].plot(self._df[logs[i]], self._df[depth], color=color[i])

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
            ax[i].spines["top"].set_position(("axes", 1.02)); ax[i].xaxis.set_ticks_position("top")
            ax[i].xaxis.set_label_position("top")
            ax[i].hlines([t for t in self._ztop], xmin=self._df[logs[i]].min(), xmax=self._df[logs[i]].max(), colors='black', linestyles='solid')
            ax[i].hlines([b for b in self._zbot], xmin=self._df[logs[i]].min(), xmax=self._df[logs[i]].max(), colors='black', linestyles='solid')
        

        #formation subplot
        ax[-1].set_ylim(top, bottom); ax[-1].invert_yaxis()
        ax[-1].set_title('Zones', pad=45)
        ax[-1].set_xticks([])
        ax[-1].set_yticklabels([])
        ax[-1].set_xticklabels([])
        ax[-1].hlines([t for t in self._ztop], xmin=0, xmax=1, colors='black', linestyles='solid')
        ax[-1].hlines([b for b in self._zbot], xmin=0, xmax=1, colors='black', linestyles='solid')
        formations = ax[-1]

        #delineating zones
        colors = [choice(next(cycol)) for i in range(len(self._zonename))]
        for i in ax:
            for t,b, c in zip(self._ztop, self._zbot, colors):
                i.axhspan(t, b, color=c, alpha=0.1)

        #adding zone names
        for label, fm_mids in zip(self._zonename, self._fm_mid):
            formations.text(0.5, fm_mids, label, rotation=90,
                    verticalalignment='center', fontweight='bold',
                    fontsize='large')
    #    
        plt.tight_layout(h_pad=1)
        fig.subplots_adjust(wspace = 0.04)
        plt.show()


def plotLog(df:pd.DataFrame, depth:str, logs:list, top:float, bottom:float, title:str='Log Plot', figsize:slice=(8, 12)):

        r'''
        Plots log curves with zonation track. For 

        Argument
        --------
        df : pd.DataFrame
            Dataframe

        depth : str
            Depth column

        logs : lsit
            A list of logs curves to include in plot
        
        top : float
            Minimum depth to zoom on

        bottom : float
            Maximum depth to zoom on

        title : str
            Plot title

        figsize : slice
            Size of plot

        Example
        -------
        >>> import petrolib.visualization.plot.plotLog
        >>> plotLog('DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, 'Volve')
        >>> plotLog(df,'DEPTH', ['NPHI'], 2751, 2834.58, 'Fre-1')
        
        '''
    
        # create the subplots; ncols equals the number of logs
        fig, ax = plt.subplots(nrows=1, ncols=len(logs), figsize=figsize)
        fig.suptitle(f'{title}', size=15, y=1.)

        cycol = cycle('bgrcmk')
        color = [choice(next(cycol)) for i in range(len(logs))]

        if len(logs) > 1:
            for i in range(len(logs)):

                if logs[i] == 'RT' or logs[i] == 'ILD':
                    # for resistivity, semilog plot
                    ax[i].semilogx(df[logs[i]], df[depth], color=color[i])
                else:
                    # for non-resistivity, normal plot
                    ax[i].plot(df[logs[i]], df[depth], color=color[i])

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
                if i == 'RT' or i == 'ILD':
                        # for resistivity, semilog plot
                    plt.semilogx(df[i], df[depth], color=next(cycol))
                else:
                        # for non-resistivity, normal plot
                    plt.plot(df[i], df[depth], color=next(cycol))

                if i == 'NPHI' or i == 'PHIE':
                    plt.xlim(df[i].max(), df[i].min())
                    
                plt.title(i, pad=15)
                plt.minorticks_on()
                plt.ylim(bottom, top);
                plt.grid(which='major', linestyle='-', linewidth=1.0, color='darkgrey')
                plt.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgrey')
                plt.tick_params(axis='x', colors=next(cycol))
        plt.tight_layout(h_pad=1)
        fig.subplots_adjust(wspace = 0.04)
        plt.show()
