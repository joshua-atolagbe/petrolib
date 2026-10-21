'''
Well tops and zonation

'''
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import cycle
from random import choice
from matplotlib import pyplot as plt
import csv
from pathlib import Path
from typing import Dict, List

from ..mnemonics import is_resistivity, is_porosity

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
        By default the file should have the following columns (other names can be set with
        `top_col`, `bot_col` and `zone_col`; spaces around names and values are ignored)::

            top, bottom, zonename
            100, 200, RES_A
            210, 220, RES_B

        If None, `zone` must be passed

    well : str, default None
        Only use the rows of this well, from the `well_col` column. Needed when `path` holds
        the tops of several wells

    well_col : str, default 'Well'
        Column with the well names in `path`

    top_col : str, default 'top'
        Column with the zone tops in `path`

    bot_col : str, default 'bottom'
        Column with the zone bottoms in `path`

    zone_col : str, default 'zonename'
        Column with the zone names in `path`

    Example
    -------
    >>> from petrolib.io import Zonation
    >>> zones = Zonation(df, path='./well tops.csv')
    >>> zones = Zonation(df, zones = [{'RES_A':[3000, 3100]}, {'RES_B':[3300, 3400]}])

    >>> #tops file of several wells, with other column names
    >>> zones = Zonation(df, path='./tops.csv', well='KPD-12', top_col='Top (m MD)',
    ...                  bot_col='Base (m MD)', zone_col='Zone')

    >>> #get reservoir information by calling the Zonation object
    >>> #ztop = top ; zbot = base; zn = zonename ; fm = formation mids to place zone name in plot
    >>> ztop, zbot, zn, fm = zones()
    
    '''
    
    def __init__(self, df:pd.DataFrame, zones:List[Dict[str, List[float]]]=None, path:Path=None,
                 well:str=None, well_col:str='Well', top_col:str='top', bot_col:str='bottom',
                 zone_col:str='zonename'):

        self._df = df
        self._zones = zones
        self._path = path
        self._ztop = []
        self._zbot = []
        self._zonename = []

        if (self._path == None) == (self._zones == None):
            raise ValueError('Pass either `path` or `zones`, not both or neither')

        if self._path != None and self._zones==None:
            with open(self._path, mode='r', encoding='utf-8-sig', newline='') as tops:
                reader = csv.DictReader(tops) #reads csv content as list of dictionaries
                columns = [c.strip() for c in (reader.fieldnames or [])]
                #strip spaces around column names and values, e.g. 'top, bottom, zonename'
                items = [{k.strip(): (v or '').strip() for k, v in item.items() if k != None} for item in reader]

            needed = [top_col, bot_col, zone_col] + ([well_col] if well != None else [])
            missing = [c for c in needed if c not in columns]
            if missing:
                raise ValueError(f'Column(s) {missing} not in {self._path}. Available columns: {columns}')

            if well != None:
                wells = sorted({item[well_col] for item in items})
                items = [item for item in items if item[well_col] == str(well).strip()]
                if not items:
                    raise ValueError(f"Well '{well}' not in {self._path}. Available wells: {wells}")
            elif well_col in columns and len({item[well_col] for item in items}) > 1:
                raise ValueError(f'{self._path} has the tops of several wells. Pass `well` to pick one')

            for item in items:
                self._ztop.append(float(item[top_col]))
                self._zbot.append(float(item[bot_col]))
                self._zonename.append(item[zone_col])


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

        Parameters
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
            
            if is_resistivity(logs[i]):
                # for resistivity, semilog plot
                ax[i].semilogx(self._df[logs[i]], self._df[depth], color=color[i], linewidth=1.)
            else:
                # for non-resistivity, normal plot
                ax[i].plot(self._df[logs[i]], self._df[depth], color=color[i], linewidth=1.)

            if is_porosity(logs[i]):
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
        
        ax[0].set_ylabel('DEPTH (m/ft)')

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
        fig.subplots_adjust(wspace = 0.0)
        plt.show()
