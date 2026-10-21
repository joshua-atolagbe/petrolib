'''
Maps of well locations

'''
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from itertools import cycle
from random import choice
from matplotlib import pyplot as plt
from pathlib import Path
import lasio

#Natural Earth 1:50m countries, used when `area` is given (geopandas >= 1.0 no longer ships it)
NATURAL_EARTH_URL = 'https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip'


def _map_libraries():

    '''
    Imports geopandas and contextily, which are only needed for maps
    '''

    try:
        import geopandas as gpd
        import contextily as ctx
    except ImportError as e:
        raise ImportError("Maps need geopandas and contextily: pip install 'petrolib[maps]'") from e
    return gpd, ctx


def _to_degrees(value) -> float:

    '''
    Converts a LAS coordinate to decimal degrees. Accepts numbers, numeric strings and
    degree-minute-second strings such as 58 26' 29.8" N
    '''

    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    parts = [float(p) for p in re.findall(r'\d+(?:\.\d+)?', str(value))]
    if not parts:
        raise ValueError(f'Cannot read coordinate {value!r}')
    degrees = sum(p / 60**i for i, p in enumerate(parts[:3]))
    return -degrees if re.search(r'[SWsw]\s*$', str(value).strip()) or str(value).strip().startswith('-') else degrees


def _country(gpd, area:str):

    '''
    Outline of a country from Natural Earth
    '''

    world = gpd.read_file(NATURAL_EARTH_URL)
    for col in ('NAME', 'ADMIN', 'name'):
        if col in world.columns and (world[col] == area).any():
            return world[world[col] == area]
    raise ValueError(f"Area '{area}' not found in Natural Earth countries")


def _extent(shape, lon:np.ndarray, lat:np.ndarray, pad:float=.05) -> tuple:

    '''
    Map limits around the wells and the outline parts nearest to them, so that far-off parts of an
    area (overseas territories, remote islands) do not shrink the map
    '''

    from shapely.geometry import Point
    parts = [g for geom in shape.geometry for g in getattr(geom, 'geoms', [geom])]
    near = {min(range(len(parts)), key=lambda i: parts[i].distance(Point(x, y))) for x, y in zip(lon, lat)}
    bounds = np.array([parts[i].bounds for i in near])
    x0, x1 = min(bounds[:, 0].min(), lon.min()), max(bounds[:, 2].max(), lon.max())
    y0, y1 = min(bounds[:, 1].min(), lat.min()), max(bounds[:, 3].max(), lat.max())
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def plotLoc(data:list[lasio.las.LASFile], shape_file:Path=None, area:str=None, figsize:tuple=(7, 7),
             label:list=None, withmap:bool=False, return_table:bool=False, basemap=None) -> pd.DataFrame|None:

    '''
    Plots location of wells
    The longitude and latitude must be available in the LAS files (LONG and LATI)

    Parameters
    ----------

    data : list
         List of lasio.las.LASFile objects

    shape_file : str, default None
        A .shp, .shx, .dbf or .prj file containing the shape file of the area where the well is located
        If not supplied, `area` must be passed

    area : str default, None
        Country where the wells are, as named in Natural Earth (e.g. 'Norway'). The outline is
        downloaded, so this needs internet access

    figsize : tuple
        Size of plot

    label : list of str
        Well name(s). Must be a list of equal length with `data`. If None, the WELL names in the LAS files are used

    withmap : bool default False
        Plots location of wells on a map if True. Needs geopandas and contextily ('petrolib[maps]')
        and internet access for the basemap.
        Plots the location on a scatter plot if False

    return_table : bool default False
        Return well information such as the long, lat and name

    basemap : contextily provider or str default None
        Tiles drawn under the map, e.g. `contextily.providers.Esri.WorldImagery` or a tile URL.
        Default: Esri World Topographic Map, which needs no API key

    Example
    -------
    >>> plotLoc(data=[las], area='Norway', label=['15_9-F-1A'], withmap=True, figsize=(30, 10))


    '''

    #getting well names
    if label!=None:
        assert len(label) == len(data), 'label and data not of same length'
        well_name = label
    else:
        well_name = list()
        for i, d in enumerate(data):
            try:
                well_name.append(str(d.well.WELL.value) or str(i))
            except (AttributeError, KeyError):
                well_name.append(str(i))

    #getting longitude and latitude from LAS
    latitude = list()
    longitude = list()

    for d in data:
        assert type(d) == lasio.las.LASFile, f'Expected a lasio data object'
        try:
            longitude.append(_to_degrees(d.well.LONG.value))
            latitude.append(_to_degrees(d.well.LATI.value))
        except (AttributeError, KeyError) as e:
            raise AttributeError('Latitude and Longitude not in LAS file') from e

    #save coordinate location in dataframe
    well_df = pd.DataFrame({'WELL_NAME':well_name, 'LAT':latitude,'LONG':longitude})
    a, b = np.array(longitude), np.array(latitude)
    cycol = cycle('bgrcmk')
    color = [choice(next(cycol)) for i in range(len(well_name))]
    np.random.shuffle(color)

    #plot well coordinates on map
    if withmap == True:

        gpd, ctx = _map_libraries()

        if shape_file == None:

            assert area != None, f'Must supply area if shape_file is None'
            shape = _country(gpd, area)
            title = f'WELL LOCATION - {area}'

        else:

            assert area == None, f'Area must be set to None'
            assert str(shape_file).endswith(('.shx', '.shp', '.dbf', '.prj')), f'Supply a valid shape file'
            shape = gpd.read_file(shape_file)
            title = 'WELL LOCATION'

        #the well coordinates are longitude/latitude, so show the outline in the same system
        shape = shape.to_crs('EPSG:4326') if shape.crs is not None else shape.set_crs('EPSG:4326')

        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title)
        shape.geometry.plot(ax=ax, color='#e3bccf', edgecolor='k', alpha=0.5, zorder=1)
        plt.plot(a, b, 'b*', markersize=15)

        for i in range(0,well_df.shape[0]):
            plt.text(a[i], b[i], well_name[i], ha='right', size='large', color=color[i], va='bottom')

        xlim, ylim = _extent(shape, a, b)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        #CartoDB tiles now need an API key and OpenStreetMap blocks bulk tile clients
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Esri.WorldTopoMap if basemap is None else basemap)

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
