'''
Plots: log tracks, cross plots, petrophysics plots and well maps

Modules
-------
logs          log, triple-combo, zone and lithofacies track plots
crossplots    neutron-density and Pickett cross plots
petrophysics  Buckles, Hingle, Winland, RQI, Lorenz, M-N, MID, Thomas-Stieber, movable hydrocarbon, histogram,
              cutoff, pay summary, volume and uncertainty plots
geomechanics  mechanical property tracks, pressure against depth and rock physics cross plots
maps          well locations (needs 'petrolib[maps]' for basemaps)

Example
-------
>>> from petrolib.plots import tripleCombo, bucklesPlot
'''
import warnings

from .logs import plotFaciesZoneCombo, tripleCombo, plotLog, plotZoneCombo, plotLogFacies, plotLogs
from .crossplots import crossPlot, picketPlot
from .petrophysics import (bucklesPlot, winlandPlot, rqiPlot, lorenzPlot, mnPlot, midPlot, thomasStieberPlot,
                           cutoffPlot, volumePlot, swUncertaintyPlot, hinglePlot, movableHydrocarbonPlot,
                           histogramPlot, paySummaryPlot)
from .geomechanics import mechanicalPlot, pressurePlot, rockPhysicsPlot
from .maps import plotLoc

__all__ = [
    'plotFaciesZoneCombo', 'tripleCombo', 'plotLog', 'plotZoneCombo', 'plotLogFacies', 'plotLogs',
    'crossPlot', 'picketPlot',
    'bucklesPlot', 'winlandPlot', 'rqiPlot', 'lorenzPlot', 'mnPlot', 'midPlot', 'thomasStieberPlot',
    'cutoffPlot', 'volumePlot', 'swUncertaintyPlot', 'hinglePlot', 'movableHydrocarbonPlot', 'histogramPlot',
    'paySummaryPlot', 'mechanicalPlot', 'pressurePlot', 'rockPhysicsPlot',
    'plotLoc',
]


def __getattr__(name):
    #Zonation moved to petrolib.io in petrolib 2.0
    if name == 'Zonation':
        warnings.warn('petrolib.plots.Zonation is deprecated; import it from petrolib.io instead',
                      DeprecationWarning, stacklevel=2)
        from ..io.tops import Zonation
        return Zonation
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
