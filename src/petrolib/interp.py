'''
Deprecated alias of `petrolib.plots.crossplots`, kept so existing code keeps working.
It will be removed in petrolib 3.0; import from `petrolib.plots.crossplots` instead.

'''
import warnings

from petrolib.plots.crossplots import crossPlot, picketPlot

warnings.warn("petrolib.interp is deprecated; import from petrolib.plots.crossplots instead", DeprecationWarning, stacklevel=2)
