'''
Deprecated alias of `petrolib.petrophysics`, kept so existing code keeps working.
It will be removed in petrolib 3.0; import from `petrolib.petrophysics` instead.

'''
import warnings

from petrolib.petrophysics import Quanti

warnings.warn("petrolib.workflow is deprecated; import from petrolib.petrophysics instead", DeprecationWarning, stacklevel=2)
