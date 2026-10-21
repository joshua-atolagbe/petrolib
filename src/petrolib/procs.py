'''
Deprecated alias of `petrolib.processing`, kept so existing code keeps working.
It will be removed in petrolib 3.0; import from `petrolib.processing` instead.

'''
import warnings

from petrolib.processing import set_alias, process_data, trim, model_facies

warnings.warn("petrolib.procs is deprecated; import from petrolib.processing instead", DeprecationWarning, stacklevel=2)
