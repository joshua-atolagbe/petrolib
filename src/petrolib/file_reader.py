'''
Deprecated alias of `petrolib.io`, kept so existing code keeps working.
It will be removed in petrolib 3.0; import from `petrolib.io` instead.

'''
import warnings

from petrolib.io import load_las, load_table

warnings.warn("petrolib.file_reader is deprecated; import from petrolib.io instead", DeprecationWarning, stacklevel=2)
