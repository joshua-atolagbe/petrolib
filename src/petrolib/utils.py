'''
Deprecated alias of `petrolib.exceptions`, kept so existing code keeps working.
It will be removed in petrolib 3.0; import from `petrolib.exceptions` instead.

'''
import warnings

from petrolib.exceptions import MnemonicError

warnings.warn("petrolib.utils is deprecated; import from petrolib.exceptions instead", DeprecationWarning, stacklevel=2)
