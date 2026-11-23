'''
petrolib: a Python package for petrophysical evaluation

Subpackages
-----------
io            loading of LAS, DLIS and table files, and well tops (Zonation)
mnemonics     recognizing curve mnemonics of any vendor, and renaming them to standard names
processing    renaming and cleaning of log curves, lithofacies modelling
petrophysics  petrophysics equations and the Quanti workflow
plots         log tracks, cross plots, petrophysics plots and well maps
stats         correlation and frequency statistics
automl        filling missing logs with AutoGluon (pip install 'petrolib[automl]')
exceptions    petrolib errors

Subpackages load on first use, e.g. `petrolib.plots` after `import petrolib`
'''
import importlib

__version__ = '2.0.0'

__all__ = ['io', 'mnemonics', 'processing', 'petrophysics', 'plots', 'stats', 'automl', 'exceptions']

#modules of petrolib 1.x, kept as deprecated aliases
_DEPRECATED = ('file_reader', 'procs', 'interp', 'workflow', 'utils')


def __getattr__(name):
    if name in __all__ or name in _DEPRECATED:
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(list(globals()) + __all__)
