import importlib
import sys
import warnings

import pytest

import petrolib


def test_version():
    assert petrolib.__version__ == '2.0.0'


@pytest.mark.parametrize('name', ['io', 'mnemonics', 'processing', 'petrophysics', 'plots', 'stats', 'automl', 'exceptions'])
def test_subpackages_load_on_attribute_access(name):
    assert getattr(petrolib, name).__name__ == f'petrolib.{name}'


def test_new_import_paths():
    from petrolib.io import load_well, load_las, load_dlis, dlis_summary, load_table, Zonation, standardize
    from petrolib.mnemonics import find_curves, resolve, register_aliases
    from petrolib.processing import set_alias, process_data, trim, model_facies
    from petrolib.petrophysics import Quanti, archie, multimineral
    from petrolib.plots import tripleCombo, crossPlot, bucklesPlot, plotLoc
    from petrolib.automl import LogImputer


@pytest.mark.parametrize('old, names', [
    ('petrolib.workflow', ['Quanti']),
    ('petrolib.file_reader', ['load_las', 'load_table']),
    ('petrolib.procs', ['process_data', 'model_facies']),
    ('petrolib.interp', ['crossPlot', 'picketPlot']),
    ('petrolib.utils', ['MnemonicError']),
])
def test_deprecated_modules_still_work(old, names):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match='deprecated'):
        module = importlib.import_module(old)
    for n in names:
        assert hasattr(module, n)


def test_deprecated_zonation_alias():
    from petrolib.io import Zonation
    with pytest.warns(DeprecationWarning):
        from petrolib.plots import Zonation as Old
    assert Old is Zonation


def test_importing_petrolib_does_not_warn():
    for m in [m for m in sys.modules if m.startswith('petrolib')]:
        del sys.modules[m]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        import petrolib
        petrolib.petrophysics, petrolib.plots, petrolib.io
