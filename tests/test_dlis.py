import numpy as np
import pandas as pd
import pytest

from petrolib.io import load_well
from petrolib.mnemonics import standardize


@pytest.fixture
def las_file(tmp_path, well):
    lasio = pytest.importorskip('lasio')
    las = lasio.LASFile()
    las.well.WELL.value = 'W-1'
    las.append_curve('DEPT', well['DEPTH'].to_numpy(), unit='m', descr='Depth')
    las.append_curve('GR_EDTC', well['GR'].to_numpy(), unit='gAPI', descr='Gamma Ray')
    las.append_curve('ILD', well['RT'].to_numpy(), unit='ohm.m', descr='Deep induction')
    las.append_curve('NPOR', well['NPHI'].to_numpy() * 100, unit='%', descr='Neutron porosity')
    las.append_curve('ZDEN', well['RHOB'].to_numpy() * 1000, unit='kg/m3', descr='Bulk density')
    path = tmp_path / 'well.las'
    las.write(str(path), version=2.0)
    return path


def test_load_well_las(las_file, well):
    raw = load_well(las_file)
    assert list(raw.columns) == ['DEPT', 'GR_EDTC', 'ILD', 'NPOR', 'ZDEN']
    assert raw.attrs['petrolib']['units']['NPOR'] == '%' and raw.attrs['petrolib']['well'] == 'W-1'

    df = load_well(las_file, standardize=True, curves=['GR', 'RT', 'NPHI', 'RHOB'])
    assert list(df.columns) == ['DEPTH', 'GR', 'RT', 'NPHI', 'RHOB']
    assert np.allclose(df['NPHI'], well['NPHI'], atol=1e-4) and np.allclose(df['RHOB'], well['RHOB'], atol=1e-4)


def test_load_well_csv(tmp_path, well):
    path = tmp_path / 'well.csv'
    well.to_csv(path, index=False)
    df = load_well(path, curves=['GR', 'RT'])
    assert list(df.columns) == ['DEPTH', 'GR', 'RT']
    other = tmp_path / 'well.xyz'
    other.write_text('x')
    with pytest.raises(ValueError, match='Unsupported'):
        load_well(other)


# ---- DLIS ------------------------------------------------------------------------------------

@pytest.fixture
def dlis_file(tmp_path, well):
    pytest.importorskip('dlisio')
    dliswriter = pytest.importorskip('dliswriter')
    n = len(well)
    f = dliswriter.DLISFile()
    lf = f.add_logical_file()
    lf.add_origin('ORIGIN', well_name='W-1', field_name='TEST')
    main = [lf.add_channel('TDEP', data=well['DEPTH'].to_numpy() / .00254, units='0.1 in', long_name='Depth'),
            lf.add_channel('GR_EDTC', data=well['GR'].to_numpy(), units='gAPI', long_name='Gamma Ray'),
            lf.add_channel('AT90', data=well['RT'].to_numpy(), units='ohm.m', long_name='Deep induction resistivity'),
            lf.add_channel('TNPH', data=well['NPHI'].to_numpy() * 100, units='%', long_name='Thermal neutron porosity'),
            lf.add_channel('RHOZ', data=np.where(np.arange(n) == 5, -999.25, well['RHOB'].to_numpy()), units='g/cm3',
                           long_name='Bulk density'),
            lf.add_channel('IMG', data=np.random.default_rng(0).random((n, 8)), long_name='Image')]
    lf.add_frame('MAIN', channels=main, index_type='BOREHOLE-DEPTH')
    hires = [lf.add_channel('TDEP_HR', data=np.linspace(10000, 10100, 2 * n) / .254, units='0.1 in'),
             lf.add_channel('HCAL', data=np.full(2 * n, 8.5), units='in', long_name='Caliper')]
    lf.add_frame('HIRES', channels=hires, index_type='BOREHOLE-DEPTH')
    path = tmp_path / 'well.dlis'
    f.write(str(path))
    return path


def test_dlis_summary(dlis_file):
    from petrolib.io import dlis_summary
    s = dlis_summary(dlis_file)
    assert set(s['frame']) == {'MAIN', 'HIRES'} and s.loc[s['channel'] == 'IMG', 'dimension'].item() == '[8]'
    assert s.loc[s['index'], 'channel'].tolist() == ['TDEP', 'TDEP_HR'] and set(s['well']) == {'W-1'}


def test_load_dlis(dlis_file, well):
    from petrolib.io import load_dlis
    df = load_dlis(dlis_file)
    assert list(df.columns) == ['TDEP', 'GR_EDTC', 'AT90', 'TNPH', 'RHOZ']                # image channel skipped
    assert np.allclose(df['TDEP'], well['DEPTH'])                                        # 0.1 in -> m
    assert np.isnan(df['RHOZ'][5])                                                       # absent value
    meta = df.attrs['petrolib']
    assert meta['well'] == 'W-1' and meta['frame'] == 'MAIN' and meta['units']['TDEP'] == 'm'

    assert list(load_dlis(dlis_file, frame='HIRES').columns) == ['TDEP_HR', 'HCAL']
    assert list(load_dlis(dlis_file, curves=['HCAL']).columns) == ['TDEP_HR', 'HCAL']    # frame chosen by curves
    assert np.allclose(load_dlis(dlis_file, depth_unit='ft')['TDEP'], well['DEPTH'] / .3048)
    with pytest.raises(ValueError, match='not scalar'):
        load_dlis(dlis_file, curves=['IMG'], frame='MAIN')

    std = standardize(df)
    assert list(std.columns) == ['DEPTH', 'GR', 'RT', 'NPHI', 'RHOB']
    assert np.allclose(std['NPHI'], well['NPHI'])                                        # % -> fraction


def test_load_well_dlis(dlis_file):
    df = load_well(dlis_file, standardize=True, curves=['GR', 'RT'])
    assert list(df.columns) == ['DEPTH', 'GR', 'RT']
