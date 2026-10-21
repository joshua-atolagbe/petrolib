import pandas as pd
import pytest

from petrolib.io import Zonation, load_table

COLS = dict(top_col='Top (m MD)', bot_col='Base (m MD)', zone_col='Zone')


def test_zonation_multi_well_file(tops_csv):
    ztop, zbot, zn, fm = Zonation(pd.DataFrame(), path=tops_csv, well='W-1', **COLS)()
    assert ztop == [1010., 1110.] and zbot == [1100., 1190.] and zn == ['SAND_A', 'SAND_B'] and fm == [1055., 1150.]


def test_zonation_default_format_with_spaces_and_bom(tmp_path):
    path = tmp_path / 'tops.csv'
    path.write_bytes(b'\xef\xbb\xbftop, bottom, zonename\n100, 200, RES_A\n')
    assert Zonation(pd.DataFrame(), path=path)()[:3] == ([100.], [200.], ['RES_A'])


def test_zonation_from_dicts():
    z = Zonation(pd.DataFrame(), zones=[{'RES_A': [100, 200]}, {'RES_B': [210, 220]}])
    assert z()[:3] == ([100., 210.], [200., 220.], ['RES_A', 'RES_B'])


@pytest.mark.parametrize('kwargs, message', [
    (dict(), 'either'),
    (dict(well='W-1'), 'not in'),                    # default column names missing
    (dict(**COLS), 'several wells'),
    (dict(well='NOPE', **COLS), 'Available wells'),
])
def test_zonation_errors(tops_csv, kwargs, message):
    if kwargs:
        kwargs['path'] = tops_csv
    with pytest.raises(ValueError, match=message):
        Zonation(pd.DataFrame(), **kwargs)


def test_load_table(tmp_path, well):
    path = tmp_path / 'well.csv'
    well.to_csv(path, index=False)
    assert list(load_table(path, curves=['GR', 'RT']).columns) == ['GR', 'RT']


def test_pay_summary_workbook_matches_the_results_style(tmp_path, well):
    openpyxl = pytest.importorskip('openpyxl')
    from petrolib.petrophysics import Quanti
    q = Quanti(well, ['A', 'B'], [1010., 1100.], [1100., 1180.], [1055., 1140.])
    q.vshale(method='clavier')
    q.porosity(method='density', rhob_shale=2.55)
    q.water_saturation(rw=.05)
    q.permeability()
    q.flags(vsh_cutoff=.5, por_cutoff=.05, sw_cutoff=.6)
    q.paySummary('W-1')
    path = q.save(str(tmp_path / 'Pay Summary'))
    assert path.name == 'Pay Summary.xlsx'

    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ['Pay Summary', 'Log Data']
    ws = wb['Pay Summary']
    header = ws['A1']
    assert header.fill.fgColor.rgb.endswith('1F4E79') and header.font.bold and header.font.color.rgb.endswith('FFFFFF')
    assert ws.freeze_panes == 'A2' and ws.row_dimensions[1].height == 28
    flags = [c.value for c in ws['C'][1:]]
    fills = {f: ws.cell(row=i + 2, column=1).fill.fgColor.rgb[-6:] for i, f in enumerate(flags)}
    fonts = {f: ws.cell(row=i + 2, column=1).font.color.rgb[-6:] for i, f in enumerate(flags)}
    assert fills == {'ROCK': 'FFFF00', 'RES': '00AA00', 'PAY': 'CC0000'}
    assert fonts == {'ROCK': '000000', 'RES': 'FFFFFF', 'PAY': 'FFFFFF'}
    assert ws['A2'].border.left.style == 'thin'
    columns = [c.value for c in ws[1]]
    assert ws.cell(row=2, column=columns.index('NTG') + 1).number_format == '0.000'
    assert ws.cell(row=2, column=columns.index('Net') + 1).number_format == '0.00'
    log_columns = [c.value for c in wb['Log Data'][1]]
    assert {'Zone', 'VShale', 'PHIE', 'SW', 'PAY_NET_FLAG'} <= set(log_columns)
