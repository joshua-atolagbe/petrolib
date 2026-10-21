Quickstart 
==========

import necessaries
::
    from pathlib import Path
    from petrolib.io import load_las, Zonation
    from petrolib.processing import process_data
    from petrolib.petrophysics import Quanti
    from petrolib.plots import tripleCombo, plotLog

Load data
::
    well_path = Path(r"./15_9-19.las")
    tops_path = Path(r'./well tops.csv')

    df, las = load_las(well_path, return_csv=True, curves=['GR', 'RT', 'NPHI', 'RHOB'])

Process data
::
    df = process_data(df, 'GR', 'RT', 'NPHI', 'RHOB')

Triple combo
::
    %matplotlib inline
    tripleCombo(df, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', ztop=3300,
                zbot=3450, res_thres=10, fill='right', palette_op='rainbow', limit='left')
                
Zone plot
::
    zones = Zonation(df, path=tops_path)
    zones.plotZone('DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, '15_9-19')
    plotLog(df, 'DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, '15_9-19')

Calling the zonation object to extra info
::
    ztop, zbot, zn, fm = zones()

Formation evaluation
::
    pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', use_mean=True)
    vsh = pp.vshale(method='clavier', show_plot=True, palette_op='cubehelix', figsize=(9,12))
    por = pp.porosity(method='density', show_plot=True, figsize=(10, 12))
    sw = pp.water_saturation(method='archie', show_plot=True, figsize=(10, 12))
    perm = pp.permeability(show_plot=True, figsize=(9, 10))
    flags = pp.flags(por_cutoff=.12, vsh_cutoff=.5, sw_cutoff=0.8, show_plot=True, palette_op='cubehelix', figsize=(20, 15))

    ps = pp.paySummary(name='15-9_F1A')

Save results to excel
::
    pp.save(file_name='Pay Summary')