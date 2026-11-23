'''
Synthetic wells built from a known quartz-clay-water model, so results can be checked exactly
'''
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

#log responses of the model components
COMPONENTS = {
    'Quartz': {'RHOB': 2.65, 'NPHI': -.02, 'DT': 55.5, 'GR': 20.},
    'Illite': {'RHOB': 2.55, 'NPHI': .35, 'DT': 90., 'GR': 150.},
    'Water':  {'RHOB': 1.00, 'NPHI': 1.00, 'DT': 189., 'GR': 0.},
}
RW = .05


def make_well(name:str='W-1', top:float=1000., n:int=400, step:float=.5, seed:int=0) -> pd.DataFrame:

    '''
    Alternating sands and shales with logs computed from the component volumes, and RT from Archie
    '''

    rng = np.random.default_rng(seed)
    depth = top + step * np.arange(n)
    vcl = np.clip(.45 + .45 * np.sin(depth / 6.) + rng.normal(0, .03, n), 0, .9)
    phi = np.clip(.3 * (1 - vcl) - .03 + rng.normal(0, .01, n), .01, .35)
    vq = 1 - vcl - phi
    vol = {'Quartz': vq, 'Illite': vcl, 'Water': phi}
    logs = {log: sum(vol[c] * COMPONENTS[c][log] for c in COMPONENTS) for log in ('RHOB', 'NPHI', 'DT', 'GR')}
    sw = np.clip(.25 + .7 * vcl, .2, 1)
    rt = RW / (phi**2 * sw**2)
    return pd.DataFrame({'WELL': name, 'DEPTH': depth, 'GR': logs['GR'], 'RT': rt, 'NPHI': logs['NPHI'],
                         'RHOB': logs['RHOB'], 'DT': logs['DT'], 'PEF': 1.81 * vq + 3.4 * vcl + .36 * phi,
                         'VQ': vq, 'VCL': vcl, 'PHI': phi, 'SW_TRUE': sw})


@pytest.fixture
def well():
    return make_well()


@pytest.fixture
def wells():
    return pd.concat([make_well(f'W-{i}', top=1000. + 50 * i, seed=i) for i in range(1, 4)], ignore_index=True)


@pytest.fixture
def tops_csv(tmp_path):
    path = tmp_path / 'tops.csv'
    path.write_text('Well,Top (m MD),Base (m MD),Zone\nW-1,1010,1100,SAND_A\nW-1,1110,1190,SAND_B\n'
                    'W-2,1060,1150,SAND_A\n')
    return path
