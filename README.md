# Petrophysical Evaluation Package

This is a python package designed to help users perform petrophysical analysis by estimating petrophysical parameters such as:
```
* Volume of Shale using various methods like Clavier, Stieber and Larionov methods
* Porosity - Effective and Total porosities using the density and Wyllie's sonic methods.
* Water Saturation - using both archie and simmandoux methods
* Permeability
```

In addition to estimating these parameters, log plots are automatically displayed for proper interpretation. Also a pay summary result/dataframe is produced to help quantify the over-all quality of the reservoirs. Cutoff such as the porosity, shale volume and water saturation are applied to flag pay regions. The pay summary include:

* net, gross and not net thicknesses
* % net-to-gross 
* average volume of shale
* average porosity
* bulk volume of water
* water saturation.

Interestingly, the parameters are computed and displayed only for the zones of interest picked. Plots such as neutron-density and pickett are available for reservoir assessment. Geolocations of the wells can also be visualised.

### Hierarachy

The package is divided into three sections:

1. **Data**. This further contains three submodules that handle:
> Loading of well data
> Processing of well log data
> Statistical analysis such as log frequencies and correlation

2. **Visualization**. This also contains two submodules to handle:
> Well log visualisation 
> Plot well locations on an actual map
> Facilitates the loading of well tops.
> Plot log curves along with zonation tracks
> Neutron-density cross plot
> Pickett Plot

3. **Workflow**. This is a workflow module that perform the actual petrophysical analysis from shale volume to pay summary result computation.

### Quick tutorial
```
#import necessaries
from pathlib import Path
from petrolib.data import procs
from petrolib.data import file_reader as fr
from petrolib.petro.workflow import Quanti
from petrolib.visualization.plot import tripleCombo, Zonation, plotLog

#load data
well_path = Path(r"C:\Users\USER\Documents\petrolib\15_9-19.las")
tops_path = Path(r'C:\Users\USER\Documents\petrolib\well tops.csv')

df, las = fr.load_las(well_path, return_csv=True, curves=['GR', 'RT', 'NPHI', RHOB'])

#process data
df = procs.process_data(df, 'GR', 'RT', 'NPHI', 'RHOB')

#triple combo
%matplotlib inline
tripleCombo(df, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', ztop=3300,
               zbot=3450, res_thres=10, fill='right', palette_op='rainbow', limit='left')
               
#zone plot
zones = Zonation(df, path=tops_path)
zones.plotZone('DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, '15_9-19')
plotLog('DEPTH', ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'], 3300, 3600, '15_9-19')

#calling the zonation object to extra info
ztop, zbot, zn, fm = zones()

#Petrophysics
pp = Quanti(df, zn, ztop, zbot, fm, 'DEPTH', 'GR', 'RT', 'NPHI', 'RHOB', use_mean=True)
vsh = pp.vshale(method='clavier', show_plot=True, palette_op='cubehelix', figsize=(9,12))
por = pp.porosity(method='density', show_plot=True, figsize=(10, 12))
sw = pp.water_saturation(method='archie', show_plot=True, figsize=(10, 12))
perm = pp.permeability(show_plot=True, figsize=(9, 10))
flags = pp.flags(por_cutoff=.12, vsh_cutoff=.5, sw_cutoff=0.8, show_plot=True, palette_op='cubehelix', figsize=(20, 15))

ps = pp.paySummary(name='15-9_F1A')

#save results to excel
pp.save(file_name='Pay Summary')
```

Tutorial [link]("https://github.com/mayor-of-geology/petrolib/tutorials")

