Overview
--------
This is a python software package designed to help geoscientists perform quick formation evaluation workflow by estimating reservoir petrophysical parameters such as:

* Volume of Shale using various methods like Clavier, Stieber and Larionov methods, and neutron-density separation
* Porosity - Effective and Total porosities using the density, neutron-density, Wyllie's sonic and Raymer-Hunt-Gardner methods
* Water Saturation - Archie, Simandoux, Indonesia, Waxman-Smits and dual water
* Permeability - Timur, Coates, Morris-Biggs and Wyllie-Rose
* Formation water resistivity - Arps, Rwa and Rw from salinity
* Rock typing - RQI, FZI, discrete rock types, Winland R35 and Lorenz coefficients
* Saturation-height functions from the Leverett J-function
* Lithology - M-N and MID parameters, and a multi-mineral solver
* Volumetrics and Monte Carlo water saturation uncertainty
* Invasion: flushed zone saturation, movable hydrocarbon index, Rw and Vshale from the SP
* Geomechanics and rock physics: elastic moduli, brittleness, UCS, overburden, Eaton pore and fracture pressure, Gassmann
* Electrofacies by k-means clustering
* Filling of missing log curves with AutoGluon, field-wide or per well (``petrolib.automl``)
* Curves of any vendor's mnemonics, recognized and renamed to standard names (``petrolib.mnemonics``)
* LAS, DLIS, CSV and Excel files (``petrolib.io.load_well``)

In addition to estimating these parameters, log plots are automatically displayed for proper interpretation. Also a pay summary result/dataframe is produced to help quantify the over-all quality of the reservoirs. Cutoff such as the porosity, shale volume and water saturation are applied to flag pay regions. The pay summary include:

* net, gross and not net thicknesses
* % net-to-gross
* average volume of shale
* average porosity
* bulk volume of water
* water saturation (pore volume weighted)
* permeability
* porosity thickness (PHIH) and hydrocarbon pore thickness (HCPV)

Interestingly, the parameters are computed and displayed only for the zones of interest picked. Plots such as neutron-density and pickett are available for reservoir assessment. Geolocations of the wells can also be visualised.

**Installation** 

You can install the package into your local machine using the `pip` command:
:: 
   pip install -U petrolib


**Functionalities**

The package is designed to handle:

* Loading of well data
* Processing of well log data
* Statistical analysis such as log frequencies and correlation
* Well log visualization
* Plot well locations on an actual map
* Facilitates the loading of well tops.
* Plot log curves along with zonation tracks
* Neutron-density cross plot
* Pickett Plot
* Buckles, Winland R35, RQI, Lorenz, M-N, MID and Thomas-Stieber plots
* Net pay cutoff sensitivity