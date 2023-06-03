Overview
--------
This is a python software package designed to help geoscientists perform quick formation evaluation workflow by estimating reservoir petrophysical parameters such as:

* Volume of Shale using various methods like Clavier, Stieber and Larionov methods
* Porosity - Effective and Total porosities using the Density and Wyllie's sonic methods.
* Water Saturation - using both archie and simmandoux methods
* Permeability

In addition to estimating these parameters, log plots are automatically displayed for proper interpretation. Also a pay summary result/dataframe is produced to help quantify the over-all quality of the reservoirs. Cutoff such as the porosity, shale volume and water saturation are applied to flag pay regions. The pay summary include:

* net, gross and not net thicknesses
* % net-to-gross
* average volume of shale
* average porosity
* bulk volume of water
* water saturation.

Interestingly, the parameters are computed and displayed only for the zones of interest picked. Plots such as neutron-density and pickett are available for reservoir assessment. Geolocations of the wells can also be visualised.

**Installation** 

You can install the package into your local machine using the `pip` command:
:: 
   pip install petrolib


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