.. Petrolib documentation master file, created by
   sphinx-quickstart on Thu Jun  1 22:30:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
Petrolib: Official Documentations
---------------------------------
.. image:: ./zones.png

Overview
--------
This is a python software package designed to help geoscientists perform quick formation evaluation workflow by estimating reservoir petrophysical parameters such as:

* Volume of Shale using various methods like Clavier, Stieber and Larionov methods
* Porosity - Effective and Total porosities using the Density and Wyllie's sonic methods.
* Water Saturation - using both archie and simmandoux methods
* Permeability

In addition to estimating these parameters, log plots are automatically displayed for proper interpretation. Also a pay summary result is generated in XLSX to help quantify the over-all quality of reservoirs. 


Installation 
============
You can install the package into your local machine using the `pip` command:
:: 
   pip install -U petrolib

To import the package, use:
::
   import petrolib as pl
   print(pl.__version__)

   >>> '1.2.6'

Contents
========
.. toctree::
   :maxdepth: 5

   overview 
   quickstart
   modules
   tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`