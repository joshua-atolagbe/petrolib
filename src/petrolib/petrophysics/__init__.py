'''
Petrophysics: equations, and the Quanti zone-by-zone workflow

Modules
-------
shale         gamma ray index and volume of shale
porosity      density, neutron-density and sonic porosity
saturation    Archie, Simandoux, Indonesia, Waxman-Smits and dual water saturation, flushed zone and movable hydrocarbons
fluids        formation temperature and water resistivity (Arps, Rwa, salinity, SP)
permeability  Timur, Coates, Morris-Biggs and Wyllie-Rose permeability, Buckles SWIRR
rocktype      RQI, FZI, discrete rock types, Winland R35 and Lorenz curves
capillary     capillary pressure, Leverett J-function and saturation-height fits
lithology     M-N, RHOMAA/UMAA and multi-mineral volumes
netpay        net pay flags, summation and cutoff sensitivity
volumetrics   STOIIP and GIIP
uncertainty   Monte Carlo water saturation
geomechanics  elastic moduli, brittleness, strength, overburden, pore and fracture pressure, Gassmann
quanti        Quanti workflow class

The functions take pandas Series, numpy arrays or floats, and return the same kind

Example
-------
>>> from petrolib.petrophysics import Quanti, archie, fzi
>>> df['SW_ARCHIE'] = archie(df['PHIE'], df['RT'], rw=.04)
'''

from .shale import VSHALE_METHODS, gamma_ray_index, vshale, vshale_nd, vshale_sp
from .porosity import (density_porosity, neutron_density_porosity, shale_corrected_nd_porosity,
                       sonic_porosity, raymer_porosity, effective_porosity)
from .saturation import (archie, simandoux, indonesia, waxman_smits, dual_water, juhasz_b, qv_from_cec,
                         bulk_volume_water, flushed_zone_saturation, movable_hydrocarbon_index, movable_hydrocarbons)
from .fluids import formation_temperature, arps, rw_from_salinity, rwa, rw_from_sp
from .permeability import (PERMEABILITY_METHODS, permeability, timur, coates, morris_biggs, wyllie_rose,
                           swirr_buckles)
from .rocktype import rqi, normalized_porosity, fzi, drt, winland_r35, pore_type, lorenz
from .capillary import pc_from_height, height_from_pc, leverett_j, fit_saturation_height, sw_from_j
from .lithology import minerals, m_n, rhomaa, dtmaa, umaa, multimineral
from .netpay import net_flags, sample_thickness, weighted_mean, summarize, cutoff_sensitivity
from .volumetrics import stoiip, giip
from .uncertainty import monte_carlo_archie
from .geomechanics import (velocity, vs_castagna, elastic_moduli, brittleness, ucs_mcnally, hydrostatic_pressure,
                           overburden, normal_compaction_trend, pore_pressure_eaton, fracture_pressure_eaton,
                           to_gradient, wood_fluid_modulus, gassmann)
from .quanti import Quanti

__all__ = [
    'Quanti',
    'VSHALE_METHODS', 'gamma_ray_index', 'vshale', 'vshale_nd', 'vshale_sp',
    'density_porosity', 'neutron_density_porosity', 'shale_corrected_nd_porosity', 'sonic_porosity',
    'raymer_porosity', 'effective_porosity',
    'archie', 'simandoux', 'indonesia', 'waxman_smits', 'dual_water', 'juhasz_b', 'qv_from_cec', 'bulk_volume_water',
    'flushed_zone_saturation', 'movable_hydrocarbon_index', 'movable_hydrocarbons',
    'formation_temperature', 'arps', 'rw_from_salinity', 'rwa', 'rw_from_sp',
    'PERMEABILITY_METHODS', 'permeability', 'timur', 'coates', 'morris_biggs', 'wyllie_rose', 'swirr_buckles',
    'rqi', 'normalized_porosity', 'fzi', 'drt', 'winland_r35', 'pore_type', 'lorenz',
    'pc_from_height', 'height_from_pc', 'leverett_j', 'fit_saturation_height', 'sw_from_j',
    'minerals', 'm_n', 'rhomaa', 'dtmaa', 'umaa', 'multimineral',
    'net_flags', 'sample_thickness', 'weighted_mean', 'summarize', 'cutoff_sensitivity',
    'stoiip', 'giip',
    'monte_carlo_archie',
    'velocity', 'vs_castagna', 'elastic_moduli', 'brittleness', 'ucs_mcnally', 'hydrostatic_pressure', 'overburden',
    'normal_compaction_trend', 'pore_pressure_eaton', 'fracture_pressure_eaton', 'to_gradient', 'wood_fluid_modulus',
    'gassmann',
]
