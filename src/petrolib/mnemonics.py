'''
Log curve mnemonics: recognize curves of any vendor or naming convention, and rename them to
standard names

Curves are matched in three passes:
    1. the mnemonic is one of the aliases of a curve type (case-insensitive), in order of preference
    2. the mnemonic without suffixes such as ':1', '_2', '_CORR' or '_EDTC' is an alias
    3. the curve description (LAS header or DLIS long name) matches the curve type

Standard curve types
--------------------
DEPTH, TVD, TVDSS, GR, SP, CALI, BS, RT (deep resistivity), RM (medium/shallow resistivity),
RXO (flushed zone resistivity), NPHI, RHOB, DRHO, PEF, DT (compressional slowness), DTS (shear slowness)

Example
-------
>>> from petrolib.mnemonics import find_curves, standardize
>>> find_curves(['DEPT', 'GR_EDTC', 'AT90', 'TNPH', 'RHOZ', 'DTCO'])
{'DEPTH': 'DEPT', 'GR': 'GR_EDTC', 'RT': 'AT90', 'NPHI': 'TNPH', 'RHOB': 'RHOZ', 'DT': 'DTCO'}
>>> df = standardize(df)          # columns renamed to DEPTH, GR, RT, NPHI, RHOB, DT
'''
from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd

from .exceptions import MnemonicError

#curve type -> mnemonics, most preferred first
ALIASES = {
    'DEPTH': ['DEPTH', 'DEPT', 'MD', 'TDEP', 'DEP', 'DEPTH_MD', 'MDEPTH', 'TDEPTH', 'INDEX'],
    'TVD':   ['TVD', 'TVDKB', 'TVD_KB', 'TVDRKB'],
    'TVDSS': ['TVDSS', 'TVD_SS', 'TVDMSL', 'SSTVD'],
    'GR':    ['GR', 'GRC', 'GR_EDTC', 'GRD', 'GRS', 'ECGR', 'GRR', 'GRN', 'GR_CAL', 'HGR', 'GRT', 'GAM', 'GAMMA',
              'GAPI', 'SGR', 'HSGR', 'CGR', 'HCGR'],
    'SP':    ['SP', 'SSP', 'SPBR', 'SPC', 'SPR', 'SPCG'],
    'CALI':  ['CALI', 'CAL', 'HCAL', 'CALX', 'CALY', 'CAL1', 'CAL2', 'C1', 'C2', 'C13', 'C24', 'CALS', 'CALD', 'LCAL'],
    'BS':    ['BS', 'BIT', 'BITSIZE', 'BIT_SIZE', 'BSZ'],
    'RT':    ['RT', 'RD', 'RDEP', 'RDEEP', 'ILD', 'LLD', 'HLLD', 'RLLD', 'RILD', 'IDPH', 'IDEP', 'AT90', 'AF90',
              'AO90', 'AHT90', 'AHO90', 'AHF90', 'M2R9', 'RLA5', 'HDRS', 'RES_DEEP', 'DRES', 'RT90', 'P40H',
              'RACEHM', 'RPCEHM', 'LL7', 'RTRUE', 'ILD1', 'RESD', 'RESDEEP', 'DEEPRES', 'RES'],
    'RM':    ['RM', 'RMED', 'ILM', 'IMPH', 'IMED', 'RILM', 'LLS', 'RLLS', 'HLLS', 'AT60', 'AT30', 'AF60', 'AF30',
              'AO60', 'AO30', 'M2R6', 'M2R3', 'RLA3', 'RLA2', 'HMRS', 'RES_MED', 'MRES', 'SFLU', 'SFL', 'RS',
              'RSHAL', 'RES_SHAL', 'P16H', 'P28H', 'RESM', 'RESS', 'RESMED', 'RESSHAL'],
    'RXO':   ['RXO', 'RXOZ', 'RXO8', 'MSFL', 'MCFL', 'MLL', 'RXOI', 'HMNO', 'RMICRO', 'RESMICRO', 'RMIC'],
    'NPHI':  ['NPHI', 'TNPH', 'NPOR', 'NPHIL', 'NPHS', 'NPLS', 'NPRL', 'CNL', 'CNC', 'CNCF', 'CNCLS', 'CNLS',
              'HNPO', 'APLC', 'NPL', 'PHIN', 'NEU', 'NEUT', 'NEUTRON', 'CN', 'BPHI', 'NPSS', 'CNSS', 'CNCSS'],
    'RHOB':  ['RHOB', 'RHOZ', 'ZDEN', 'DEN', 'DENS', 'RHO8', 'DENB', 'DENC', 'ZDNC', 'RHOM', 'HROM', 'BDEL',
              'BDCX', 'RHO', 'DENSITY'],
    'DRHO':  ['DRHO', 'HDRA', 'ZCOR', 'DCOR', 'DRH', 'DRHB', 'QRHO', 'CORR'],
    'PEF':   ['PEF', 'PE', 'PEFZ', 'PEF8', 'PEFL', 'PEFA', 'PDPE', 'PEB', 'PEFS'],
    'DT':    ['DT', 'DTC', 'DTCO', 'DT4P', 'DTCO3', 'DTLN', 'DTLF', 'DTL', 'DTP', 'DTCOMP', 'AC', 'ACC', 'ACCO',
              'ACL', 'DT24', 'SONIC', 'DELT'],
    'DTS':   ['DTS', 'DTSM', 'DTSH', 'DT4S', 'DTSD', 'DTS1', 'DTSHEAR', 'ACS'],
}

#curve type -> regular expression on the lower-case curve description
DESCRIPTIONS = {
    'DRHO':  r'density correction|delta.?rho|density.*correction',
    'DTS':   r'shear',
    'TVDSS': r'(sub.?sea|msl).*vertical|vertical.*(sub.?sea|msl)',
    'TVD':   r'true vertical',
    'RXO':   r'micro.*(resist|laterolog|spherical)|flushed zone|\brxo\b',
    'RM':    r'(medium|shallow).*(resist|induction|laterolog)|(resist|induction|laterolog).*(medium|shallow)',
    'RT':    r'(deep|true).*(resist|induction|laterolog)|(resist|induction|laterolog).*\bdeep',
    'NPHI':  r'neutron',
    'RHOB':  r'density(?!.*(correct|porosity))',
    'PEF':   r'photo.?electric|\bpe\b',
    'DT':    r'compressional|sonic|slowness|transit time|delta.?t\b',
    'CALI':  r'caliper|hole diameter|borehole diameter',
    'SP':    r'spontaneous|self.?potential',
    'GR':    r'gamma',
    'BS':    r'bit size',
    'DEPTH': r'\bdepth\b',
}

#suffixes dropped before the second matching pass
_SUFFIXES = {'CORR', 'COR', 'C', 'EDTC', 'FINAL', 'FIN', 'NORM', 'RAW', 'COMP', 'MERGED', 'MRG', 'SPLICE', 'SPL',
             'LS', 'SS', 'QC', 'EDIT', 'ED', 'MAIN', 'REPEAT', 'REP'}

#porosity curves petrolib computes, shown with a reversed axis like neutron logs
_POROSITY = {'PHIE', 'PHIT', 'PHID', 'PHIN', 'PHIS', 'PHIND', 'PHI', 'POR', 'PHIE_ND', 'PHIT_ND'}

#unit conversions done by `standardize`: curve type -> (units, factor, new unit)
_UNIT_FIXES = {
    'NPHI': ({'%', 'pu', 'p.u.', 'percent', 'pct', 'v/v%', 'pu(ls)', 'pu(ss)'}, .01, 'v/v'),
    'RHOB': ({'kg/m3', 'kg/m^3', 'kg/m**3'}, .001, 'g/cm3'),
    'DT':   ({'us/m', 'µs/m', 'usec/m', 'uspm'}, 1 / 3.28084, 'us/ft'),
    'DTS':  ({'us/m', 'µs/m', 'usec/m', 'uspm'}, 1 / 3.28084, 'us/ft'),
}


def _lookup() -> dict:
    return {alias: kind for kind, aliases in ALIASES.items() for alias in reversed(aliases)}


_LOOKUP = _lookup()


def base_name(mnemonic) -> str:

    r'''
    Mnemonic in upper case without separators and common suffixes, e.g. 'GR:1' -> 'GR',
    'RHOB_CORR' -> 'RHOB', 'NPHI_LS' -> 'NPHI'
    '''

    s = re.split(r'[:.\[\s]', str(mnemonic).upper().strip())[0]
    parts = s.split('_')
    while len(parts) > 1 and (parts[-1].isdigit() or parts[-1] in _SUFFIXES):
        parts.pop()
    return '_'.join(parts)


def log_kind(mnemonic) -> str|None:

    r'''
    Standard curve type of a mnemonic, or None if it is not recognized

    Example
    -------
    >>> log_kind('AT90'), log_kind('rhoz'), log_kind('GR:1')
    ('RT', 'RHOB', 'GR')
    '''

    return _LOOKUP.get(str(mnemonic).upper().strip()) or _LOOKUP.get(base_name(mnemonic))


def is_resistivity(mnemonic) -> bool:

    r'''
    True for deep, medium and flushed zone resistivity curves, which are shown on a log scale
    '''

    return log_kind(mnemonic) in ('RT', 'RM', 'RXO')


def is_porosity(mnemonic) -> bool:

    r'''
    True for neutron and computed porosity curves, which are shown with a reversed axis
    '''

    return log_kind(mnemonic) == 'NPHI' or base_name(mnemonic) in _POROSITY


def register_aliases(kind:str, mnemonics:list, description:str=None):

    r'''
    Adds mnemonics of your data to a curve type, ahead of the built-in ones. A new curve type
    is created if `kind` does not exist

    Parameters
    ----------
    kind : str
        Curve type, e.g. 'RT'

    mnemonics : list of str
        Mnemonics to recognize as that type, most preferred first

    description : str default None
        Regular expression on lower-case curve descriptions for that type

    Example
    -------
    >>> register_aliases('RT', ['RDEEP_XYZ'])
    '''

    global _LOOKUP
    upper = [str(m).upper() for m in mnemonics]
    ALIASES[kind] = upper + [m for m in ALIASES.get(kind, []) if m not in upper]
    if description is not None:
        DESCRIPTIONS[kind] = description
    _LOOKUP = _lookup()


def _columns_and_descriptions(data) -> tuple[list, dict]:

    '''
    Curve names and descriptions of a DataFrame, a lasio LASFile or a list of names
    '''

    if isinstance(data, pd.DataFrame):
        descr = (data.attrs.get('petrolib') or {}).get('descriptions', {})
        return list(data.columns), dict(descr)
    if hasattr(data, 'curves') and hasattr(data, 'well'):                 #lasio.LASFile
        return [c.mnemonic for c in data.curves], {c.mnemonic: c.descr for c in data.curves}
    return list(data), {}


def find_curves(data, kinds:list=None, descriptions:dict=None) -> dict:

    r'''
    Finds the curve of each standard type

    Parameters
    ----------
    data : pd.DataFrame, lasio.LASFile or list of str
        Data, or curve names

    kinds : list of str default None
        Curve types to look for. Default: all types in ALIASES

    descriptions : dict default None
        Curve name -> description, for the third pass. Read from LAS files and from DataFrames
        loaded with `petrolib.io.load_well` or `load_dlis` automatically

    Returns
    -------
    dict of curve type -> curve name, for the types found

    Example
    -------
    >>> find_curves(df, kinds=['GR', 'RT', 'NPHI', 'RHOB'])
    {'GR': 'GR_EDTC', 'RT': 'AT90', 'NPHI': 'TNPH', 'RHOB': 'RHOZ'}
    '''

    columns, descr = _columns_and_descriptions(data)
    descr.update(descriptions or {})
    kinds = list(kinds) if kinds is not None else list(ALIASES)
    unknown = [k for k in kinds if k not in ALIASES]
    if unknown:
        raise ValueError(f'Unknown curve types {unknown}. Use {list(ALIASES)} or register_aliases()')

    found, used = dict(), set()
    for name_of in (lambda c: str(c).upper().strip(), base_name):
        names = {c: name_of(c) for c in columns}
        for kind in kinds:
            if kind in found:
                continue
            for alias in ALIASES[kind]:
                hit = next((c for c in columns if c not in used and names[c] == alias), None)
                if hit is not None:
                    found[kind] = hit
                    used.add(hit)
                    break

    for kind in [k for k in DESCRIPTIONS if k in kinds and k not in found]:
        hit = next((c for c in columns if c not in used and descr.get(c)
                    and re.search(DESCRIPTIONS[kind], str(descr[c]).lower())), None)
        if hit is not None:
            found[kind] = hit
            used.add(hit)

    return {k: found[k] for k in kinds if k in found}


def resolve(data, name:str, kind:str=None) -> str:

    r'''
    Column of `data` to use for `name`: `name` itself if it is a column, otherwise the recognized
    curve of the same type (e.g. 'AT90' for 'RT'). Raises MnemonicError if there is none

    Example
    -------
    >>> resolve(df, 'RT')
    'AT90'
    '''

    columns = list(data.columns) if isinstance(data, pd.DataFrame) else list(data)
    if name in columns:
        return name
    kind = kind or (name if name in ALIASES else log_kind(name))
    hit = find_curves(data, [kind]).get(kind) if kind in ALIASES else None
    if hit is None:
        what = f' and no {kind} curve was recognized' if kind else ''
        raise MnemonicError(f"No '{name}' curve in the data{what}. Columns: {columns}. "
                            f"Pass the column name, or add it with petrolib.mnemonics.register_aliases")
    return hit


def standardize(df:pd.DataFrame, mapping:dict=None, kinds:list=None, units:dict=None, descriptions:dict=None,
                convert_units:bool=True, return_mapping:bool=False):

    r'''
    Renames recognized curves to the standard names (DEPTH, GR, RT, NPHI, RHOB, DT, ...) and
    converts units where the unit is unambiguous:

        NPHI in % or p.u.  -> fraction (also when no unit is known and the median is above 1.5)
        RHOB in kg/m3      -> g/cm3    (also when no unit is known and the median is above 100)
        DT, DTS in us/m    -> us/ft

    Other columns are kept unchanged. The curve mapping and conversions are stored in
    `df.attrs['petrolib']`

    Parameters
    ----------
    df : pd.DataFrame
        Log data

    mapping : dict default None
        Curve type -> column, to override what is found automatically, e.g. {'RT': 'RLA3'}

    kinds : list default None
        Curve types to standardize. Default: all

    units, descriptions : dict default None
        Column -> unit and column -> description. Default: from `df.attrs['petrolib']`, which
        `petrolib.io.load_well` and `load_dlis` fill in

    convert_units : bool default True
        Apply the unit conversions above

    return_mapping : bool default False
        Also return the curve type -> original column mapping

    Example
    -------
    >>> df = standardize(load_well('well.dlis'))
    >>> df, mapping = standardize(df, mapping={'RT': 'RLA3'}, return_mapping=True)
    '''

    meta = dict(df.attrs.get('petrolib') or {})
    units = {**meta.get('units', {}), **(units or {})}
    descriptions = {**meta.get('descriptions', {}), **(descriptions or {})}
    found = find_curves(df, kinds, descriptions)
    for kind, col in (mapping or {}).items():
        #a column given by the user is no longer used for the type it was found for
        found = {k: c for k, c in found.items() if c != col}
        found[kind] = col

    rename = dict()
    for kind, col in list(found.items()):
        if col not in df.columns:
            raise MnemonicError(f"'{col}' given for {kind} is not a column")
        if col != kind and kind in df.columns:
            warnings.warn(f"Not renaming '{col}' to {kind}: a '{kind}' column already exists")
            found[kind] = kind
        elif col != kind:
            rename[col] = kind

    out = df.rename(columns=rename)
    conversions = list()
    if convert_units:
        for kind, (unit_set, factor, new_unit) in _UNIT_FIXES.items():
            if kind not in found:
                continue
            unit = str(units.get(found[kind], '') or '').strip().lower().replace(' ', '')
            values = pd.to_numeric(out[kind], errors='coerce')
            median = np.nanmedian(values) if values.notna().any() else np.nan
            guess = not unit and ((kind == 'NPHI' and median > 1.5) or (kind == 'RHOB' and median > 100))
            if unit in unit_set or guess:
                out[kind] = values * factor
                conversions.append(f"{kind}: {unit or 'unknown unit, median ' + format(median, '.3g')} -> {new_unit}")
                units[found[kind]] = new_unit

    out.attrs['petrolib'] = {**meta, 'curves': dict(found), 'conversions': conversions,
                             'units': {rename.get(c, c): u for c, u in units.items()},
                             'descriptions': {rename.get(c, c): d for c, d in descriptions.items()}}
    return (out, found) if return_mapping else out
