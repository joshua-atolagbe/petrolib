'''
Loading of DLIS files (RP66 v1) with dlisio

'''
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

#length units of DLIS depth indexes, in metres
_LENGTH = {'m': 1., 'meter': 1., 'metre': 1., 'meters': 1., 'metres': 1., 'ft': .3048, 'f': .3048, 'feet': .3048,
           'in': .0254, '0.1in': .00254, '0.1 in': .00254, 'cm': .01, 'mm': .001}

#absent values commonly written in DLIS files
NULL_VALUES = (-999.25, -9999., -999., -9999.25)


def _dlis():

    '''
    dlisio's DLIS module, imported on first use
    '''

    try:
        from dlisio import dlis
    except ImportError as e:
        raise ImportError("Reading DLIS files needs dlisio: pip install 'petrolib[dlis]'") from e
    return dlis


def _text(value) -> str:
    if value is None:
        return ''
    return str(getattr(value, 'name', value)) if not isinstance(value, str) else value


def _is_scalar(channel) -> bool:
    return list(channel.dimension or [1]) == [1]


def dlis_summary(file:Path|str) -> pd.DataFrame:

    r'''
    One row per channel of a DLIS file: logical file, frame, channel name, units, long name,
    dimension and whether it is the frame index. Use it to find the curves and frames to load

    Example
    -------
    >>> from petrolib.io import dlis_summary
    >>> dlis_summary('well.dlis').query('dimension == "[1]"')
    '''

    dlis = _dlis()
    rows = list()
    with dlis.load(str(file)) as files:
        for i, f in enumerate(files):
            well = _text(f.origins[0].well_name) if f.origins else ''
            for frame in f.frames:
                for j, ch in enumerate(frame.channels):
                    rows.append({'logical_file': i, 'well': well, 'frame': frame.name, 'channel': ch.name,
                                 'units': _text(ch.units), 'long_name': _text(ch.long_name),
                                 'dimension': str(list(ch.dimension or [1])), 'index': j == 0})
    return pd.DataFrame(rows)


def load_dlis(file:Path|str, curves:list=None, frame:str|int=None, logical_file:int=0, depth_unit:str='m',
              null_values:tuple=NULL_VALUES) -> pd.DataFrame:

    r'''
    Loads one frame of a DLIS file into a DataFrame, one column per scalar channel. Array channels
    (images, waveforms) are skipped; read them with dlisio directly

    Parameters
    ----------
    file : pathlib.Path or str
        DLIS file

    curves : list of str default None
        Channels to load, besides the index. Default: all scalar channels of the frame

    frame : str or int default None
        Frame name or position. Default: the frame with the most of `curves`, or with the most
        scalar channels. See `dlis_summary` for the frames of a file

    logical_file : int default 0
        Logical file to read, for files with several

    depth_unit : str default 'm'
        Unit to convert a depth index to: 'm' or 'ft'. DLIS depths are often stored in 0.1 in.
        None keeps the stored unit

    null_values : tuple
        Values replaced with NaN

    Returns
    -------
    DataFrame whose first column is the frame index (e.g. TDEP). Units, long names, the well name
    and the frame are in `df.attrs['petrolib']`, which `petrolib.mnemonics.standardize` uses

    Example
    -------
    >>> from petrolib.io import load_dlis
    >>> from petrolib.mnemonics import standardize
    >>> df = standardize(load_dlis('well.dlis'))      # DEPTH in m, GR, RT, NPHI, RHOB, DT, ...
    '''

    dlis = _dlis()
    if not Path(file).exists():
        raise FileNotFoundError(f'{file} path does not exists.')

    with dlis.load(str(file)) as files:
        f = files[logical_file]
        frames = list(f.frames)
        if not frames:
            raise ValueError(f'No frame in logical file {logical_file} of {file}')

        if isinstance(frame, int):
            fr = frames[frame]
        elif isinstance(frame, str):
            names = [x.name for x in frames]
            if frame not in names:
                raise ValueError(f"Frame '{frame}' not in {file}. Frames: {names}")
            fr = frames[names.index(frame)]
        else:
            def score(x):
                scalar = [c.name for c in x.channels if _is_scalar(c)]
                return (len(set(scalar) & set(curves)) if curves else 0, len(scalar))
            fr = max(frames, key=score)

        channels = list(fr.channels)
        index = channels[0]
        wanted = [c for c in channels[1:] if _is_scalar(c) and (curves is None or c.name in curves)]
        if curves is not None:
            missing = set(curves) - {c.name for c in wanted}
            if missing:
                raise ValueError(f'Channels {sorted(missing)} not scalar channels of frame {fr.name}')

        try:
            table = fr.curves()
            values = {c.name: table[c.name] for c in [index] + wanted}
            if len(values) < len(wanted) + 1:                          #repeated channel names
                raise ValueError
        except ValueError:
            values = dict()
            for c in [index] + wanted:
                name = c.name if c.name not in values else f'{c.name}_{len(values)}'
                values[name] = c.curves()

        units = {name: _text(c.units) for name, c in zip(values, [index] + wanted)}
        descriptions = {name: _text(c.long_name) for name, c in zip(values, [index] + wanted)}
        well = _text(f.origins[0].well_name) if f.origins else ''

    df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in values.items()})
    df = df.replace(list(null_values), np.nan)

    index_name = next(iter(values))
    unit = units[index_name].strip().lower()
    if depth_unit is not None and unit in _LENGTH:
        df[index_name] = df[index_name] * _LENGTH[unit] / _LENGTH[depth_unit]
        units[index_name] = depth_unit

    df.attrs['petrolib'] = {'source': str(file), 'well': well, 'frame': fr.name, 'units': units,
                            'descriptions': descriptions}
    return df
