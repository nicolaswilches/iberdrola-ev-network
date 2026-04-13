"""
Grid capacity classification logic.
"""

import pandas as pd

from src.constants import GRID_SUFFICIENT_MIN_MW, GRID_MODERATE_MIN_MW, DEFAULT_STATUS_IF_NO_SUBSTATION


def classify_grid_status(available_capacity_mw):
    """
    Classify grid status based on available capacity at nearest substation.

    Returns: 'Sufficient', 'Moderate', or 'Congested'
    """
    if available_capacity_mw is None:
        return DEFAULT_STATUS_IF_NO_SUBSTATION
    if available_capacity_mw >= GRID_SUFFICIENT_MIN_MW:
        return 'Sufficient'
    elif available_capacity_mw >= GRID_MODERATE_MIN_MW:
        return 'Moderate'
    else:
        return 'Congested'


def is_friction_point(grid_status):
    """Return True if this station is a friction point (Moderate or Congested)."""
    return grid_status in ['Moderate', 'Congested']


def consolidate_substations(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse voltage-level records into one row per physical substation.

    The raw distributor files frequently contain one row per transformer or
    voltage tap. Grouping by name alone is unsafe because some DSOs reuse
    substation names across provinces. We therefore key on distributor,
    substation name, and rounded coordinates to preserve distinct assets while
    still merging repeated voltage-level rows for the same physical site.
    """
    if grid_df is None or len(grid_df) == 0:
        return pd.DataFrame(columns=getattr(grid_df, 'columns', []))

    df = grid_df.copy()
    df['_lat_key'] = df['latitude'].round(6)
    df['_lon_key'] = df['longitude'].round(6)

    agg = {
        'provincia': 'first',
        'municipio': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'available_capacity_mw': 'max',
    }
    if 'voltage_kv' in df.columns:
        agg['voltage_kv'] = 'max'

    out = (
        df.groupby(
            ['distributor_network', 'substation_name', '_lat_key', '_lon_key'],
            dropna=False,
            as_index=False,
        )
        .agg(agg)
        .drop(columns=['_lat_key', '_lon_key'])
    )

    out['grid_status'] = out['available_capacity_mw'].apply(classify_grid_status)
    out['is_friction'] = out['grid_status'].apply(is_friction_point)
    return out
