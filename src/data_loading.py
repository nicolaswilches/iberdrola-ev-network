"""
Reusable functions for loading each dataset.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / 'data' / 'raw'
DATA_PROCESSED = Path(__file__).parent.parent / 'data' / 'processed'


def load_road_network(filepath=None):
    """Load Ministry of Transport road network (GeoJSON/shapefile)."""
    if filepath is None:
        filepath = DATA_RAW / 'ministry_roads'
    # TODO: implement
    raise NotImplementedError


def load_nap_charging_points(filepath=None):
    """Load NAP existing EV charging station locations."""
    if filepath is None:
        filepath = DATA_RAW / 'nap_charging_points'
    # TODO: implement
    raise NotImplementedError


def load_grid_capacity(distributor='all'):
    """Load grid capacity data for i-DE, Endesa, or Viesgo."""
    # TODO: implement
    raise NotImplementedError


def load_dgt_registrations(filepath=None):
    """Load DGT monthly vehicle registration data."""
    if filepath is None:
        filepath = DATA_RAW / 'dgt_registrations'
    # TODO: implement
    raise NotImplementedError


def load_ev_forecast(filepath=None):
    """Load EV fleet projection output from datos.gob.es fork."""
    if filepath is None:
        filepath = DATA_RAW / 'datos_gob_ev_forecast'
    # TODO: implement
    raise NotImplementedError


def load_imd_traffic(filepath=None):
    """Load DGT Intensidad Media Diaria traffic count data."""
    if filepath is None:
        filepath = DATA_RAW / 'additional' / 'dgt_imd_traffic'
    # TODO: implement
    raise NotImplementedError
