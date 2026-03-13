"""
Core optimization algorithm for EV charging station placement.
Objective: minimize total stations while ensuring no road gap > MAX_STATION_SPACING_KM.
"""

import numpy as np
import pandas as pd
from src.constants import MAX_STATION_SPACING_KM, AFIR_SPACING_KM


def compute_coverage_gaps(road_segments_gdf, existing_stations_gdf, spacing_km=MAX_STATION_SPACING_KM):
    """
    Identify road segments with insufficient charging coverage.
    Returns GeoDataFrame of road sub-segments that need new stations.
    """
    # TODO: implement greedy interval covering algorithm
    raise NotImplementedError


def place_stations_greedy(gap_segments, demand_df):
    """
    Greedy station placement: place stations at midpoints of uncovered gaps,
    weighted by demand.
    Returns DataFrame with proposed station coordinates and n_chargers.
    """
    # TODO: implement
    raise NotImplementedError


def calculate_chargers_needed(daily_ev_traffic, charging_probability=0.07,
                               avg_charge_hours=0.4, operating_hours=18):
    """Calculate number of 150 kW chargers needed for a given traffic volume."""
    daily_demand_hours = daily_ev_traffic * charging_probability * avg_charge_hours
    n_chargers = int(np.ceil(daily_demand_hours / operating_hours))
    return max(2, min(n_chargers, 8))  # Clamp between 2 and 8
