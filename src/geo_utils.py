"""
Spatial operations: nearest substation, point-on-road, haversine distance.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def find_nearest_substation(station_lat, station_lon, substations_gdf, max_radius_km=25):
    """
    Find nearest substation to a proposed charging station.
    Returns (substation_id, distance_km, available_capacity_mw, distributor_network)
    or None if no substation within max_radius_km.
    """
    # TODO: implement using spatial index for efficiency
    raise NotImplementedError


def snap_point_to_road(lat, lon, roads_gdf, max_distance_km=2):
    """Snap a point to the nearest road segment."""
    # TODO: implement
    raise NotImplementedError


def get_road_segment_id(lat, lon, roads_gdf):
    """Return the route_segment identifier for the nearest road."""
    # TODO: implement
    raise NotImplementedError
