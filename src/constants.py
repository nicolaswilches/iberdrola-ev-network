"""
Fixed parameters and constants for the Iberdrola Datathon 2026.
All values are documented and justified in references/assumptions.md
"""

# === MANDATED BY DATATHON RULES (DO NOT CHANGE) ===
POWER_PER_CHARGER_KW = 150  # Fixed at 150 kW per charger for all teams

# === EV AUTONOMY AND SPACING ===
AVG_EV_RANGE_KM = 300       # WLTP average range
USABLE_RANGE_FACTOR = 0.80  # Drivers won't go below 20% battery
EFFECTIVE_RANGE_KM = AVG_EV_RANGE_KM * USABLE_RANGE_FACTOR  # 240 km
MAX_STATION_SPACING_KM = 150  # Conservative max gap between stations
AFIR_SPACING_KM = 60        # EU AFIR requirement on TEN-T core corridors

# === GRID STATUS THRESHOLDS (MW available at nearest substation) ===
GRID_SUFFICIENT_MIN_MW = 5.0    # >= 5 MW → Sufficient
GRID_MODERATE_MIN_MW = 1.0      # >= 1 MW and < 5 MW → Moderate
# < 1 MW → Congested

GRID_STATUS_LABELS = {
    'sufficient': 'Sufficient',
    'moderate': 'Moderate',
    'congested': 'Congested',
}

# === SUBSTATION MATCHING ===
MAX_SUBSTATION_SEARCH_RADIUS_KM = 25  # Max distance to match a station to a substation
DEFAULT_STATUS_IF_NO_SUBSTATION = 'Congested'  # If no substation found within radius

# === DEMAND MODELING ===
CHARGING_PROBABILITY = 0.07     # 7% of passing EVs will charge at a given station
AVG_CHARGE_DURATION_HOURS = 0.4 # ~24 minutes average DC fast charge session
EFFECTIVE_OPERATING_HOURS = 18  # Hours per day the station is effectively available

# === SEASONAL MULTIPLIERS (for peak summer on coastal corridors) ===
SEASONAL_MULTIPLIERS = {
    'default': 1.0,
    'mediterranean_summer': 2.0,  # AP-7, A-7 Jul-Aug
    'atlantic_summer': 1.5,       # A-8, AP-9 Jul-Aug
}

# === OUTPUT FILE NAMES ===
OUTPUT_FILE_1 = 'File_1.csv'  # Global Network KPIs
OUTPUT_FILE_2 = 'File_2.csv'  # Proposed Charging Locations
OUTPUT_FILE_3 = 'File_3.csv'  # Friction Points

# === OUTPUT SCHEMAS ===
FILE_1_COLUMNS = [
    'total_proposed_stations',
    'total_existing_stations_baseline',
    'total_friction_points',
    'total_ev_projected_2027',
]

FILE_2_COLUMNS = [
    'location_id',
    'latitude',
    'longitude',
    'route_segment',
    'n_chargers_proposed',
    'grid_status',
]

FILE_3_COLUMNS = [
    'bottleneck_id',
    'latitude',
    'longitude',
    'route_segment',
    'distributor_network',
    'estimated_demand_kw',
    'grid_status',
]

# === ACCEPTED VALUES ===
VALID_GRID_STATUSES_FILE2 = ['Sufficient', 'Moderate', 'Congested']
VALID_GRID_STATUSES_FILE3 = ['Moderate', 'Congested']  # Sufficient NOT allowed in File 3
VALID_DISTRIBUTORS = ['i-DE', 'Endesa', 'Viesgo']
