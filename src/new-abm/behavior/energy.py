"""Energy consumption and charging-time models.

These functions are deliberately small and replaceable.  The default
implementation is a piecewise model that captures:

  - speed-dependent highway consumption
  - slope penalty / regeneration
  - a credible CC-CV charging curve with power taper above 80% SOC

To improve accuracy later, replace the consumption function with a
physics-based model (rolling resistance + air drag + drivetrain losses)
or a lookup table calibrated from real WLTP data.
"""

from __future__ import annotations

from typing import Dict, List

from models.network import RoadNetwork


# ---------------------------------------------------------------------------
# Consumption model
# ---------------------------------------------------------------------------

def compute_segment_energy(
    nodes: List[str],
    network: RoadNetwork,
    consumption_kwh_per_km: float,
    high_speed_penalty: float = 0.0015,
    slope_factor: float = 0.04,
) -> float:
    """
    Estimate kWh consumed driving a sequence of consecutive road nodes.

    Parameters
    ----------
    nodes:
        Ordered list of node IDs forming the path segment.
    network:
        Road network to query edge attributes from.
    consumption_kwh_per_km:
        Agent-specific base consumption at ~100 km/h.
    high_speed_penalty:
        Extra kWh per km per km/h above 100 km/h (aerodynamic drag).
    slope_factor:
        Extra kWh per km per % grade (uphill) / regeneration (downhill).

    Returns
    -------
    Total energy consumption in kWh (always ≥ 0).
    """
    total_kwh = 0.0

    for i in range(len(nodes) - 1):
        from_n, to_n = nodes[i], nodes[i + 1]
        attrs = network.get_edge_attrs(from_n, to_n)
        if attrs is None:
            continue

        dist_km = attrs["distance_km"]
        speed_kmh = attrs.get("speed_limit_kmh", 100.0)
        slope = attrs.get("slope_grade", 0.0)

        # Base consumption
        edge_kwh = dist_km * consumption_kwh_per_km

        # Speed penalty (above 100 km/h aerodynamic drag increases sharply)
        if speed_kmh > 100.0:
            edge_kwh += dist_km * high_speed_penalty * (speed_kmh - 100.0)

        # Slope: uphill costs, downhill recovers (regen efficiency ~60%)
        if slope > 0:
            edge_kwh += dist_km * slope_factor * slope
        else:
            regen_efficiency = 0.60
            edge_kwh += dist_km * slope_factor * slope * regen_efficiency  # negative

        total_kwh += max(0.0, edge_kwh)

    return total_kwh


# ---------------------------------------------------------------------------
# Charging curve
# ---------------------------------------------------------------------------

def compute_charge_duration(
    soc_before_kwh: float,
    soc_target_kwh: float,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    vehicle_max_acceptance_kw: float,
    taper_threshold: float = 0.80,
    taper_min_fraction: float = 0.10,
) -> float:
    """
    Estimate charging duration (minutes) using a piecewise CC-CV curve.

    The charging curve has two phases:
    - **Constant-current (CC)**: full power from soc_before up to
      ``taper_threshold`` × battery_capacity.
    - **Constant-voltage (CV)**: power tapers linearly from full power down
      to ``taper_min_fraction`` × full_power at 100% SOC.

    Parameters
    ----------
    soc_before_kwh:
        SOC at start of session (kWh).
    soc_target_kwh:
        Desired SOC at end of session (kWh).
    battery_capacity_kwh:
        Physical battery capacity (kWh).
    charger_power_kw:
        Station connector max power (kW).
    vehicle_max_acceptance_kw:
        Vehicle DCFC acceptance limit (kW).
    taper_threshold:
        SOC fraction above which power tapers (default 0.80 = 80%).
    taper_min_fraction:
        Power fraction at 100% SOC (default 0.10 = 10% of peak).

    Returns
    -------
    Charging duration in minutes (≥ 0).
    """
    if soc_target_kwh <= soc_before_kwh:
        return 0.0

    peak_power_kw = min(charger_power_kw, vehicle_max_acceptance_kw)
    taper_kwh = taper_threshold * battery_capacity_kwh

    total_min = 0.0

    # --- CC phase (up to taper threshold) ---
    if soc_before_kwh < taper_kwh:
        cc_end_kwh = min(soc_target_kwh, taper_kwh)
        cc_energy = cc_end_kwh - soc_before_kwh
        total_min += (cc_energy / peak_power_kw) * 60.0

    # --- CV phase (above taper threshold) ---
    if soc_target_kwh > taper_kwh:
        cv_start_kwh = max(soc_before_kwh, taper_kwh)
        cv_end_kwh = soc_target_kwh
        cv_energy = cv_end_kwh - cv_start_kwh

        if cv_energy > 0:
            # Average power fraction over the taper range
            start_fraction = (cv_start_kwh / battery_capacity_kwh - taper_threshold) / (
                1.0 - taper_threshold
            )
            end_fraction = (cv_end_kwh / battery_capacity_kwh - taper_threshold) / (
                1.0 - taper_threshold
            )
            # Linear taper: power_fraction = 1 - (1 - taper_min) * soc_fraction
            slope = 1.0 - taper_min_fraction
            avg_power_fraction = 1.0 - slope * (start_fraction + end_fraction) / 2.0
            avg_power_kw = peak_power_kw * max(taper_min_fraction, avg_power_fraction)
            total_min += (cv_energy / avg_power_kw) * 60.0

    return max(0.0, total_min)


def energy_needed_for_path(
    nodes: List[str],
    network: RoadNetwork,
    consumption_kwh_per_km: float,
    **kwargs,
) -> float:
    """Convenience wrapper: energy for a complete path (list of nodes)."""
    return compute_segment_energy(nodes, network, consumption_kwh_per_km, **kwargs)
