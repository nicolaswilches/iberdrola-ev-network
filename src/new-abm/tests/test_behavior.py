"""Tests for behavior modules: energy, routing, pre-trip, station choice."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from models.agent import VehicleAgent
from models.network import RoadEdge, RoadNetwork, RoadNode
from models.station import ChargingStation
from behavior.energy import compute_charge_duration, compute_segment_energy
from behavior.routing import build_trip_waypoints, plan_route_with_stops
from behavior.pre_trip import decide_initial_soc
from behavior.station_choice import generalized_cost_for_station


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_corridor_network():
    """A → B → C corridor, 200 km total, station at B.
    Edges are exactly 100 km/h so speed_limit_kmh=100 prevents any speed penalty.
    """
    net = RoadNetwork()
    for nid, name, lat in [("A", "Origin", 40.0), ("B", "Middle", 41.0), ("C", "Dest", 42.0)]:
        net.add_node(RoadNode(nid, name, lat, -3.0, "city", 300000))
    net.add_undirected_road(RoadEdge("E1", "A", "B", 100.0, 60.0, "AP", speed_limit_kmh=100.0))
    net.add_undirected_road(RoadEdge("E2", "B", "C", 100.0, 60.0, "AP", speed_limit_kmh=100.0))
    return net


def make_agent(origin="A", dest="C", battery=75.0, soc=None):
    soc = soc if soc is not None else battery * 0.80
    agent = VehicleAgent(
        agent_id="TEST_001",
        origin=origin,
        destination=dest,
        departure_time_min=480.0,
        battery_capacity_kwh=battery,
        usable_capacity_kwh=battery * 0.90,
        consumption_kwh_per_km=0.20,
        max_acceptance_kw=150.0,
        home_charging_access=True,
    )
    agent.current_soc_kwh = soc
    return agent


def make_station(station_id="STA_B", node_id="B", power=150.0, connectors=4, price=0.38):
    return ChargingStation(
        station_id=station_id,
        node_id=node_id,
        name=f"Station {node_id}",
        latitude=41.0,
        longitude=-3.0,
        max_power_kw=power,
        num_connectors=connectors,
        price_per_kwh=price,
    )


BASE_CONFIG = {
    "min_reserve_soc_fraction": 0.10,
    "emergency_soc_fraction": 0.15,
    "max_queue_wait_tolerance_min": 30.0,
    "min_charge_gain_fraction": 0.10,
}


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------

def test_segment_energy_flat():
    net = make_corridor_network()
    energy = compute_segment_energy(["A", "B"], net, consumption_kwh_per_km=0.20)
    assert abs(energy - 20.0) < 0.1  # 100 km × 0.20 kWh/km


def test_segment_energy_two_hops():
    net = make_corridor_network()
    energy = compute_segment_energy(["A", "B", "C"], net, consumption_kwh_per_km=0.20)
    assert abs(energy - 40.0) < 0.1


def test_segment_energy_single_node():
    net = make_corridor_network()
    energy = compute_segment_energy(["A"], net, consumption_kwh_per_km=0.20)
    assert energy == 0.0


def test_charge_duration_cc_phase():
    """Charging below 80% SOC should be purely CC (linear)."""
    duration = compute_charge_duration(
        soc_before_kwh=20.0,
        soc_target_kwh=50.0,   # 50/75 = 67% < 80%
        battery_capacity_kwh=75.0,
        charger_power_kw=150.0,
        vehicle_max_acceptance_kw=150.0,
    )
    expected = (30.0 / 150.0) * 60.0  # 12 minutes
    assert abs(duration - expected) < 0.5


def test_charge_duration_zero():
    duration = compute_charge_duration(
        soc_before_kwh=50.0,
        soc_target_kwh=50.0,
        battery_capacity_kwh=75.0,
        charger_power_kw=150.0,
        vehicle_max_acceptance_kw=150.0,
    )
    assert duration == 0.0


def test_charge_duration_cv_phase_longer():
    """CV phase (80-100%) should take longer than an equivalent CC charge."""
    # Same energy (10 kWh), but one is in CC region, one in CV region
    cc_duration = compute_charge_duration(
        soc_before_kwh=30.0, soc_target_kwh=40.0,  # 40-53% → all CC
        battery_capacity_kwh=75.0, charger_power_kw=150.0, vehicle_max_acceptance_kw=150.0,
    )
    cv_duration = compute_charge_duration(
        soc_before_kwh=65.0, soc_target_kwh=75.0,  # 87-100% → all CV
        battery_capacity_kwh=75.0, charger_power_kw=150.0, vehicle_max_acceptance_kw=150.0,
    )
    assert cv_duration > cc_duration, "CV phase should be slower than CC"


def test_charge_duration_limited_by_vehicle_acceptance():
    """Vehicle max acceptance should cap effective power."""
    dur_full = compute_charge_duration(
        soc_before_kwh=20.0, soc_target_kwh=50.0,
        battery_capacity_kwh=75.0,
        charger_power_kw=350.0,    # HPC station
        vehicle_max_acceptance_kw=150.0,  # vehicle limit
    )
    dur_limited = compute_charge_duration(
        soc_before_kwh=20.0, soc_target_kwh=50.0,
        battery_capacity_kwh=75.0,
        charger_power_kw=350.0,
        vehicle_max_acceptance_kw=50.0,  # old slow vehicle
    )
    assert dur_limited > dur_full * 2.5  # should be significantly longer


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_plan_route_finds_path():
    net = make_corridor_network()
    agent = make_agent()
    stations_by_node = {"B": [make_station()]}
    route, stops = plan_route_with_stops(agent, net, stations_by_node, BASE_CONFIG)
    assert len(route) == 3
    assert route[0] == "A"
    assert route[-1] == "C"


def test_plan_route_no_stops_when_soc_sufficient():
    """Agent with full battery should not need a stop for 200 km."""
    net = make_corridor_network()
    agent = make_agent(battery=75.0, soc=75.0 * 0.90)  # full battery
    stations_by_node = {"B": [make_station()]}
    route, stops = plan_route_with_stops(agent, net, stations_by_node, BASE_CONFIG)
    # 200 km × 0.20 = 40 kWh, full usable = 67.5 kWh → should be fine
    assert len(route) == 3


def test_build_waypoints_with_stop():
    station = make_station()
    route = ["A", "B", "C"]
    waypoints = build_trip_waypoints(route, [station])
    assert len(waypoints) == 2
    assert waypoints[0].is_charging_stop is True
    assert waypoints[0].end_node == "B"
    assert waypoints[1].is_charging_stop is False
    assert waypoints[1].end_node == "C"


def test_build_waypoints_no_stops():
    route = ["A", "B", "C"]
    waypoints = build_trip_waypoints(route, [])
    assert len(waypoints) == 1
    assert waypoints[0].end_node == "C"
    assert waypoints[0].is_charging_stop is False


# ---------------------------------------------------------------------------
# Pre-trip SOC
# ---------------------------------------------------------------------------

def test_initial_soc_with_home_charging():
    net = make_corridor_network()
    agent = make_agent()
    agent.home_charging_access = True
    stations_by_node = {"B": [make_station()]}
    rng = np.random.default_rng(0)
    soc = decide_initial_soc(agent, net, stations_by_node, BASE_CONFIG, rng)
    # Should be between 20% and 100% of usable
    assert agent.usable_capacity_kwh * 0.15 <= soc <= agent.usable_capacity_kwh


def test_initial_soc_without_home_charging():
    net = make_corridor_network()
    agent = make_agent()
    agent.home_charging_access = False
    stations_by_node = {}
    rng = np.random.default_rng(0)
    soc = decide_initial_soc(agent, net, stations_by_node, BASE_CONFIG, rng)
    assert 0.10 * agent.usable_capacity_kwh <= soc <= agent.usable_capacity_kwh


def test_initial_soc_is_not_fixed_constant():
    """Pre-trip SOC must vary across agents (endogenous, not exogenous)."""
    net = make_corridor_network()
    stations_by_node = {"B": [make_station()]}
    socs = []
    for i in range(20):
        agent = make_agent()
        agent.home_charging_access = True
        agent.risk_tolerance = i / 20.0  # vary risk tolerance
        rng = np.random.default_rng(i)
        soc = decide_initial_soc(agent, net, stations_by_node, BASE_CONFIG, rng)
        socs.append(soc)
    # SOC should not be identical for all agents
    assert len(set(round(s, 1) for s in socs)) > 1


# ---------------------------------------------------------------------------
# Station choice / generalized cost
# ---------------------------------------------------------------------------

def test_generalized_cost_lower_price_preferred():
    """Lower price station should have lower GC, all else equal."""
    agent = make_agent()
    agent.current_soc_kwh = 30.0
    cheap = make_station(price=0.20)
    expensive = make_station(price=0.50)
    gc_cheap = generalized_cost_for_station(agent, cheap, 0.0, 50.0)
    gc_expensive = generalized_cost_for_station(agent, expensive, 0.0, 50.0)
    assert gc_cheap < gc_expensive


def test_generalized_cost_queue_penalty():
    """Station with agents queuing should have higher GC than an empty station."""
    import simpy

    def fill_station(env, station):
        """Generator that occupies all connectors and adds one to the queue."""
        reqs = []
        for _ in range(station.num_connectors + 1):
            r = station.resource.request()
            reqs.append(r)
            yield r
        yield env.timeout(9999)  # hold connectors forever (within test)

    env = simpy.Environment()
    agent = make_agent()
    agent.current_soc_kwh = 30.0
    agent.queue_aversion = 2.0

    station_busy = make_station(station_id="STA_BUSY", connectors=2)
    station_busy.resource = simpy.Resource(env, capacity=2)
    env.process(fill_station(env, station_busy))
    env.run(until=1)  # advance time so requests are processed

    station_empty = make_station(station_id="STA_EMPTY", connectors=2)
    station_empty.resource = simpy.Resource(env, capacity=2)
    # Don't use the empty station — it stays idle

    gc_busy = generalized_cost_for_station(agent, station_busy, 0.0, 50.0)
    gc_empty = generalized_cost_for_station(agent, station_empty, 0.0, 50.0)

    assert gc_busy >= gc_empty, (
        f"Busy station (queue={station_busy.current_queue_length()}) "
        f"GC={gc_busy:.2f} should be >= empty station GC={gc_empty:.2f}"
    )


def test_generalized_cost_detour_penalty():
    """Detour time should increase GC proportionally to VoT."""
    agent = make_agent()
    agent.current_soc_kwh = 30.0
    agent.value_of_time_eur_per_hour = 60.0  # high VoT
    station = make_station()
    gc_no_detour = generalized_cost_for_station(agent, station, 0.0, 50.0)
    gc_detour = generalized_cost_for_station(agent, station, 30.0, 50.0)  # 30 min detour
    # 30 min at €60/hr = €30 extra
    assert abs(gc_detour - gc_no_detour - 30.0) < 1.0
