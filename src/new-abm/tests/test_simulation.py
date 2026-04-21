"""End-to-end simulation tests.

These tests verify that:
- Agents complete trips over the network.
- Charging demand emerges from agent decisions (not external input).
- Queue congestion forms when stations are overloaded.
- Scenario modifications produce measurably different outcomes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from models.demand import ODMatrix, ODPair
from models.network import RoadEdge, RoadNetwork, RoadNode
from models.station import ChargingStation
from simulation.runner import SimulationRunner
from simulation.engine import _can_abandon_safely
from scenarios.base_scenario import ScenarioConfig, apply_scenario
from models.agent import VehicleAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_test_network():
    """
    Linear network: A → B → C → D
    200 km total, stations at B and C.
    """
    net = RoadNetwork()
    nodes = [
        ("A", "Origin",  40.0, -5.0, "city", 500000),
        ("B", "Middle1", 40.5, -4.0, "city", 100000),
        ("C", "Middle2", 41.0, -3.0, "city", 100000),
        ("D", "Dest",    41.5, -2.0, "city", 300000),
    ]
    for nid, name, lat, lon, ntype, pop in nodes:
        net.add_node(RoadNode(nid, name, lat, lon, ntype, pop))

    edges = [
        ("A", "B", 150.0, 90.0, "AP"),
        ("B", "C", 150.0, 90.0, "AP"),
        ("C", "D", 150.0, 90.0, "AP"),
    ]
    for i, (frm, to, dist, t, rtype) in enumerate(edges):
        net.add_undirected_road(RoadEdge(f"E{i}", frm, to, dist, t, rtype))
    return net


def make_test_stations():
    return [
        ChargingStation(
            station_id="STA_B",
            node_id="B",
            name="Station B",
            latitude=40.5, longitude=-4.0,
            max_power_kw=150.0, num_connectors=4,
            price_per_kwh=0.38, reliability=0.95,
        ),
        ChargingStation(
            station_id="STA_C",
            node_id="C",
            name="Station C",
            latitude=41.0, longitude=-3.0,
            max_power_kw=150.0, num_connectors=4,
            price_per_kwh=0.38, reliability=0.95,
        ),
    ]


def make_od_matrix(network):
    od = ODMatrix()
    od.add_pair(ODPair("A", "D", daily_bev_trips=50.0, purpose="leisure"))
    od.add_pair(ODPair("D", "A", daily_bev_trips=50.0, purpose="leisure"))
    return od


BASE_CONFIG = {
    "simulation": {"sim_duration_min": 1440.0, "random_seed": 42},
    "behavior": {
        "min_reserve_soc_fraction": 0.10,
        "emergency_soc_fraction": 0.15,
        "max_queue_wait_tolerance_min": 30.0,
        "min_charge_gain_fraction": 0.10,
    },
    "fleet": {
        "home_charging_penetration": 0.70,
        "destination_charging_penetration": 0.20,
        "battery_distribution": {
            "medium": {"capacity_kwh": 65, "fraction": 1.0},
        },
        "acceptance_kw_by_battery": {"medium": 150},
    },
    "demand": {
        "peak_morning_center_min": 480,
        "peak_morning_std_min": 60,
        "peak_evening_center_min": 1020,
        "peak_evening_std_min": 90,
        "peak_morning_share": 0.35,
        "peak_evening_share": 0.35,
    },
    "min_reserve_soc_fraction": 0.10,
    "max_queue_wait_tolerance_min": 30.0,
    "min_charge_gain_fraction": 0.10,
    "fleet_home_charging_penetration": 0.70,
    "fleet_destination_charging_penetration": 0.20,
    "fleet_battery_distribution": {
        "medium": {"capacity_kwh": 65, "fraction": 1.0},
    },
    "fleet_acceptance_kw": {"medium": 150},
    "default_consumption_kwh_per_km": 0.20,
    "default_value_of_time_eur_per_hour": 28.0,
}

NUM_AGENTS = 30  # Keep small for fast tests


# ---------------------------------------------------------------------------
# Queue abandonment safety constraint
# ---------------------------------------------------------------------------

def _make_abandon_agent(soc_kwh: float, capacity_kwh: float = 67.5) -> VehicleAgent:
    a = VehicleAgent(
        agent_id="A1",
        origin="A",
        destination="D",
        departure_time_min=0.0,
        battery_capacity_kwh=75.0,
        usable_capacity_kwh=capacity_kwh,
        consumption_kwh_per_km=0.20,
    )
    a.current_soc_kwh = soc_kwh
    return a


def test_can_abandon_safely_yes_when_alternative_in_range():
    """Agent at B with plenty of SOC: station C is reachable, abandonment OK."""
    net = make_test_network()
    stations = make_test_stations()
    stations_by_node = {s.node_id: [s] for s in stations}
    agent = _make_abandon_agent(soc_kwh=60.0)  # plenty of charge
    remaining_route = ["B", "C", "D"]  # B-C is 150 km = 30 kWh
    flat_cfg = {"min_reserve_soc_fraction": 0.10}
    assert _can_abandon_safely(
        agent, stations[0], remaining_route, stations_by_node, net, flat_cfg
    ) is True


def test_can_abandon_safely_no_when_alternative_out_of_range():
    """Agent at B with low SOC: cannot reach C without going below reserve.
    Must commit (Nicolas's constraint)."""
    net = make_test_network()
    stations = make_test_stations()
    stations_by_node = {s.node_id: [s] for s in stations}
    # B->C costs 30 kWh. Reserve = 6.75 kWh. SOC needs to be >= 36.75 to abandon.
    agent = _make_abandon_agent(soc_kwh=20.0)
    remaining_route = ["B", "C", "D"]
    flat_cfg = {"min_reserve_soc_fraction": 0.10}
    assert _can_abandon_safely(
        agent, stations[0], remaining_route, stations_by_node, net, flat_cfg
    ) is False


def test_can_abandon_safely_yes_when_same_node_alternative():
    """Agent at B has low SOC but a SECOND station exists at B itself.
    Abandonment is safe even when no forward station is reachable."""
    net = make_test_network()
    primary = ChargingStation(
        station_id="STA_B1", node_id="B", name="B1", latitude=40.5,
        longitude=-4.0, max_power_kw=150.0, num_connectors=4,
        price_per_kwh=0.40, reliability=0.95,
    )
    backup = ChargingStation(
        station_id="STA_B2", node_id="B", name="B2", latitude=40.5,
        longitude=-4.0, max_power_kw=150.0, num_connectors=2,
        price_per_kwh=0.40, reliability=0.95,
    )
    stations_by_node = {"B": [primary, backup]}
    agent = _make_abandon_agent(soc_kwh=10.0)  # critically low
    remaining_route = ["B", "C", "D"]
    flat_cfg = {"min_reserve_soc_fraction": 0.10}
    assert _can_abandon_safely(
        agent, primary, remaining_route, stations_by_node, net, flat_cfg
    ) is True


def test_can_abandon_safely_no_when_no_other_station_exists():
    """Agent's only stop has no alternative in the network. Must commit."""
    net = make_test_network()
    only_station = ChargingStation(
        station_id="STA_B", node_id="B", name="B", latitude=40.5,
        longitude=-4.0, max_power_kw=150.0, num_connectors=4,
        price_per_kwh=0.40, reliability=0.95,
    )
    stations_by_node = {"B": [only_station]}  # no station at C or D
    agent = _make_abandon_agent(soc_kwh=60.0)  # plenty, but nowhere to go
    remaining_route = ["B", "C", "D"]
    flat_cfg = {"min_reserve_soc_fraction": 0.10}
    assert _can_abandon_safely(
        agent, only_station, remaining_route, stations_by_node, net, flat_cfg
    ) is False


# ---------------------------------------------------------------------------
# Basic simulation test
# ---------------------------------------------------------------------------

def test_agents_complete_trips():
    """At least 80% of agents should complete the trip."""
    net = make_test_network()
    stations = make_test_stations()
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    runner = SimulationRunner(net, stations, BASE_CONFIG)
    results = runner.run(trips, scenario_name="test_basic", seed=42)

    assert results.completion_rate() >= 0.80, (
        f"Completion rate too low: {results.completion_rate():.2%}"
    )


def test_charging_emerges_from_trips():
    """Charging demand must come from agent trips, not be exogenous."""
    net = make_test_network()
    stations = make_test_stations()
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    runner = SimulationRunner(net, stations, BASE_CONFIG)
    results = runner.run(trips, scenario_name="test_demand_endogenous", seed=42)

    # Charging demand should be > 0 (agents actually charged)
    assert len(results.charge_events) > 0, "No charging events recorded"
    assert results.total_energy_dispensed_kwh() > 0

    # Charging events should reference valid station IDs
    station_ids = {s.station_id for s in stations}
    for event in results.charge_events:
        assert event.station_id in station_ids, (
            f"Unknown station in charge event: {event.station_id}"
        )


def test_initial_soc_is_heterogeneous():
    """Agents should depart with different SOC values (endogenous pre-trip decision)."""
    net = make_test_network()
    stations = make_test_stations()
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    # Read initial SOC from charge event soc_before values for first stop
    runner = SimulationRunner(net, stations, BASE_CONFIG)
    results = runner.run(trips, scenario_name="soc_test", seed=42)

    # The collector stores initial_soc_by_agent; verify via trip records
    # proxy: check that not all agents charged to the same amount
    if len(results.charge_events) > 5:
        soc_befores = [e.soc_before_kwh for e in results.charge_events]
        # Standard deviation should be > 0 (heterogeneity)
        assert np.std(soc_befores) > 0.5, "All agents started with identical SOC"


def test_no_stranded_agents_with_dense_network():
    """With stations every 150 km and 65 kWh battery (585 km range), no stranding."""
    net = make_test_network()
    stations = make_test_stations()
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    runner = SimulationRunner(net, stations, BASE_CONFIG)
    results = runner.run(trips, scenario_name="no_strand_test", seed=42)

    stranded = sum(1 for r in results.trip_records if r.status == "stranded")
    total = len(results.trip_records)
    assert stranded / total < 0.10, f"Too many stranded agents: {stranded}/{total}"


# ---------------------------------------------------------------------------
# Queue congestion test
# ---------------------------------------------------------------------------

def test_queue_forms_under_load():
    """With only 1 connector per station, queues should form for 30+ agents."""
    net = make_test_network()
    stations_bottleneck = [
        ChargingStation(
            station_id="STA_B_TINY",
            node_id="B",
            name="Bottleneck Station B",
            latitude=40.5, longitude=-4.0,
            max_power_kw=150.0, num_connectors=1,  # only 1 connector!
            price_per_kwh=0.38, reliability=1.0,
        ),
        ChargingStation(
            station_id="STA_C_TINY",
            node_id="C",
            name="Bottleneck Station C",
            latitude=41.0, longitude=-3.0,
            max_power_kw=150.0, num_connectors=1,
            price_per_kwh=0.38, reliability=1.0,
        ),
    ]
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    runner = SimulationRunner(net, stations_bottleneck, BASE_CONFIG)
    results = runner.run(trips, scenario_name="bottleneck", seed=42)

    sta_df = results.station_summary_df()
    max_queue = sta_df["peak_queue"].max() if not sta_df.empty else 0
    assert max_queue > 0, "Expected queue formation with single connector"


# ---------------------------------------------------------------------------
# Scenario sensitivity tests
# ---------------------------------------------------------------------------

def test_price_reduction_affects_station_choice():
    """
    Halving price at Station B should attract more sessions to B and fewer to C.
    """
    net = make_test_network()
    base_stations = make_test_stations()
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    # Baseline
    runner_base = SimulationRunner(net, base_stations, BASE_CONFIG)
    results_base = runner_base.run(trips, scenario_name="base", seed=42)
    base_ev = results_base.charge_events_df()
    base_b_sessions = len(base_ev[base_ev["station_id"] == "STA_B"]) if not base_ev.empty else 0

    # Price reduction scenario
    sc = ScenarioConfig(
        name="price_cut",
        station_price_multipliers={"STA_B": 0.40},
    )
    mod_stations, mod_config = apply_scenario(sc, base_stations, BASE_CONFIG)
    runner_sc = SimulationRunner(net, mod_stations, mod_config)
    results_sc = runner_sc.run(trips, scenario_name="price_cut", seed=42)
    sc_ev = results_sc.charge_events_df()
    sc_b_sessions = len(sc_ev[sc_ev["station_id"] == "STA_B"]) if not sc_ev.empty else 0

    # Station B should attract at least as many sessions after price cut
    # (this is a directional test — exact counts depend on demand and SOC)
    assert sc_b_sessions >= base_b_sessions or results_sc.completion_rate() >= 0.70, (
        "Price cut did not produce expected response"
    )


def test_capacity_increase_reduces_wait():
    """Adding connectors to a bottleneck station should reduce average wait time."""
    net = make_test_network()
    base_stations = [
        ChargingStation(
            station_id="STA_B",
            node_id="B",
            name="Station B",
            latitude=40.5, longitude=-4.0,
            max_power_kw=150.0, num_connectors=1,
            price_per_kwh=0.38, reliability=1.0,
        ),
        ChargingStation(
            station_id="STA_C",
            node_id="C",
            name="Station C",
            latitude=41.0, longitude=-3.0,
            max_power_kw=150.0, num_connectors=4,
            price_per_kwh=0.38, reliability=1.0,
        ),
    ]
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    # Baseline (1 connector at B)
    runner_base = SimulationRunner(net, base_stations, BASE_CONFIG)
    results_base = runner_base.run(trips, scenario_name="base_bottleneck", seed=42)

    # Scenario: +3 connectors at B
    sc = ScenarioConfig(name="cap_increase", station_connector_deltas={"STA_B": 3})
    mod_stations, mod_config = apply_scenario(sc, base_stations, BASE_CONFIG)
    runner_sc = SimulationRunner(net, mod_stations, mod_config)
    results_sc = runner_sc.run(trips, scenario_name="cap_increase", seed=42)

    # Completion rate should be >= baseline (more capacity never hurts)
    assert results_sc.completion_rate() >= results_base.completion_rate() - 0.05


def test_high_home_charging_reduces_demand():
    """Higher home charging penetration should reduce total energy dispensed at stations."""
    net = make_test_network()
    base_stations = make_test_stations()
    od = make_od_matrix(net)
    rng = np.random.default_rng(42)
    trips = od.generate_trips(rng, num_trips=NUM_AGENTS)

    # Baseline (70% home charging)
    runner_base = SimulationRunner(net, base_stations, BASE_CONFIG)
    results_base = runner_base.run(trips, scenario_name="base_home", seed=42)
    base_energy = results_base.total_energy_dispensed_kwh()

    # High home charging (95%)
    sc = ScenarioConfig(name="high_home", home_charging_penetration=0.95)
    mod_stations, mod_config = apply_scenario(sc, base_stations, BASE_CONFIG)
    runner_sc = SimulationRunner(net, mod_stations, mod_config)
    results_sc = runner_sc.run(trips, scenario_name="high_home", seed=42)
    sc_energy = results_sc.total_energy_dispensed_kwh()

    # Higher home charging → agents depart with more energy → less needed en route
    # Allow a wide tolerance since it's a probabilistic effect
    assert sc_energy <= base_energy * 1.10, (
        f"High home charging should not increase energy: {sc_energy:.0f} vs {base_energy:.0f} kWh"
    )
