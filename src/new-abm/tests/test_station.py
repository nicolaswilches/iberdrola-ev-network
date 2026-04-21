"""Tests for ChargingStation model and SimPy resource integration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import simpy
import pytest
from models.station import ChargingStation


def make_station(connectors=4, power=150.0, price=0.38, reliability=0.95):
    return ChargingStation(
        station_id="STA_TEST",
        node_id="NODE_A",
        name="Test Station",
        latitude=40.0,
        longitude=-3.0,
        max_power_kw=power,
        num_connectors=connectors,
        price_per_kwh=price,
        reliability=reliability,
    )


# ---------------------------------------------------------------------------
# Basic attribute tests
# ---------------------------------------------------------------------------

def test_station_attributes():
    s = make_station()
    assert s.station_id == "STA_TEST"
    assert s.max_power_kw == 150.0
    assert s.num_connectors == 4
    assert s.price_per_kwh == 0.38


def test_effective_connectors_reliability():
    """effective_connectors should account for reliability."""
    s = make_station(connectors=10, reliability=0.90)
    eff = s.effective_connectors()
    assert eff <= 10
    assert eff >= 1


def test_expected_wait_no_queue():
    s = make_station()
    env = simpy.Environment()
    s.init_resource(env)
    # No one using it → zero wait
    assert s.expected_wait_time_min() == 0.0


def test_expected_wait_with_queue():
    """With all connectors busy + 2 waiting, expected wait should be > 0."""
    s = make_station(connectors=2)
    env = simpy.Environment()
    s.init_resource(env)

    # Occupy both connectors
    req1 = s.resource.request()
    req2 = s.resource.request()
    env.step()
    env.step()

    # Queue 2 more (they can't get a connector yet)
    req3 = s.resource.request()
    req4 = s.resource.request()

    wait = s.expected_wait_time_min(avg_session_min=20.0)
    assert wait > 0.0


def test_is_full():
    s = make_station(connectors=2)
    env = simpy.Environment()
    s.init_resource(env)

    assert not s.is_full()

    req1 = s.resource.request()
    req2 = s.resource.request()
    env.step()
    env.step()

    assert s.is_full()


def test_record_session_accumulates():
    s = make_station()
    s.record_session(wait_time_min=5.0, charge_time_min=20.0, energy_kwh=15.0)
    s.record_session(wait_time_min=10.0, charge_time_min=25.0, energy_kwh=20.0)
    assert s.total_sessions == 2
    assert abs(s.total_energy_kwh - 35.0) < 0.01
    assert abs(s.total_wait_time_min - 15.0) < 0.01


def test_utilization_rate():
    s = make_station(connectors=2)
    # 2 connectors × 1440 min = 2880 connector-minutes available
    s.total_charge_time_min = 1440.0  # half utilized
    rate = s.utilization_rate(sim_duration_min=1440.0)
    assert abs(rate - 0.50) < 0.01


def test_utilization_rate_clamped():
    s = make_station(connectors=2)
    s.total_charge_time_min = 99999.0
    rate = s.utilization_rate(sim_duration_min=1440.0)
    assert rate <= 1.0


# ---------------------------------------------------------------------------
# SimPy queueing integration test
# ---------------------------------------------------------------------------

def test_simpy_resource_queueing():
    """
    Verify that when more agents request a connector than are available,
    they queue up and are served sequentially.
    """
    s = make_station(connectors=2)
    env = simpy.Environment()
    s.init_resource(env)

    results = []

    def agent(agent_id, charge_time):
        with s.resource.request() as req:
            yield req
            yield env.timeout(charge_time)
            results.append((agent_id, env.now))

    # 4 agents, 2 connectors, each charges 10 min
    for i in range(4):
        env.process(agent(i, 10.0))

    env.run(until=50)

    assert len(results) == 4
    # First two finish at t=10, second two at t=20
    finish_times = sorted(r[1] for r in results)
    assert finish_times[0] == finish_times[1] == 10.0
    assert finish_times[2] == finish_times[3] == 20.0
