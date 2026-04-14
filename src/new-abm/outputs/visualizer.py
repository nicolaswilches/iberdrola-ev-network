"""Visualization functions.

All plots use matplotlib with the Agg backend so they work in
headless environments (CI, scripts) as well as interactive notebooks.
Call ``plt.show()`` after any function if running interactively.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from models.network import RoadNetwork
from models.results import SimulationResults
from models.station import ChargingStation

logger = logging.getLogger(__name__)

# Consistent color palette
COLORS = {
    "baseline": "#2196F3",
    "price_reduction": "#4CAF50",
    "capacity_increase": "#FF9800",
    "high_home_charging": "#9C27B0",
    "default": "#607D8B",
}

SCENARIO_ORDER = ["baseline", "price_reduction", "capacity_increase", "high_home_charging"]


def _scenario_color(name: str) -> str:
    return COLORS.get(name, COLORS["default"])


# ---------------------------------------------------------------------------
# Network map
# ---------------------------------------------------------------------------

def plot_network(
    network: RoadNetwork,
    stations: List[ChargingStation],
    title: str = "Spain Demo Road Network",
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Draw the road network with nodes colour-coded by type and
    charging stations marked with a star.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f0f4f8")

    # Draw edges
    for edge_id, edge in network.edges.items():
        from_n = network.nodes.get(edge.from_node)
        to_n   = network.nodes.get(edge.to_node)
        if not from_n or not to_n:
            continue
        color = {"AP": "#e53935", "A": "#1e88e5", "N": "#43a047"}.get(
            edge.road_type, "#90a4ae"
        )
        lw = 2.0 if edge.road_type == "AP" else 1.2
        ax.plot(
            [from_n.longitude, to_n.longitude],
            [from_n.latitude, to_n.latitude],
            color=color, linewidth=lw, alpha=0.6, zorder=1,
        )

    # Draw nodes
    type_colors = {
        "city": "#1565c0",
        "junction": "#f57f17",
        "waypoint": "#78909c",
    }
    for nid, node in network.nodes.items():
        c = type_colors.get(node.node_type, "#78909c")
        size = max(30, min(200, node.population / 20000)) if node.population else 40
        ax.scatter(node.longitude, node.latitude, s=size, c=c, zorder=3,
                   edgecolors="white", linewidths=0.5)
        if node.node_type == "city" and node.population > 200000:
            ax.annotate(
                node.name, (node.longitude, node.latitude),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points",
            )

    # Draw stations
    station_nodes = {s.node_id for s in stations}
    for nid in station_nodes:
        if nid in network.nodes:
            n = network.nodes[nid]
            ax.scatter(n.longitude, n.latitude, s=120, marker="*",
                       c="#ff6f00", zorder=4, edgecolors="white", linewidths=0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#e53935", label="Motorway (AP)"),
        mpatches.Patch(facecolor="#1e88e5", label="National (A)"),
        mpatches.Patch(facecolor="#1565c0", label="City node"),
        plt.scatter([], [], s=100, c="#ff6f00", marker="*", label="Charging station"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Station utilization
# ---------------------------------------------------------------------------

def plot_station_utilization(
    results: SimulationResults,
    top_n: int = 15,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Bar chart of station utilization rates."""
    sta_df = results.station_summary_df()
    if sta_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    df = sta_df.nlargest(top_n, "total_sessions").sort_values(
        "utilization_rate", ascending=True
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        df["name"], df["utilization_rate"] * 100,
        color=_scenario_color(results.scenario_name), alpha=0.85,
    )
    ax.axvline(100, color="red", linestyle="--", lw=1.5, label="100% (saturated)")
    ax.set_xlabel("Utilization rate (%)")
    ax.set_title(
        f"Station Utilization — {results.scenario_name}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8)

    # Annotate session counts
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['total_sessions'])} sessions",
            va="center", ha="left", fontsize=8,
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wait time distribution
# ---------------------------------------------------------------------------

def plot_wait_time_distribution(
    results: SimulationResults,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Histogram of per-session wait times at stations."""
    ev_df = results.charge_events_df()
    if ev_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No charge events", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Wait time distribution
    ax = axes[0]
    waits = ev_df["wait_time_min"]
    ax.hist(waits, bins=30, color=_scenario_color(results.scenario_name), alpha=0.8)
    ax.axvline(waits.mean(), color="red", linestyle="--",
               label=f"Mean={waits.mean():.1f} min")
    ax.set_xlabel("Wait time (minutes)")
    ax.set_ylabel("Number of sessions")
    ax.set_title(f"Wait Time Distribution — {results.scenario_name}")
    ax.legend(fontsize=8)

    # Charge time distribution
    ax2 = axes[1]
    charge_times = ev_df["charge_time_min"]
    ax2.hist(charge_times, bins=30, color="#4CAF50", alpha=0.8)
    ax2.axvline(charge_times.mean(), color="red", linestyle="--",
                label=f"Mean={charge_times.mean():.1f} min")
    ax2.set_xlabel("Charge time (minutes)")
    ax2.set_ylabel("Number of sessions")
    ax2.set_title(f"Charge Time Distribution — {results.scenario_name}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    results_dict: Dict[str, SimulationResults],
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """
    Multi-panel comparison of key metrics across scenarios.
    """
    from outputs.aggregator import compare_scenarios, station_demand_table

    scenarios = [s for s in SCENARIO_ORDER if s in results_dict]
    scenarios += [s for s in results_dict if s not in SCENARIO_ORDER]

    comp = compare_scenarios(results_dict)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    def bar(ax, column: str, label: str, yformat="%") -> None:
        vals = []
        names = []
        for sc in scenarios:
            if sc in comp.index and column in comp.columns:
                raw = comp.loc[sc, column]
                vals.append(float(raw) if raw is not None else 0.0)
                names.append(sc)
        colors = [_scenario_color(n) for n in names]
        ax.bar(names, vals, color=colors, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        if yformat == "%":
            ax.set_ylabel("%")
        else:
            ax.set_ylabel(yformat)

    bar(axes[0], "completion_rate_pct", "Trip Completion Rate (%)")
    bar(axes[1], "avg_wait_min", "Avg Wait Time (min)", yformat="min")
    bar(axes[2], "avg_charge_stops", "Avg Charge Stops per Trip", yformat="stops")
    bar(axes[3], "total_energy_kwh", "Total Energy Dispensed (kWh)", yformat="kWh")
    bar(axes[4], "max_peak_queue", "Max Peak Queue (connectors)", yformat="agents")
    bar(axes[5], "avg_utilization", "Avg Station Utilization", yformat="fraction")

    fig.suptitle("Scenario Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_hourly_demand(
    results_dict: Dict[str, SimulationResults],
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Line chart of charging sessions per hour for each scenario."""
    from outputs.aggregator import hourly_demand_table

    fig, ax = plt.subplots(figsize=figsize)
    scenarios = [s for s in SCENARIO_ORDER if s in results_dict]
    scenarios += [s for s in results_dict if s not in SCENARIO_ORDER]

    for sc in scenarios:
        hourly = hourly_demand_table(results_dict[sc])
        if hourly.empty:
            continue
        ax.plot(
            hourly["hour"], hourly["num_sessions"],
            marker="o", markersize=4, label=sc,
            color=_scenario_color(sc), linewidth=2,
        )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Charging sessions")
    ax.set_title("Charging Demand by Hour", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_all_plots(
    network: RoadNetwork,
    stations: List[ChargingStation],
    results_dict: Dict[str, SimulationResults],
    output_dir: str = "outputs/plots",
) -> List[str]:
    """Generate and save all standard plots.  Returns list of saved paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # Network map (once)
    fig = plot_network(network, stations)
    path = str(out / "network_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    # Per-scenario plots
    for sc_name, results in results_dict.items():
        safe = sc_name.replace(" ", "_")

        fig = plot_station_utilization(results)
        p = str(out / f"{safe}_station_utilization.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        fig = plot_wait_time_distribution(results)
        p = str(out / f"{safe}_wait_times.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # Cross-scenario comparison (if more than one scenario)
    if len(results_dict) > 1:
        fig = plot_scenario_comparison(results_dict)
        p = str(out / "scenario_comparison.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        fig = plot_hourly_demand(results_dict)
        p = str(out / "hourly_demand.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    logger.info("Saved %d plots to %s/", len(saved), output_dir)
    return saved
