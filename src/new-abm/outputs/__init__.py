from .aggregator import aggregate_results, station_demand_table, trip_summary

__all__ = [
    "aggregate_results",
    "station_demand_table",
    "trip_summary",
    "plot_network",
    "plot_station_utilization",
    "plot_wait_time_distribution",
    "plot_scenario_comparison",
    "save_all_plots",
]


def __getattr__(name):
    if name in {
        "plot_network",
        "plot_station_utilization",
        "plot_wait_time_distribution",
        "plot_scenario_comparison",
        "save_all_plots",
    }:
        from . import visualizer

        return getattr(visualizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
