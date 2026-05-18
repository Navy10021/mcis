try:
    from mcis.viz.maps import plot_traffic_density, plot_vessel_tracks, plot_grid_anomaly
except ImportError:
    def _missing_dep(name: str):
        def _raise(*args, **kwargs):
            raise ImportError(
                f"Optional dependency required for {name}. "
                "Install geo extras: pip install mcis[geo]"
            )
        return _raise
    plot_traffic_density = _missing_dep("plot_traffic_density")
    plot_vessel_tracks = _missing_dep("plot_vessel_tracks")
    plot_grid_anomaly = _missing_dep("plot_grid_anomaly")

from mcis.viz.timeseries import (
    plot_before_after,
    plot_multi_metric_panel,
    plot_flag_composition,
)
from mcis.viz.heatmaps import plot_correlation_heatmap, plot_feature_time_heatmap

__all__ = [
    "plot_traffic_density",
    "plot_vessel_tracks",
    "plot_grid_anomaly",
    "plot_before_after",
    "plot_multi_metric_panel",
    "plot_flag_composition",
    "plot_correlation_heatmap",
    "plot_feature_time_heatmap",
]
