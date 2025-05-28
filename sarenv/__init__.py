# sarenv/__init__.py
"""
SAR Environment (sarenv) package for simulating and evaluating Search and Rescue scenarios.
"""

# Import key classes and functions to make them easily accessible at the package level
from .core.environment import Environment, EnvironmentBuilder
from .core.geometries import (
    GeoData,
    GeoPoint,
    GeoPolygon,
    GeoMultiPolygon,
    GeoTrajectory,
    GeoMultiTrajectory,
)
from .io.osm_query import query_features as query_osm_features # Renamed for clarity
from .planning.decomposition import boustrophedon_decomposition
from .planning.path_generators import generate_search_tasks # Placeholder for now
from .utils.logging_setup import get_logger
from .utils.plot_utils import plot_basemap


__version__ = "0.1.0" # Example version

__all__ = [
    "Environment",
    "EnvironmentBuilder",
    "GeoData",
    "GeoPoint",
    "GeoPolygon",
    "GeoMultiPolygon",
    "GeoTrajectory",
    "GeoMultiTrajectory",
    "GeospatialDataLoader",
    "query_osm_features",
    "boustrophedon_decomposition",
    "generate_search_tasks",
    "get_logger",
    "plot_basemap",
    "__version__",
]

log = get_logger()
log.info(f"SARenv package version {__version__} initialized.")