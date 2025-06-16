# sarenv/__init__.py
"""
SAR Environment (sarenv) package for simulating and evaluating Search and Rescue scenarios.
"""

# Import key classes and functions to make them easily accessible at the package level
from .core.environment import DataGenerator, Environment, EnvironmentBuilder
from .core.geometries import (
    GeoData,
    GeoMultiPolygon,
    GeoMultiTrajectory,
    GeoPoint,
    GeoPolygon,
    GeoTrajectory,
)
from .io.loaders import DatasetLoader, SARDatasetItem
from .io.osm_query import query_features as query_osm_features  # Renamed for clarity
from .planning.decomposition import boustrophedon_decomposition
from .planning.path_generators import (  # Placeholder for now
    generate_fixed_width_contours,
    generate_search_tasks,
)
from .utils.logging_setup import get_logger
from .utils.plot_utils import plot_basemap, FEATURE_COLOR_MAP, DEFAULT_COLOR
__version__ = "0.1.0"

__all__ = [
    "DataGenerator",
    "DatasetLoader",
    "Environment",
    "EnvironmentBuilder",
    "GeoData",
    "GeoMultiPolygon",
    "GeoMultiTrajectory",
    "GeoPoint",
    "GeoPolygon",
    "GeoTrajectory",
    "GeospatialDataLoader",
    "SARDatasetItem",
    "__version__",
    "boustrophedon_decomposition",
    "generate_fixed_width_contours",
    "generate_search_tasks",
    "get_logger",
    "plot_basemap",
    "query_osm_features",
]

log = get_logger()
log.info(f"SARenv package version {__version__} initialized.")