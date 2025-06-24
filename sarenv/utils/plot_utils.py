# sarenv/utils/plot_utils.py
import contextily as ctx  # Ensure contextily is in requirements.txt
import matplotlib.pyplot as plt
import shapely
from shapely.affinity import translate

FEATURE_COLOR_MAP = {
    # --- Infrastructure / Man-made ---
    # Greys and browns for concrete, metal, and wood structures.
    "structure": '#636363',  # Dark Grey (e.g., buildings)
    "road": '#bdbdbd',       # Light Grey (e.g., roads, paths)
    "linear": '#8B4513',     # Saddle Brown (e.g., fences, railways, pipelines)

    # --- Water Features ---
    # Blues for all water-related elements.
    "water": '#3182bd',      # Strong Blue (e.g., lakes, rivers)
    "drainage": '#9ecae1',   # Light Blue (e.g., ditches, canals)

    # --- Vegetation ---
    # Greens and yellows for different types of plant life.
    "woodland": '#31a354',   # Forest Green (e.g., forests)
    "scrub": '#78c679',      # Muted Green (e.g., scrubland)
    "brush": '#c2e699',      # Very Light Green (e.g., grass)
    "field": '#fee08b',      # Golden Yellow (e.g., farmland, meadows)
    
    # --- Natural Terrain ---
    # Earth tones for rock and soil.
    "rock": '#969696',       # Stony Grey (e.g., cliffs, bare rock)
}
DEFAULT_COLOR = '#f0f0f0' # A very light, neutral default color.


def plot_basemap(
    ax=None, source=ctx.providers.Esri.WorldImagery, crs="EPSG:4326"
):  # Default to WGS84 for basemaps typically
    """
    Adds a basemap to a matplotlib axes.

    Args:
        ax (matplotlib.axes.Axes, optional): The axes to add the basemap to.
            If None, uses the current axes. Defaults to None.
        provider (contextily.providers object, optional): The tile provider.
            Defaults to ctx.providers.Esri.WorldImagery.
        crs (str, optional): The coordinate reference system of the plot data,
            so contextily can reproject tiles correctly. Defaults to "EPSG:4326".

    Returns:
        matplotlib.axes.Axes: The axes with the basemap.
    """
    if ax is None:
        ax = plt.gca()
    try:
        ctx.add_basemap(ax, source=source, crs=crs)
    except Exception as e:
        print(
            f"Could not add basemap: {e}. Ensure you have an internet connection and an appropriate CRS."
        )
    return ax


def normalize_coordinates(
    boundary_polygon: shapely.Polygon,
    geometries_to_normalize: list | shapely.MultiPolygon | shapely.MultiLineString,
):
    """
    Normalizes coordinates of geometries by translating them so that the
    min x and min y of the boundary_polygon's bounds become (0,0).

    Args:
        boundary_polygon (shapely.Polygon): The reference polygon whose bounds define the translation.
        geometries_to_normalize (list | shapely.MultiPolygon | shapely.MultiLineString):
            A list of Shapely geometries or a single MultiPolygon/MultiLineString to normalize.

    Returns:
        list: A list of normalized Shapely geometries.
              Returns an empty list if input geometries_to_normalize is empty or not of expected type.
    """
    if not boundary_polygon or boundary_polygon.is_empty:
        # print("Warning: Boundary polygon is empty or None. Cannot normalize.")
        return []  # Or return original geometries

    min_x, min_y, _, _ = boundary_polygon.bounds
    translation_x_offset = -min_x
    translation_y_offset = -min_y

    normalized_geoms = []

    geoms_to_process = []
    if isinstance(geometries_to_normalize, list):
        geoms_to_process = geometries_to_normalize
    elif hasattr(
        geometries_to_normalize, "geoms"
    ):  # Handles MultiPolygon, MultiLineString
        geoms_to_process = list(geometries_to_normalize.geoms)
    elif isinstance(
        geometries_to_normalize, shapely.geometry.base.BaseGeometry
    ):  # Single geometry
        geoms_to_process = [geometries_to_normalize]

    for geom in geoms_to_process:
        if geom and not geom.is_empty:
            normalized_geoms.append(
                translate(geom, xoff=translation_x_offset, yoff=translation_y_offset)
            )
    return normalized_geoms


# The commented-out cv2 code (create_image_from_multilinestring) can remain here if you plan to use it.
# Ensure cv2 (opencv-python) and numpy are in requirements.txt if you uncomment and use it.
# Example:
# import cv2
# from scipy.ndimage import gaussian_filter # Ensure scipy is in requirements.txt
# def create_image_from_multilinestring(...): ...
