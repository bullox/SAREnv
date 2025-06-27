
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sarenv import (
    DatasetLoader,
    SARDatasetItem,
    get_logger,
)
from sarenv.utils.plot_utils import DEFAULT_COLOR, FEATURE_COLOR_MAP
from shapely.geometry import Point

log = get_logger()


def get_utm_epsg(lon: float, lat: float) -> str:
    """Calculates the appropriate UTM zone EPSG code for a given point."""
    zone = int((lon + 180) / 6) + 1
    epsg_code = f"326{zone}" if lat >= 0 else f"327{zone}"
    log.info(f"Determined UTM zone for point ({lon}, {lat}) as EPSG:{epsg_code}")
    return f"EPSG:{epsg_code}"


def visualize_heatmap(item: SARDatasetItem, plot_basemap: bool = True):
    """
    Creates a plot to visualize the heatmap of a single SARDatasetItem.

    Args:
        item (SARDatasetItem): The loaded dataset item to visualize.
    """
    # Updated to use 'size' instead of 'quantile'
    log.info(f"Generating heatmap visualization for size: {item.size}...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Determine the correct projected CRS for the item's location
    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    minx, miny, maxx, maxy = item.bounds
    # Plot the heatmap with the calculated extent
    im = ax.imshow(
        item.heatmap, extent=(minx, maxx, miny, maxy), origin="lower", cmap="inferno"
    )
    # Add a colorbar to show the probability scale
    fig.colorbar(im, ax=ax, shrink=0.8, label="Probability Density")
    if plot_basemap:
        cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7)

    # Updated title to use 'size'
    ax.set_title(f"Heatmap Visualization: Size '{item.size}'")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")
    plt.tight_layout()


def visualize_features(item: SARDatasetItem, plot_basemap: bool = True):
    """
    Creates a single plot showing the largest dataset in full, with the radii
    of the smaller datasets overlaid as circular borders.

    Args:
        items (List[SARDatasetItem]): A list of loaded dataset items. The function
                                     will use the one with the largest radius as the base.
    """
    if not item:
        log.warning("No dataset items provided to visualize.")
        return

    # Sort items by radius to easily find the largest one
    log.info(f"Generating nested visualization using '{item.size}' as the base...")
    center_point_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    )
    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    fig, ax = plt.subplots(figsize=(13, 13))

    legend_handles = []
    for feature_type, data in item.features.groupby("feature_type"):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
        data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.7)
        legend_handles.append(Patch(color=color, label=feature_type.capitalize()))

    radius_circle = center_point_proj.buffer(item.radius_km * 1000).iloc[0]


    gpd.GeoSeries([radius_circle], crs=data_crs).boundary.plot(
        ax=ax, edgecolor="blue", linestyle="--", linewidth=2.5
    )

    # Create a legend handle for the radius (updated to use 'size')
    label = f"Radius: {item.size} ({item.radius_km} km)"
    legend_handles.append(
        Line2D([0], [0], color=color, lw=2.5, linestyle="--", label=label)
    )
    if plot_basemap:
        cx.add_basemap(ax, crs=item.features.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

    ax.legend(handles=legend_handles, title="Legend", loc="upper left")
    # Updated title to use 'size'
    ax.set_title(f"Nested Dataset Visualization (Base: {item.size})")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")

    plt.tight_layout()
    plt.plot()


def run_loading_example():
    """
    An example function demonstrating how to load and visualize a single dataset.
    """
    log.info("--- Starting Single Dataset Loading and Visualization Example ---")

    dataset_dir = "sarenv_dataset"
    size_to_load = "xlarge"  # Define which single size you want to see

    try:
        # Initialize the new DynamicDatasetLoader
        loader = DatasetLoader(dataset_directory=dataset_dir)

        log.info(f"Loading data for size: '{size_to_load}'")
        item = loader.load_size(size_to_load)

        if item:
            # Call the new all-in-one visualization function
            visualize_features(item, False)
            visualize_heatmap(item, False)
            plt.show()
        else:
            log.error(f"Could not load the specified size: '{size_to_load}'")

    except FileNotFoundError:
        log.error(
            f"Error: The dataset directory '{dataset_dir}' or its master files were not found."
        )
        log.error(
            "Please run the `export_dataset()` method from the DataGenerator first."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    run_loading_example()
