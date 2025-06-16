# examples/dataset_loading_example.py
import os
from typing import List
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import Point

from sarenv import (
    DatasetLoader,
    SARDatasetItem,
    get_logger,
)
from sarenv.utils.plot_utils import FEATURE_COLOR_MAP, DEFAULT_COLOR

log = get_logger()

def get_utm_epsg(lon: float, lat: float) -> str:
    """Calculates the appropriate UTM zone EPSG code for a given point."""
    zone = int((lon + 180) / 6) + 1
    epsg_code = f"326{zone}" if lat >= 0 else f"327{zone}"
    log.info(f"Determined UTM zone for point ({lon}, {lat}) as EPSG:{epsg_code}")
    return f"EPSG:{epsg_code}"



def visualize_heatmap(item: SARDatasetItem):
    """
    Creates a plot to visualize the heatmap of a single SARDatasetItem.

    Args:
        item (SARDatasetItem): The loaded dataset item to visualize.
    """
    log.info(f"Generating heatmap visualization for quantile: {item.quantile}...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Determine the correct projected CRS for the item's location
    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    
    # Project features to the local CRS to get the correct coordinate extent
    features_proj = item.features.to_crs(crs=data_crs)
    minx, miny, maxx, maxy = features_proj.total_bounds

    # Plot the heatmap with the calculated extent
    im = ax.imshow(item.heatmap, extent=(minx, maxx, miny, maxy), origin='lower', cmap='inferno')
    
    # Add a colorbar to show the probability scale
    fig.colorbar(im, ax=ax, shrink=0.8, label='Probability Density')
    
    # # Add a geographical basemap for context
    # cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7)
    
    ax.set_title(f"Heatmap Visualization: Quantile '{item.quantile}'")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")
    plt.tight_layout()
    plt.show()


def visualize_features(items: List[SARDatasetItem]):
    """
    Creates a single plot showing the largest dataset in full, with the radii
    of the smaller datasets overlaid as circular borders.

    Args:
        items (List[SARDatasetItem]): A list of loaded dataset items. The function
                                     will use the one with the largest radius as the base.
    """
    if not items:
        log.warning("No dataset items provided to visualize.")
        return

    # Sort items by radius to easily find the largest one
    items.sort(key=lambda i: i.radius_km)
    largest_item = items[-1]
    log.info(f"Generating nested visualization using '{largest_item.quantile}' as the base...")
    center_point_gdf = gpd.GeoDataFrame(geometry=[Point(largest_item.center_point)], crs="EPSG:4326")
    data_crs = get_utm_epsg(largest_item.center_point[0], largest_item.center_point[1])

    largest_item.features.to_crs(crs=data_crs, inplace=True)
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    fig, ax = plt.subplots(figsize=(13, 13))

    # 3. Plot the features of the largest dataset
    legend_handles = []
    for feature_type, data in largest_item.features.groupby('feature_type'):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)

        data.plot(ax=ax, color=color, label=feature_type.capitalize(),alpha=0.7)
        legend_handles.append(Patch(color=color, label=feature_type.capitalize()))

    # 4. Plot the circular borders for ALL radii
    radii_colors = ['yellow', 'orange', 'red', 'blue'] # Colors for q1, median, q3
    for i, item in enumerate(items):
        # Create a circle geometry from the item's center point and radius
        center_point_gdf = gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
        radius_circle = center_point_proj.buffer(item.radius_km * 1000).iloc[0]

        # Plot the boundary of the circle
        color = radii_colors[i % len(radii_colors)]
        gpd.GeoSeries([radius_circle], crs=data_crs).boundary.plot(
            ax=ax,
            edgecolor=color,
            linestyle='--',
            linewidth=2.5
        )
        # Create a legend handle for the radius
        label = f"Radius: {item.quantile} ({item.radius_km} km)"
        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, linestyle='--', label=label))

    # 5. Add basemap and finalize the plot
    cx.add_basemap(ax, crs=largest_item.features.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

    ax.legend(handles=legend_handles, title="Legend", loc='upper left')
    ax.set_title(f"Nested Dataset Visualization (Base: {largest_item.quantile})")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")

    plt.tight_layout()
    plt.show()


def run_loading_example():
    """
    An example function demonstrating how to load and visualize nested datasets.
    """
    log.info("--- Starting Nested Radius Loading and Visualization Example ---")

    dataset_dir = "sarenv_dataset"

    try:
        # Initialize the loader from sarenv/io/loaders.py
        loader = DatasetLoader(dataset_directory=dataset_dir)

        # Define which quantiles we want to visualize
        quantiles_to_load = ['q1', 'median', 'q3', 'q95']
        loaded_items = []

        log.info(f"Loading data for quantiles: {quantiles_to_load}")
        for q in quantiles_to_load:
            # SARDatasetItem is defined in sarenv/io/loaders.py
            item = loader.load_quantile(q)
            if item:
                loaded_items.append(item)

        if loaded_items:
            # Call the new visualization function with the list of loaded items
            visualize_features(loaded_items)
            visualize_heatmap(loaded_items[-1]) # The list is sorted by radius

        else:
            log.error(f"Could not load any of the specified quantiles: {quantiles_to_load}")

    except FileNotFoundError:
        log.error(f"Error: The dataset directory '{dataset_dir}' was not found.")
        log.error("Please run the 'run_export_example()' from a previous step first.")
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    run_loading_example()
