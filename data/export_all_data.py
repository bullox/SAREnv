import os
from sarenv import (
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
    DataGenerator,
    get_logger,
)

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
from sarenv.utils.geo import get_utm_epsg
from sarenv.utils.plot import DEFAULT_COLOR, FEATURE_COLOR_MAP
from shapely.geometry import Point
from sarenv.utils.lost_person_behavior import get_environment_radius

log = get_logger()


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
    # ax.set_title(f"Heatmap Visualization: Size '{item.size}'")
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
    radii = get_environment_radius(item.environment_type, item.environment_climate)
    
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
    colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta"]
    for idx, r in enumerate(radii):
        circle = center_point_proj.buffer(r * 1000).iloc[0]
        color = colors[idx % len(colors)]
        gpd.GeoSeries([circle], crs=data_crs).boundary.plot(
            ax=ax, edgecolor=color, linestyle="--", linewidth=1.5, alpha=0.5
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
    # ax.set_title(f"Nested Dataset Visualization (Base: {item.size})")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")

    plt.tight_layout()
    plt.savefig(f"features_{item.size}.png")
    plt.plot()


log = get_logger()

# Your points data: (latitude, longitude, climate, environment_type)
# We'll use longitude, latitude order for center_point as in your example (x, y)
points = [
    # Flat points
    (2.648347, 51.341294, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (11.558208, 55.360132, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (-82.720334, 32.278283, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (-69.333197, 45.153006, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (-1.252642, 48.578875, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (4.583587, 49.29732, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (23.758429, 52.668738, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (1.60060306317937, 52.6157711622207, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (-88.9468665054608, 40.7887227887736, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (-89.941929851357, 46.5435953664409, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (11.950155112765817, 55.808401710098, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (4.864062127616984, 52.8291580321954, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (-76.3553947956702, 37.759607769476, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
    (11.283101409164392, 53.9091960386377, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),

    # Mountainous points
    (9.83830402184241, 46.8265127522274, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (6.415801583995766, 45.7029165747228, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (-3.374092471406765, 51.8529288806832, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (2.739522, 45.599168, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (12.820894, 47.229553, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (11.629584, 46.533445, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (-83.570884, 35.615547, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (-3.069811, 54.467693, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (8.48299455519022, 46.5888440483149, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (-123.224362031999, 49.5498941573725, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (6.65686147014046, 60.3385017431981, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (-5.51765837073881, 57.569832500662, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (-7.19519, 62.153469, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    (-23.1622490822439, 66.1032965556656, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
]

def generate_all():
    data_gen = DataGenerator()
    base_output_dir = "sarenv_dataset"

    count_flat = 0
    count_mountainous = 0
    size_to_load = "xlarge" 
    for lon, lat, climate, env_type in points:
        if env_type == ENVIRONMENT_TYPE_FLAT:
            count_flat += 1
            out_dir = os.path.join(base_output_dir, "temperate", "flat", str(count_flat))
        else:
            count_mountainous += 1
            out_dir = os.path.join(base_output_dir, "temperate", "mountainous", str(count_mountainous))

        os.makedirs(out_dir, exist_ok=True)
        log.info(f"Generating dataset for point ({lat}, {lon}) at {out_dir}")

        data_gen.export_dataset(
            center_point=(lon, lat),
            output_directory=out_dir,
            environment_climate=climate,
            environment_type=env_type,
            meter_per_bin=30,
        )

    for lon, lat, climate, env_type in points:
        if env_type == ENVIRONMENT_TYPE_FLAT:
            count_flat += 1
            out_dir = os.path.join(base_output_dir, "temperate", "flat", str(count_flat))
        else:
            count_mountainous += 1
            out_dir = os.path.join(base_output_dir, "temperate", "mountainous", str(count_mountainous))

        os.makedirs(out_dir, exist_ok=True)
        log.info(f"Generating dataset for point ({lat}, {lon}) at {out_dir}")

        try:
            # Initialize the new DynamicDatasetLoader
            loader = DatasetLoader(dataset_directory=out_dir)

            log.info(f"Loading data for size: '{size_to_load}'")
            item = loader.load_environment(size_to_load)

            if item:
                # Call the new all-in-one visualization function
                visualize_features(item, False)
                plt.savefig(os.path.join(out_dir, f"features_{item.size}.png"))
                visualize_heatmap(item, False)
                plt.savefig(os.path.join(out_dir, f"heatmap_{item.size}.png"))
            else:
                log.error(f"Could not load the specified size: '{size_to_load}'")

        except FileNotFoundError:
            log.error(
                f"Error: The dataset directory '{out_dir}' or its master files were not found."
            )
            log.error(
                "Please run the `export_dataset()` method from the DataGenerator first."
            )
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    generate_all()
