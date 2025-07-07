import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sarenv import (
    DatasetLoader,
    LostPersonLocationGenerator,
    get_logger,
)
from sarenv.utils.plot import DEFAULT_COLOR, FEATURE_COLOR_MAP

log = get_logger()

def run_lost_person_generation_example(num_locations=1000, size_to_load="small"):
    """
    An example demonstrating how to load a dataset, generate lost_person
    locations, and visualize them.

    Args:
        num_locations (int): The number of lost_person locations to generate.
        size_to_load (str): The dataset size to use for the environment.
    """
    log.info("--- Starting lost_person Location Generation Example ---")

    dataset_dir = "sarenv_dataset"

    try:
        # 1. Load the dataset for a specific size
        log.info(f"Loading data for size: '{size_to_load}'")
        loader = DatasetLoader(dataset_directory=dataset_dir)
        dataset_item = loader.load_environment(size_to_load)

        if not dataset_item:
            log.error(f"Could not load the dataset for size '{size_to_load}'.")
            return

        # 2. Initialize the lost_person location generator with the loaded data
        log.info("Initializing the lost_personLocationGenerator.")
        lost_person_generator = LostPersonLocationGenerator(dataset_item)

        # 3. Generate lost_person locations
        log.info(f"Generating {num_locations} lost_person locations...")
        locations = lost_person_generator.generate_locations(num_locations)

        if not locations:
            log.error("No lost_person locations were generated. Cannot visualize.")
            return

        # 4. Visualize the results
        log.info("Visualizing the generated locations...")
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot the features from the dataset
        legend_handles = []
        for feature_type, data in dataset_item.features.groupby("feature_type"):
            color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
            data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.6)
            legend_handles.append(Patch(color=color, label=feature_type.capitalize()))

        # Plot the generated lost_person locations
        lost_person_gdf = gpd.GeoDataFrame(geometry=locations, crs=dataset_item.features.crs)
        lost_person_gdf.plot(ax=ax, marker='*', color='red', markersize=250, zorder=10, label="Lost Person")
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Lost Person'))

        # Add basemap for context
        cx.add_basemap(ax, crs=lost_person_gdf.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

        ax.legend(handles=legend_handles, title="Legend", loc="upper left")
        ax.set_title(f"Generated Lost Person Locations within '{size_to_load.capitalize()}' Area")
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        plt.tight_layout()
        plt.show()

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
    # Before running this, ensure you have run basic_usage_example.py to create the dataset
    run_lost_person_generation_example(num_locations=1000, size_to_load="xlarge")
