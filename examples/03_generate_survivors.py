import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sarenv import (
    DatasetLoader,
    SurvivorLocationGenerator,
    get_logger,
)
from sarenv.utils.plot import DEFAULT_COLOR, FEATURE_COLOR_MAP

log = get_logger()

def run_survivor_generation_example(num_locations=1000, size_to_load="small"):
    """
    An example demonstrating how to load a dataset, generate survivor
    locations, and visualize them.

    Args:
        num_locations (int): The number of survivor locations to generate.
        size_to_load (str): The dataset size to use for the environment.
    """
    log.info("--- Starting Survivor Location Generation Example ---")

    dataset_dir = "sarenv_dataset"

    try:
        # 1. Load the dataset for a specific size
        log.info(f"Loading data for size: '{size_to_load}'")
        loader = DatasetLoader(dataset_directory=dataset_dir)
        dataset_item = loader.load_size(size_to_load)

        if not dataset_item:
            log.error(f"Could not load the dataset for size '{size_to_load}'.")
            return

        # 2. Initialize the survivor location generator with the loaded data
        log.info("Initializing the SurvivorLocationGenerator.")
        survivor_generator = SurvivorLocationGenerator(dataset_item)

        # 3. Generate survivor locations
        log.info(f"Generating {num_locations} survivor locations...")
        locations = []
        for i in range(num_locations):
            location = survivor_generator.generate_location()
            if location:
                log.info(f"Generated location {i+1}: {location.wkt}")
                locations.append(location)
            else:
                log.warning(f"Failed to generate location {i+1}.")

        if not locations:
            log.error("No survivor locations were generated. Cannot visualize.")
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

        # Plot the generated survivor locations
        survivor_gdf = gpd.GeoDataFrame(geometry=locations, crs=dataset_item.features.crs)
        survivor_gdf.plot(ax=ax, marker='*', color='red', markersize=250, zorder=10, label="Survivor")
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Survivor'))

        # Add basemap for context
        cx.add_basemap(ax, crs=survivor_gdf.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

        ax.legend(handles=legend_handles, title="Legend", loc="upper left")
        ax.set_title(f"Generated Survivor Locations within '{size_to_load.capitalize()}' Area")
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
    run_survivor_generation_example()
