# examples/basic_usage_example.py
from sarenv import (
    EnvironmentBuilder,
    get_logger,
)  # Use the public API from sarenv's __init__

log = get_logger()  # Initialize and get the logger


def run_basic_example():
    log.info("Starting SARenv basic usage example...")

    tags_mapping = {
        "structure": {
            "building": True,
            "man_made": True,
            "bridge": True,
            "tunnel": True,
        },
        "road": {"highway": True, "tracktype": True},
        "linear": {
            "railway": True,
            "barrier": True,
            "fence": True,
            "wall": True,
            "pipeline": True,
        },
        "drainage": {"waterway": ["drain", "ditch", "culvert", "canal"]},
        "water": {
            "natural": ["water", "wetland"],
            "water": True,
            "wetland": True,
            "reservoir": True,
        },
        "brush": {
            "landuse": ["grass"]
        },  # TODO: check if meadow is supposed to be in this feature
        "scrub": {"natural": "scrub"},
        "woodland": {"landuse": ["forest", "wood"], "natural": "wood"},
        "field": {"landuse": ["farmland", "farm", "meadow"]},
        "rock": {"natural": ["rock", "bare_rock", "scree", "cliff"]},
    }

    builder = EnvironmentBuilder()
    for feature_category, osm_tags in tags_mapping.items():
        builder.set_feature(feature_category, osm_tags)

    # You'll need to provide a GeoJSON file for the boundary.
    # Create a dummy GeoJSON for testing if you don't have one readily available.
    # Example: FlatTerrainNature.geojson
    # Make sure this file exists in the same directory as this script, or provide a full path.
    polygon_geojson_file = (
        "FlatTerrainNature.geojson"  # Replace with your actual file path
    )

    try:
        log.info(f"Attempting to load environment from: {polygon_geojson_file}")
        env = (
            builder.set_polygon_file(polygon_geojson_file).set_meter_per_bin(5).build()
        )
        log.info("Environment built successfully.")

        log.info("Visualizing environment features...")
        env.visualise_environment()  # Shows raw features

        log.info(
            "Generating and displaying combined heatmap with interactive sliders..."
        )
        # Ensure heatmaps are generated before calling plot_heatmap_interactive if it relies on self.heatmaps
        # env.generate_heatmaps() # This is now called within get_combined_heatmap if needed

        initial_combined_heatmap = env.get_combined_heatmap()  # Get an initial heatmap
        if initial_combined_heatmap is not None:
            env.plot_heatmap_interactive(
                initial_heatmap=initial_combined_heatmap,
                show_basemap=False,
                show_features=False,  # Keep this false for interactive heatmap clarity
                show_coverage_paths=False,
                export_final_image=True,
            )
        else:
            log.error("Failed to generate initial combined heatmap.")

        # Example of using the general plot function
        # log.info("Displaying general plot...")
        # env.plot(show_heatmap=True, show_coverage=True, show_features=True)

    except FileNotFoundError:
        log.error(f"Error: The GeoJSON file '{polygon_geojson_file}' was not found.")
        log.error("Please create this file or provide the correct path.")
        log.error("A simple GeoJSON polygon example:")
        log.error(
            """
        {
          "type": "FeatureCollection",
          "features": [
            {
              "type": "Feature",
              "properties": {},
              "geometry": {
                "type": "Polygon",
                "coordinates": [
                  [ [10.0, 55.0], [10.1, 55.0], [10.1, 55.1], [10.0, 55.1], [10.0, 55.0] ]
                ]
              }
            }
          ]
        }
        """
        )
    except Exception as e:
        log.error(f"An error occurred during the example run: {e}", exc_info=True)


if __name__ == "__main__":
    run_basic_example()
