# examples/contour_planning_example.py
import matplotlib.pyplot as plt
from sarenv import EnvironmentBuilder, get_logger
from sarenv.planning import generate_fixed_width_contours # Import your new function
import geopandas as gpd

log = get_logger()

def run_contour_planning_example():
    log.info("Starting SARenv contour planning example...")

    # --- 1. Setup Environment (same as your basic example) ---
    tags_mapping = {
        "structure": {"building": True},
        "road": {"highway": True},
        "water": {"natural": "water"},
        "woodland": {"natural": "wood"},
    }
    builder = EnvironmentBuilder()
    for feature, tags in tags_mapping.items():
        builder.set_feature(feature, tags)

    # Make sure this GeoJSON file exists or use your own
    polygon_geojson_file = "FlatTerrainNature.geojson"
    try:
        env = builder.set_polygon(polygon_geojson_file).set_meter_per_bin(3).build()
    except FileNotFoundError:
        log.error(f"'{polygon_geojson_file}' not found. Please create it or use a valid path.")
        return

    # --- 2. Generate the Heatmap ---
    log.info("Generating combined heatmap...")
    # Customize sigma (blur) and alpha (weight) for each feature type
    sigma_features = {"road": 5, "water": 8, "woodland": 10, "structure": 2}
    alpha_features = {"road": 2, "water": 5, "woodland": 3, "structure": 1}
    heatmap = env.get_combined_heatmap(sigma_features, alpha_features)

    if heatmap is None or heatmap.sum() == 0:
        log.error("Failed to generate a valid heatmap. Exiting.")
        return

    # --- 3. Generate Fixed-Width Contour Trajectories ---
    SENSOR_WIDTH_METERS = 20.0 # Example: 20-meter effective sensor width for drone
    trajectories = generate_fixed_width_contours(
        environment=env,
        heatmap=heatmap,
        fixed_width=SENSOR_WIDTH_METERS,
        threshold=1e-6 # Adjust this based on your heatmap's values
    )

    if not trajectories:
        log.warning("No trajectories were generated. Try adjusting the threshold or heatmap parameters.")
        return

    # --- 4. Visualize the Results ---
    log.info("Visualizing results...")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Contour Trajectories (Fixed Width: {SENSOR_WIDTH_METERS}m)")

    # Plot the heatmap lightly in the background
    extent = [env.xedges[0], env.xedges[-1], env.yedges[0], env.yedges[-1]]
    ax.imshow(heatmap, extent=extent, origin="lower", cmap='viridis', alpha=0.5)

    # Plot the generated trajectories
    gpd.GeoSeries(trajectories).plot(ax=ax, color="magenta", linewidth=1.5, label="Coverage Trajectories")

    # Plot the original boundary
    env.polygon.plot(ax=ax, facecolor="none", edgecolor="black", linestyle="--", linewidth=1.5, label="Search Boundary")

    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    run_contour_planning_example()