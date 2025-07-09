# examples/04_evaluate_coverage_paths.py
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from shapely.geometry import Point, LineString
import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo
from sarenv.utils.plot import DEFAULT_COLOR, FEATURE_COLOR_MAP
from sarenv.analytics.evaluator import ComparativeDatasetEvaluator, ComparativeEvaluator

log = sarenv.get_logger()

COLORS_GRAY = ["#FFFFFF", "#808080", "#000000"]
COLORS_BLUE = ["#68FFFC", "#0099FF", "#00008B"]


# --- Parameters ---
GRAPHS_DIR = "graphs/paths"
DATASET_DIR = "sarenv_dataset/19"
EVALUATION_SIZE = "medium"  # Options: "small", "medium", "large", "xlarge"
NUM_DRONES = 3
NUM_VICTIMS = 100

# Drone/Path Parameters
FOV_DEGREES = 45.0
ALTITUDE_METERS = 80.0
OVERLAP_RATIO = 0.25
PATH_POINT_SPACING_M = 10.0
TRANSITION_DISTANCE_M = 50.0
PIZZA_BORDER_GAP_M = 15.0
DISCOUNT_FACTOR = 0.999
# DATASET_DIRS = [f"sarenv_dataset/{i}" for i in range(1, 61)]
DATASET_DIRS = [f"sarenv_dataset/{i}" for i in range(1, 4)]
# Define all path generator lambdas as in your code, but with item passed for context if needed
EXHAUSTIVE_PATH_NAMES = ["Spiral", "Concentric", "Pizza"]
PATH_GENERATORS = {
    "RandomWalk": lambda cx, cy, r, item, max_length: paths.generate_random_walk_path(
        cx, cy, NUM_DRONES, item.heatmap, item.bounds, max_length
    ),
    # "RandomNonColliding": lambda cx, cy, r, item: paths.generate_random_noncolliding_paths(
    #     cx, cy, r, NUM_DRONES, PATH_POINT_SPACING_M
    # ),
    "Greedy": lambda cx, cy, r, item, max_length: paths.generate_greedy_path(
        cx, cy, NUM_DRONES, item.heatmap, item.bounds, max_length
    ),
    # "Spiral": lambda cx, cy, r, item: paths.generate_spiral_path(
    #     cx, cy, r, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M
    # ),
    "Concentric": lambda cx, cy, r, item, max_length: paths.generate_concentric_circles_path(
        cx, cy, r, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M, TRANSITION_DISTANCE_M, max_length
    ),
    "Pizza": lambda cx, cy, r, item, max_length: paths.generate_pizza_zigzag_path(
        cx, cy, r, NUM_DRONES, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, PATH_POINT_SPACING_M, PIZZA_BORDER_GAP_M, max_length
    )
}

# --- Plotting Functions ---

def plot_features_map_clipped(item, features_clipped, victims_gdf, generated_paths, name, center_x, center_y, max_radius_m, x_min, x_max, y_min, y_max, output_file):
    fig, ax = plt.subplots(figsize=(9, 9))
    legend_handles = []
    # Plot features by type
    for feature_type, data in features_clipped.groupby("feature_type"):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
        data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.7)
        legend_handles.append(Patch(color=color, label=feature_type.capitalize()))
    # Plot victims
    if not victims_gdf.empty:
        victims_gdf.plot(ax=ax, color='red', marker='x', label='Victims', zorder=5)
        legend_handles.append(Line2D([0], [0], color='red', marker='x', linestyle='', label='Victims'))
    # Plot paths
    # colors = ['black', 'white'] # Alternate black and white
    colors = COLORS_BLUE

    line_width = 3.0
    for idx, path in enumerate(generated_paths):
        color = colors[idx % len(colors)]
        if path.geom_type == 'MultiLineString':
            for line in path.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
        else:
            x, y = path.xy
            ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
    # ax.set_title(f"{name} Coverage Path on Features Map (Clipped)")
    # x_ticks = ax.get_xticks()
    # y_ticks = ax.get_yticks()
    # ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks])
    # ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.legend(handles=legend_handles, title="Legend", loc="upper left")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    fig.savefig(output_file, format='pdf', dpi=200)
    plt.close(fig)

def plot_heatmap(item, victims_gdf, generated_paths, name, x_min, x_max, y_min, y_max, output_file):
    fig, ax = plt.subplots(figsize=(9, 9))
    if item.heatmap is not None:
        ax.imshow(
            item.heatmap,
            extent=[x_min, x_max, y_min, y_max],
            cmap='YlOrRd',
            alpha=0.7,
            origin='lower'
        )
    if not victims_gdf.empty:
        victims_gdf.plot(ax=ax, color='red', marker='x', label='Victims', zorder=5)
    # Plot paths
    colors = COLORS_BLUE
    line_width = 3.0
    for idx, path in enumerate(generated_paths):
        color = colors[idx % len(colors)]
        if path.geom_type == 'MultiLineString':
            for line in path.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
        else:
            x, y = path.xy
            ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
    # ax.set_title(f"{name} Coverage Path on Heatmap")
    # x_ticks = ax.get_xticks()
    # y_ticks = ax.get_yticks()
    # ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks])
    # ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.legend()
    fig.savefig(output_file, format='pdf', dpi=200)
    plt.close(fig)

def plot_side_by_side(features_clipped, item, victims_gdf, generated_paths, name, x_min, x_max, y_min, y_max, output_file):
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    # Left: Features
    ax = axs[0]
    for feature_type, data in features_clipped.groupby("feature_type"):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
        data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.7)
    if not victims_gdf.empty:
        victims_gdf.plot(ax=ax, color='red', marker='x', label='Victims', zorder=5)
    colors = COLORS_BLUE
    line_width = 3.0
    for idx, path in enumerate(generated_paths):
        color = colors[idx % len(colors)]
        if path.geom_type == 'MultiLineString':
            for line in path.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
        else:
            x, y = path.xy
            ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
    # ax.set_title(f"{name} Coverage Path on Features Map (Clipped)")
    # x_ticks = ax.get_xticks()
    # y_ticks = ax.get_yticks()
    # ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks])
    # ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    # Right: Heatmap
    ax = axs[1]
    if item.heatmap is not None:
        ax.imshow(
            item.heatmap,
            extent=[x_min, x_max, y_min, y_max],
            cmap='YlOrRd',
            alpha=0.7,
            origin='lower'
        )
    if not victims_gdf.empty:
        victims_gdf.plot(ax=ax, color='red', marker='x', label='Victims', zorder=5)
    for idx, path in enumerate(generated_paths):
        color = colors[idx % len(colors)]
        if path.geom_type == 'MultiLineString':
            for line in path.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
        else:
            x, y = path.xy
            ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
    # ax.set_title(f"{name} Coverage Path on Heatmap")
    # x_ticks = ax.get_xticks()
    # y_ticks = ax.get_yticks()
    # ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks])
    # ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(output_file, format='pdf', dpi=200)
    plt.close(fig)

def plot_combined_metrics(
    resample_distances,
    resampled_combined_cumulative_likelihood,
    resampled_combined_cumulative_victims,
    output_dir='graphs/plots',
    strategy_name='strategy'
):

    resample_distances_km = np.array(resample_distances) / 1000.0  # Convert to km

    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_likelihood = '#1f77b4'  # Blue for likelihood
    ax1.set_xlabel('Distance Covered (km)')
    ax1.set_ylabel('Total Accumulated Likelihood', color=color_likelihood)
    ax1.plot(
        resample_distances_km,
        resampled_combined_cumulative_likelihood,
        color=color_likelihood,
        label='Total Accumulated Likelihood'
    )
    ax1.tick_params(axis='y', labelcolor=color_likelihood)

    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color_victims = "#ff0e0e"  # Red for victims
    ax2.set_ylabel('Total Victims Found', color=color_victims)
    ax2.plot(
        resample_distances_km,
        resampled_combined_cumulative_victims,
        color=color_victims,
        label='Total Victims Found'
    )
    ax2.tick_params(axis='y', labelcolor=color_victims)

    plt.title(f'Combined Metrics for {strategy_name}')
    fig.tight_layout()

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    filename = os.path.join(output_dir, f'{strategy_name.lower()}_{EVALUATION_SIZE}_{NUM_DRONES}drones_{NUM_VICTIMS}victims_combined_metrics.pdf')
    plt.savefig(filename)
    plt.close()
    print(f"Combined plot saved to {filename}")

def crop_paths_to_length(paths, length):
    # Crop each LineString to the specified length
    return [LineString(list(path.coords)[:length]) for path in paths]

def run_evaluation():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    try:
        # 1. Load Data
        loader = sarenv.DatasetLoader(dataset_directory=DATASET_DIR)
        item = loader.load_environment(EVALUATION_SIZE)
        if not item:
            log.error(f"Could not load data. Please run '01_generate_dataset.py' first.")
            return

        data_crs = geo.get_utm_epsg(item.center_point[0], item.center_point[1])
        
        # 2. Generate Victim Locations
        log.info(f"Generating {NUM_VICTIMS} victim locations...")
        victim_generator = sarenv.LostPersonLocationGenerator(item)
        victim_points = [p for p in (victim_generator.generate_location() for _ in range(NUM_VICTIMS)) if p]
        victims_gdf = gpd.GeoDataFrame(geometry=victim_points, crs=data_crs) if victim_points else gpd.GeoDataFrame(columns=['geometry'], crs=data_crs)

        # 3. Setup for Path Generation and Evaluation
        evaluator = metrics.PathEvaluator(
            item.heatmap,
            item.bounds,
            victims_gdf,
            FOV_DEGREES,
            ALTITUDE_METERS,
            loader._meter_per_bin
        )
        center_proj = gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326").to_crs(data_crs).geometry.iloc[0]
        center_x, center_y = center_proj.x, center_proj.y
        max_radius_m = item.radius_km * 1000

        path_generators = PATH_GENERATORS

        # Generate 
        all_generated_paths = {}
        all_paths_flat = []
        exhaustive_paths_max_distance = 0
        for name, generator in path_generators.items():
            if name not in EXHAUSTIVE_PATH_NAMES:
                continue  # Skip non-exhaustive paths for this evaluation
            generated_paths = generator(
                center_x, center_y, max_radius_m, item, None
            )
            generated_paths_max_distance = paths.calculate_max_distance_in_paths(generated_paths)
            if (generated_paths_max_distance > exhaustive_paths_max_distance):
                exhaustive_paths_max_distance = generated_paths_max_distance

        # Generate all again with max_length
        for name, generator in path_generators.items():
            generated_paths = generator(
                center_x, center_y, max_radius_m, item, exhaustive_paths_max_distance
            )
            all_generated_paths[name] = generated_paths
            all_paths_flat.extend(generated_paths)

        # 4. Evaluate and Visualize Each Path
        for name, generated_paths in all_generated_paths.items():
            log.info(f"--- Evaluating '{name}' Path ---")
            
            all_metrics = evaluator.calculate_all_metrics(generated_paths, discount_factor=0.999)
            likelihood = all_metrics['total_likelihood_score']
            discounted = all_metrics['total_time_discounted_score']
            victim_metrics = all_metrics['victim_detection_metrics']

            print(f"  Likelihood Score: {likelihood:.2f}")
            print(f"  Time-Discounted Score: {discounted:.2f}")
            print(f"  Victims Found: {victim_metrics['percentage_found']:.2f}%")
            print(f"  Found Victim Indices: {victim_metrics['found_victim_indices']}")

            # Prepare features and circle area for plotting
            features_proj = item.features.to_crs(data_crs)
            circle_area = center_proj.buffer(max_radius_m)
            features_clipped = gpd.clip(features_proj, circle_area)

            # Axis limits
            x_min, x_max = center_x - max_radius_m, center_x + max_radius_m
            y_min, y_max = center_y - max_radius_m, center_y + max_radius_m

            # File names
            output_file_left = os.path.join(GRAPHS_DIR, f"path_{name.lower()}_{EVALUATION_SIZE}_{NUM_DRONES}drones_{NUM_VICTIMS}victims_features_map.pdf")
            output_file_right = os.path.join(GRAPHS_DIR, f"path_{name.lower()}_{EVALUATION_SIZE}_{NUM_DRONES}drones_{NUM_VICTIMS}victims_heatmap.pdf")
            output_file_side = os.path.join(GRAPHS_DIR, f"path_{name.lower()}_{EVALUATION_SIZE}_{NUM_DRONES}drones_{NUM_VICTIMS}victims_side_by_side.pdf")

            # Plot and save visualizations
            log.info(f"Generating plots for {name} path...")
            plot_features_map_clipped(
                item, features_clipped, victims_gdf, generated_paths, name,
                center_x, center_y, max_radius_m, x_min, x_max, y_min, y_max, output_file_left
            )
            plot_heatmap(
                item, victims_gdf, generated_paths, name,
                x_min, x_max, y_min, y_max, output_file_right
            )
            plot_side_by_side(
                features_clipped, item, victims_gdf, generated_paths, name,
                x_min, x_max, y_min, y_max, output_file_side
            )
            log.info(f"Saved visualizations for {name} to '{GRAPHS_DIR}/'")
            
            plot_combined_metrics(
                all_metrics['resample_distances'],
                all_metrics['resampled_combined_cumulative_likelihood'],
                all_metrics['resampled_combined_cumulative_victims'],
                output_dir='graphs/plots',
                strategy_name=name
            )

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    log.info("--- Starting Multi-Drone Path Evaluation and Visualization ---")
    run_evaluation()
    log.info("--- Finished Multi-Drone Path Evaluation and Visualization ---")


    # log.info("--- Starting Comparative Dataset Evaluator ---")
    # # 1. Initialize the evaluator
    # evaluator = ComparativeDatasetEvaluator(
    #     dataset_dirs=DATASET_DIRS,
    #     path_generators=PATH_GENERATORS,
    #     num_victims=NUM_VICTIMS,
    #     evaluation_size=EVALUATION_SIZE,
    #     fov_degrees=FOV_DEGREES,
    #     altitude_meters=ALTITUDE_METERS,
    #     overlap_ratio=OVERLAP_RATIO,
    #     num_drones=NUM_DRONES,
    #     path_point_spacing_m=PATH_POINT_SPACING_M,
    #     transition_distance_m=TRANSITION_DISTANCE_M,
    #     pizza_border_gap_m=PIZZA_BORDER_GAP_M,
    #     discount_factor=DISCOUNT_FACTOR
    # )

    # # 2. Run the evaluations
    # baseline_results = evaluator.evaluate()

    # # 3. Show summary of results
    # per_dataset_results_df = evaluator.get_results_per_dataset()
    # summarized_results_df = evaluator.summarize_results()
    # log.info("--- Summary of Results ---")
    # print(summarized_results_df)

    # # 4. Save results to CSV
    # per_dataset_results_csv_path = os.path.join("graphs/comparative_plots", f'per_dataset_comparative_evaluation_results_{EVALUATION_SIZE}.csv')
    # summarized_results_csv_path = os.path.join("graphs/comparative_plots", f'summarized_comparative_evaluation_results_{EVALUATION_SIZE}.csv')
    # per_dataset_results_df.to_csv(per_dataset_results_csv_path, index=False)
    # summarized_results_df.to_csv(summarized_results_csv_path, index=False)
    # log.info(f"Results saved to {per_dataset_results_csv_path} and {summarized_results_csv_path}")

    # # 5. Generate comparative plots
    # # evaluator.plot_aggregate_bars(output_dir="graphs/comparative_plots")
    # evaluator.plot_combined_normalized_bars(output_dir="graphs/comparative_plots")
    # evaluator.plot_time_series_with_ci(output_dir="graphs/comparative_plots")
    # log.info("--- Finished Comparative Dataset Evaluator ---")