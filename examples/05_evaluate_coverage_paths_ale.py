# examples/04_evaluate_coverage_paths.py
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from shapely.geometry import Point
import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo
from sarenv.utils.plot import DEFAULT_COLOR, FEATURE_COLOR_MAP
from sarenv.analytics.evaluator import ComparativeEvaluator

log = sarenv.get_logger()

COLORS_GRAY = ["#FFFFFF", "#808080", "#000000"]
COLORS_BLUE = ["#68FFFC", "#0099FF", "#00008B"]


# --- Parameters ---
GRAPHS_DIR = "graphs"
DATASET_DIR = "sarenv_dataset/19"
EVALUATION_SIZE = "small"
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
DATASET_DIRS = [f"sarenv_dataset/{i}" for i in range(1, 11)]
# Define all path generator lambdas as in your code, but with item passed for context if needed
PATH_GENERATORS = {
    "RandomWalk": lambda cx, cy, r, item: paths.generate_random_walk_path(
        cx, cy, NUM_DRONES, item.heatmap, item.bounds, 3000
    ),
    # "RandomNonColliding": lambda cx, cy, r, item: paths.generate_random_noncolliding_paths(
    #     cx, cy, r, NUM_DRONES, PATH_POINT_SPACING_M
    # ),
    "Greedy": lambda cx, cy, r, item: paths.generate_greedy_path(
        cx, cy, NUM_DRONES, item.heatmap, item.bounds
    ),
    # "Spiral": lambda cx, cy, r, item: paths.generate_spiral_path(
    #     cx, cy, r, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M
    # ),
    "Concentric": lambda cx, cy, r, item: paths.generate_concentric_circles_path(
        cx, cy, r, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M, TRANSITION_DISTANCE_M
    ),
    "Pizza": lambda cx, cy, r, item: paths.generate_pizza_zigzag_path(
        cx, cy, r, NUM_DRONES, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, PATH_POINT_SPACING_M, PIZZA_BORDER_GAP_M
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
    combined_likelihood,
    combined_victims,
    output_dir='graphs/plots',
    strategy_name='strategy'
):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_likelihood = '#1f77b4'  # Blue for likelihood
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Total Accumulated Likelihood', color=color_likelihood)
    ax1.plot(combined_likelihood, color=color_likelihood, label='Total Accumulated Likelihood')
    ax1.tick_params(axis='y', labelcolor=color_likelihood)

    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color_victims = "#ff0e0e"  # Red for victims
    ax2.set_ylabel('Total Victims Found', color=color_victims)
    ax2.plot(combined_victims, color=color_victims, label='Total Victims Found')
    ax2.tick_params(axis='y', labelcolor=color_victims)

    plt.title(f'Combined Metrics for {strategy_name}')
    fig.tight_layout()

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    filename = os.path.join(output_dir, f'{strategy_name}_combined_metrics.pdf')
    plt.savefig(filename)
    plt.close()
    print(f"Combined plot saved to {filename}")

def run_evaluation():
    log.info("--- Starting Multi-Drone Path Evaluation and Visualization ---")
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
        
        # 4. Evaluate and Visualize Each Path
        for name, generator in path_generators.items():
            log.info(f"--- Evaluating '{name}' Path ---")
            generated_paths = generator(center_x, center_y, max_radius_m, item)
            
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
            output_file_left = os.path.join(GRAPHS_DIR, f"path_{name.lower()}_features_clipped.pdf")
            output_file_right = os.path.join(GRAPHS_DIR, f"path_{name.lower()}_heatmap.pdf")
            output_file_side = os.path.join(GRAPHS_DIR, f"path_{name.lower()}_side_by_side_clipped.pdf")

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

            # plot_combined_metrics(
            #     all_metrics['combined_cumulative_likelihood'],
            #     all_metrics['combined_cumulative_victims'],
            #     output_dir='graphs/plots',
            #     strategy_name= name.lower()
            # )

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    log.info("--- Starting Multi-Drone Path Evaluation and Visualization ---")
    run_evaluation()


    log.info("--- Starting Dataset Evaluation ---")
    evaluator = metrics.DatasetEvaluator(
        dataset_dirs=DATASET_DIRS,
        path_generators=PATH_GENERATORS,
        num_victims=NUM_VICTIMS,
        evaluation_size=EVALUATION_SIZE,
        fov_degrees=FOV_DEGREES,
        altitude_meters=ALTITUDE_METERS,
        overlap_ratio=OVERLAP_RATIO,
        num_drones=NUM_DRONES,
        path_point_spacing_m=PATH_POINT_SPACING_M,
        transition_distance_m=TRANSITION_DISTANCE_M,
        pizza_border_gap_m=PIZZA_BORDER_GAP_M,
        discount_factor=DISCOUNT_FACTOR
    )
    evaluator.evaluate()

    # log.info("--- Initializing the Search and Rescue Toolkit ---")
    # data_dir = "sarenv_dataset/1"  # Path to the dataset directory

    # # 1. Initialize the evaluator
    # evaluator = ComparativeEvaluator(
    #     dataset_directory=data_dir,
    #     evaluation_sizes=["large"], # Use a single size for a quick test
    #     num_drones=5,
    #     num_lost_persons=100,
    # )

    # # 2. Run the evaluations
    # baseline_results = evaluator.run_baseline_evaluations()

    # # 3. Plot the results from the baseline run
    # if baseline_results is not None and not baseline_results.empty:
    #     evaluator.plot_results(baseline_results)