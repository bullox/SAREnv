# examples/04_evaluate_coverage_paths.py
import os
import geopandas as gpd
from shapely.geometry import Point
import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo, plot
from sarenv.analytics.evaluator import ComparativeEvaluator
log = sarenv.get_logger()

# --- Parameters ---
GRAPHS_DIR = "graphs"
DATASET_DIR = "sarenv_dataset"
EVALUATION_SIZE = "xlarge"
NUM_DRONES = 5
NUM_VICTIMS = 5

# Drone/Path Parameters
FOV_DEGREES = 45.0
ALTITUDE_METERS = 80.0
OVERLAP_RATIO = 0.25
PATH_POINT_SPACING_M = 10.0
TRANSITION_DISTANCE_M = 50.0
PIZZA_BORDER_GAP_M = 15.0

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

        path_generators = {
            "Greedy": lambda: paths.generate_greedy_path(center_x, center_y, item.heatmap, item.bounds),
            "Spiral": lambda: paths.generate_spiral_path(center_x, center_y, max_radius_m, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M),
            "Concentric": lambda: paths.generate_concentric_circles_path(center_x, center_y, max_radius_m, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M, TRANSITION_DISTANCE_M),
            "Pizza": lambda: paths.generate_pizza_zigzag_path(center_x, center_y, max_radius_m, NUM_DRONES, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, PATH_POINT_SPACING_M, PIZZA_BORDER_GAP_M)
        }
        
        colors = {"Spiral": ['#00FFFF'], "Concentric": ['#FF00FF'], "Pizza": ['#33FF57']}

        # 4. Evaluate and Visualize Each Path
        for name, generator in path_generators.items():
            log.info(f"--- Evaluating '{name}' Path ---")
            generated_paths = generator()
            
            likelihood = evaluator.calculate_likelihood_score(generated_paths)
            discounted = evaluator.calculate_time_discounted_score(generated_paths)
            victim_score = evaluator.calculate_victims_found_score(generated_paths)

            print(f"  Likelihood Score: {likelihood:.2f}")
            print(f"  Time-Discounted Score: {discounted:.2f}")
            print(f"  Victims Found: {victim_score['percentage_found']:.2f}%")
            print(f"  Detection Timeliness: {victim_score['detection_timeliness']:.3f}")

            output_file = os.path.join(GRAPHS_DIR, f"path_{name.lower()}.html")
            plot.visualize_path_plotly(item, name, generated_paths, colors[name], victims_gdf, data_crs, output_file)
            log.info(f"Saved {name} visualization to {output_file}")

        heatmap_output = os.path.join(GRAPHS_DIR, "heatmap.html")
        plot.visualize_heatmap_plotly(item, heatmap_output)
        log.info(f"Saved heatmap visualization to {heatmap_output}")

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    # run_evaluation()

    log.info("--- Initializing the Search and Rescue Toolkit ---")
    data_dir = "sarenv_dataset"  # Path to the dataset directory

    # 1. Initialize the evaluator
    evaluator = ComparativeEvaluator(
        dataset_directory=data_dir,
        evaluation_sizes=["large"], # Use a single size for a quick test
        num_drones=10,
        num_lost_persons=100,
    )

    # 2. Run the evaluations
    baseline_results = evaluator.run_baseline_evaluations()

    # 3. Plot the results from the baseline run
    if baseline_results is not None and not baseline_results.empty:
        evaluator.plot_results(baseline_results)
