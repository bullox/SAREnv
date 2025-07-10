# examples/05_evaluate_all_datasets.py
import os
import pandas as pd
import numpy as np
import sarenv
from sarenv.analytics.evaluator import ComparativeEvaluator, ComparativeDatasetEvaluator, PathGenerator, PathGeneratorConfig
from sarenv.analytics import paths

log = sarenv.get_logger()

# --- Parameters ---
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
DATASET_DIRS = [f"sarenv_dataset/{i}" for i in range(1,5)]


def create_custom_path_generator():
    """
    Example of how to create a custom path generator.
    This is a simple example that creates a straight line path.
    """
    def custom_straight_line_path(center_x, center_y, max_radius, **kwargs):
        """Custom path generator that creates a straight line."""
        num_drones = kwargs.get('num_drones', 3)
        path_point_spacing_m = kwargs.get('path_point_spacing_m', 10.0)
        
        # Create a simple straight line path
        num_points = int(max_radius * 2 / path_point_spacing_m)
        x_coords = np.linspace(center_x - max_radius, center_x + max_radius, num_points)
        y_coords = np.full_like(x_coords, center_y)
        
        from shapely.geometry import LineString
        full_path = LineString(zip(x_coords, y_coords))
        
        # Split for multiple drones
        return paths.split_path_for_drones(full_path, num_drones)
    
    return PathGenerator(
        name="CustomStraightLine",
        func=custom_straight_line_path,
        description="Custom straight line path generator"
    )


if __name__ == "__main__":
    log.info("--- Starting Comparative Dataset Evaluator ---")

    # Example 1: Using default path generators
    evaluator_default = ComparativeDatasetEvaluator(
        dataset_dirs=DATASET_DIRS, budget=200_000, num_drones=5,evaluation_size="medium" # Budget in meters
    )

    # # Example 2: Using custom path generators
    # custom_generators = {
    #     "Spiral": PathGenerator("Spiral", paths.generate_spiral_path),
    #     "CustomLine": create_custom_path_generator(),
    #     "Greedy": PathGenerator("Greedy", paths.generate_greedy_path),
    # }

    # evaluator_custom = ComparativeDatasetEvaluator(
    #     dataset_dirs=DATASET_DIRS,
    # )

    # Choose which evaluator to use
    evaluator = evaluator_default  # Switch to evaluator_custom to test custom generators

    log.info(f"Using path generators: {list(evaluator.path_generators.keys())}")

    # 2. Run the evaluations
    baseline_results = evaluator.evaluate()

    # 3. Show summary of results
    per_dataset_results_df = evaluator.get_results_per_dataset()
    summarized_results_df = evaluator.summarize_results()
    log.info("--- Summary of Results ---")
    print(summarized_results_df)

    # 4. Save results to CSV
    os.makedirs("graphs/comparative_plots", exist_ok=True)
    per_dataset_results_csv_path = os.path.join("graphs/comparative_plots", f'per_dataset_comparative_evaluation_results_{EVALUATION_SIZE}.csv')
    summarized_results_csv_path = os.path.join("graphs/comparative_plots", f'summarized_comparative_evaluation_results_{EVALUATION_SIZE}.csv')
    per_dataset_results_df.to_csv(per_dataset_results_csv_path, index=False)
    summarized_results_df.to_csv(summarized_results_csv_path, index=False)
    log.info(f"Results saved to {per_dataset_results_csv_path} and {summarized_results_csv_path}")

    # 5. Generate comparative plots
    evaluator.plot_combined_normalized_bars(output_dir="graphs/comparative_plots")
    evaluator.plot_time_series_with_ci(output_dir="graphs/comparative_plots")
    evaluator.plot_combined_time_series_with_ci(output_dir="graphs/comparative_plots")
    log.info("--- Finished Comparative Dataset Evaluator ---")