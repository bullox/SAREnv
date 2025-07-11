# examples/05_evaluate_all_datasets.py
import numpy as np
import pandas as pd
from pathlib import Path
import sarenv
from sarenv.analytics.evaluator import ComparativeDatasetEvaluator, PathGenerator
from sarenv.analytics import paths

log = sarenv.get_logger()


def create_custom_path_generator():
    """
    Example of how to create a custom path generator.
    This is a simple example that creates a straight line path.
    """

    def custom_straight_line_path(center_x, center_y, max_radius, **kwargs):
        """Custom path generator that creates a straight line."""
        num_drones = kwargs.get("num_drones", 3)
        path_point_spacing_m = kwargs.get("path_point_spacing_m", 10.0)

        # Create a simple straight line path
        num_points = int(max_radius * 2 / path_point_spacing_m)
        x_coords = np.linspace(center_x - max_radius, center_x + max_radius, num_points)
        y_coords = np.full_like(x_coords, center_y)

        from shapely.geometry import LineString

        full_path = LineString(zip(x_coords, y_coords, strict=True))

        # Split for multiple drones
        return paths.split_path_for_drones(full_path, num_drones)

    return PathGenerator(
        name="CustomStraightLine",
        func=custom_straight_line_path,
        description="Custom straight line path generator",
    )


if __name__ == "__main__":
    log.info("--- Starting Comparative Dataset Evaluator ---")
    evaluation_size = "small"  # Options: "small", "medium", "large", "xlarge"

    # Example 1: Using default path generators
    evaluator = ComparativeDatasetEvaluator(
        dataset_dirs=[f"sarenv_dataset/{i}" for i in range(1, 5)],
        budget=100_000,
        num_drones=1,
        evaluation_size=evaluation_size,  # Budget in meters
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

    log.info(f"Using path generators: {list(evaluator.path_generators.keys())}")

    # 2. Run the evaluations
    metrics_df, time_series_df = evaluator.evaluate(output_dir="results")

    # 3. Show summary of results
    per_dataset_results_df = evaluator.get_results_per_dataset()
    summarized_results_df = evaluator.summarize_results()
    log.info("--- Summary of Results ---")
    log.info(str(summarized_results_df))

    log.info("--- Finished Comparative Dataset Evaluator ---")
