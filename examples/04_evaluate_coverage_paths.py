# examples/04_evaluate_coverage_paths.py
import os
import geopandas as gpd
from shapely.geometry import Point
import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo, plot
from sarenv.analytics.evaluator import ComparativeEvaluator
log = sarenv.get_logger()

if __name__ == "__main__":
    log.info("--- Initializing the Search and Rescue Toolkit ---")
    data_dir = "sarenv_dataset/1"  # Path to the dataset directory
    # 1. Initialize the evaluator
    evaluator = ComparativeEvaluator(
        dataset_directory=data_dir,
        evaluation_sizes=["medium"], # Use a single size for a quick test
        num_drones=5,
        num_lost_persons=100,
    )

    # 2. Run the evaluations
    baseline_results = evaluator.run_baseline_evaluations()

    # 3. Plot the results from the baseline run
    if baseline_results is not None and not baseline_results.empty:
        evaluator.plot_results(baseline_results)
