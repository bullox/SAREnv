# main.py
from toolkit import ComparativeEvaluator, log

# 1. Initialize the evaluator.
# This step prepares the toolkit with your desired configuration.
evaluator = ComparativeEvaluator(
    evaluation_sizes=["small", "medium"], # Optional: specify sizes
    num_drones=5,
    num_victims=10
)

# 2. Load data.
# This single line handles finding, loading, and preparing all datasets.
evaluator.load_datasets()

# 3. Run all baseline algorithms across all loaded datasets.
# This single line runs the entire simulation for the baseline algorithms.
baseline_results = evaluator.run_baseline_evaluation()

# 4. Plot the results.
# This generates and saves PNG images of the comparison charts.
evaluator.plot_results(baseline_results)

log.info("Evaluation complete. Check the 'graphs' folder for plots.")