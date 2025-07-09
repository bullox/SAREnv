# examples/05_evaluate_all_datasets.py
import os
import pandas as pd
import numpy as np
import sarenv
from sarenv.analytics.evaluator import ComparativeEvaluator

log = sarenv.get_logger()

# --- Parameters ---
DATASET_BASE_DIR = "sarenv_dataset"
EVALUATION_SIZES = ["small", "medium", "large"]
NUM_DRONES = 5
NUM_VICTIMS = 100
MAX_DATASETS = None  # Limit for testing - set to None to process all


def aggregate_results_with_stats(all_results: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregates results from multiple dataset runs and calculates mean ± std.

    Args:
        all_results: List of DataFrames, each containing results from one dataset

    Returns:
        DataFrame with aggregated statistics
    """
    if not all_results:
        log.error("No results to aggregate")
        return pd.DataFrame()

    # Concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Group by Algorithm and Dataset to calculate statistics
    grouped = combined_df.groupby(["Algorithm", "Dataset"])

    # Calculate mean and std for each metric
    stats_list = []
    for (algorithm, dataset), group in grouped:
        stats = {
            "Algorithm": algorithm,
            "Dataset": dataset,
        }

        # For each numeric metric, calculate mean ± std
        numeric_columns = [
            "Likelihood Score",
            "Time-Discounted Score",
            "Victims Found (%)",
            "Area Covered (km²)",
            "Total Path Length (km)",
        ]

        for col in numeric_columns:
            if col in group.columns:
                mean_val = group[col].mean()
                std_val = group[col].std()
                stats[f"{col} (Mean)"] = mean_val
                stats[f"{col} (Std)"] = std_val
                stats[f"{col} (Mean ± Std)"] = f"{mean_val:.2f} ± {std_val:.2f}"

        stats["Sample Size"] = len(group)
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def run_comprehensive_evaluation():
    """
    Runs ComparativeEvaluator on all datasets in the sarenv_dataset folder
    and aggregates results with statistics.
    """
    log.info("--- Starting Comprehensive Dataset Evaluation ---")

    # Get all dataset directories
    if not os.path.exists(DATASET_BASE_DIR):
        log.error(
            f"Dataset directory '{DATASET_BASE_DIR}' not found. Please run dataset generation first."
        )
        return

    dataset_dirs = [
        d
        for d in os.listdir(DATASET_BASE_DIR)
        if os.path.isdir(os.path.join(DATASET_BASE_DIR, d)) and d.isdigit()
    ]
    dataset_dirs.sort(key=int)  # Sort numerically

    # Limit datasets for testing
    if MAX_DATASETS is not None:
        dataset_dirs = dataset_dirs[:MAX_DATASETS]

    if not dataset_dirs:
        log.error(f"No dataset subdirectories found in '{DATASET_BASE_DIR}'")
        return

    log.info(f"Found {len(dataset_dirs)} datasets: {dataset_dirs}")

    all_results = []
    successful_runs = 0

    # Run evaluation on each dataset
    for dataset_id in dataset_dirs:
        dataset_path = os.path.join(DATASET_BASE_DIR, dataset_id)
        log.info(f"\n--- Evaluating Dataset {dataset_id} ---")

        try:
            # Initialize evaluator for this dataset
            evaluator = ComparativeEvaluator(
                dataset_directory=dataset_path,
                evaluation_sizes=EVALUATION_SIZES,
                num_drones=NUM_DRONES,
                num_lost_persons=NUM_VICTIMS,
            )

            # Run baseline evaluations
            results = evaluator.run_baseline_evaluations()

            if results is not None and not results.empty:
                # Add dataset ID for tracking
                results["Dataset_ID"] = dataset_id
                all_results.append(results)
                successful_runs += 1
                log.info(f"Successfully evaluated dataset {dataset_id}")
            else:
                log.warning(f"No results obtained for dataset {dataset_id}")

        except Exception as e:
            log.error(f"Error evaluating dataset {dataset_id}: {e}", exc_info=True)
            continue

    if not all_results:
        log.error("No successful evaluations completed")
        return

    log.info(f"\n--- Aggregating Results from {successful_runs} Datasets ---")

    # Aggregate results with statistics
    aggregated_stats = aggregate_results_with_stats(all_results)

    # Display results
    log.info("\n=== COMPREHENSIVE EVALUATION RESULTS ===")
    print("\nAggregated Results (Mean ± Standard Deviation):")
    print("=" * 80)

    # Print summary table
    for size in EVALUATION_SIZES:
        size_data = aggregated_stats[aggregated_stats["Dataset"] == size]
        if not size_data.empty:
            print(f"\n{size.upper()} Dataset Results:")
            print("-" * 40)
            for _, row in size_data.iterrows():
                print(f"\nAlgorithm: {row['Algorithm']}")
                print(f"  Sample Size: {row['Sample Size']}")
                if "Likelihood Score (Mean ± Std)" in row:
                    print(f"  Likelihood Score: {row['Likelihood Score (Mean ± Std)']}")
                if "Time-Discounted Score (Mean ± Std)" in row:
                    print(
                        f"  Time-Discounted Score: {row['Time-Discounted Score (Mean ± Std)']}"
                    )
                if "Victims Found (%) (Mean ± Std)" in row:
                    print(
                        f"  Victims Found (%): {row['Victims Found (%) (Mean ± Std)']}"
                    )
                if "Area Covered (km²) (Mean ± Std)" in row:
                    print(
                        f"  Area Covered (km²): {row['Area Covered (km²) (Mean ± Std)']}"
                    )
                if "Total Path Length (km) (Mean ± Std)" in row:
                    print(
                        f"  Total Path Length (km): {row['Total Path Length (km) (Mean ± Std)']}"
                    )

    # Save detailed results
    output_dir = "graphs"
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregated statistics
    stats_file = os.path.join(output_dir, "comprehensive_evaluation_stats.csv")
    aggregated_stats.to_csv(stats_file, index=False)
    log.info(f"Saved aggregated statistics to: {stats_file}")

    # Save raw combined results
    if all_results:
        combined_raw = pd.concat(all_results, ignore_index=True)
        raw_file = os.path.join(output_dir, "comprehensive_evaluation_raw.csv")
        combined_raw.to_csv(raw_file, index=False)
        log.info(f"Saved raw results to: {raw_file}")

    # Create summary table for each algorithm across all dataset sizes
    print("\n" + "=" * 80)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 80)

    algorithms = aggregated_stats["Algorithm"].unique()
    for algorithm in algorithms:
        algo_data = aggregated_stats[aggregated_stats["Algorithm"] == algorithm]
        print(f"\n{algorithm}:")
        print("-" * len(algorithm))
        for _, row in algo_data.iterrows():
            print(
                f"  {row['Dataset']:8} | "
                + f"Likelihood: {row.get('Likelihood Score (Mean ± Std)', 'N/A'):15} | "
                + f"Victims: {row.get('Victims Found (%) (Mean ± Std)', 'N/A'):15}"
            )

    log.info("\n--- Comprehensive Evaluation Complete ---")
    return aggregated_stats, all_results


if __name__ == "__main__":
    results = run_comprehensive_evaluation()
