# toolkit.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo
from shapely.geometry import Point
from sarenv.utils.logging_setup import get_logger
from scipy import stats


log = get_logger()


class ComparativeDatasetEvaluator:
    def __init__(
        self,
        dataset_dirs,
        path_generators,
        num_victims,
        evaluation_size,
        fov_degrees,
        altitude_meters,
        overlap_ratio,
        num_drones,
        path_point_spacing_m,
        transition_distance_m,
        pizza_border_gap_m,
        discount_factor
    ):
        self.dataset_dirs = dataset_dirs
        self.path_generators = path_generators
        self.num_victims = num_victims
        self.evaluation_size = evaluation_size
        self.fov_degrees = fov_degrees
        self.altitude_meters = altitude_meters
        self.overlap_ratio = overlap_ratio
        self.num_drones = num_drones
        self.path_point_spacing_m = path_point_spacing_m
        self.transition_distance_m = transition_distance_m
        self.pizza_border_gap_m = pizza_border_gap_m
        self.discount_factor = discount_factor

        # Storage for summary and time-series results
        self.summary_results = []
        self.time_series_results = {name: [] for name in self.path_generators}

    def evaluate(self):
        for dataset_dir in self.dataset_dirs:
            loader = sarenv.DatasetLoader(dataset_directory=dataset_dir)
            item = loader.load_environment(self.evaluation_size)
            if not item:
                continue

            data_crs = geo.get_utm_epsg(item.center_point[0], item.center_point[1])
            victim_generator = sarenv.LostPersonLocationGenerator(item)
            victim_points = [
                p for p in (victim_generator.generate_location() for _ in range(self.num_victims)) if p
            ]
            if victim_points:
                victims_gdf = gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
            else:
                victims_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=data_crs)

            evaluator = metrics.PathEvaluator(
                item.heatmap,
                item.bounds,
                victims_gdf,
                self.fov_degrees,
                self.altitude_meters,
                loader._meter_per_bin
            )

            center_proj = gpd.GeoDataFrame(
                geometry=[Point(item.center_point)], crs="EPSG:4326"
            ).to_crs(data_crs).geometry.iloc[0]
            center_x, center_y = center_proj.x, center_proj.y
            max_radius_m = item.radius_km * 1000

            for name, generator in self.path_generators.items():
                paths = generator(center_x, center_y, max_radius_m, item, None)
                results = evaluator.calculate_all_metrics(paths, self.discount_factor)

                # Store summary results
                victim_metrics = results['victim_detection_metrics']
                self.summary_results.append({
                    "Algorithm": name,
                    "Dataset": os.path.basename(dataset_dir),
                    "Likelihood Score": results['total_likelihood_score'],
                    "Time-Discounted Score": results['total_time_discounted_score'],
                    "Victims Found (%)": victim_metrics['percentage_found'],
                    "Avg. Detection Distance (m)": victim_metrics['average_detection_distance'],
                })

                # Store time-series results for CI plots
                self.time_series_results[name].append({
                    'combined_cumulative_likelihood': results['combined_cumulative_likelihood'],
                    'combined_cumulative_victims': results['combined_cumulative_victims'],
                })

    def get_results_per_dataset(self):
        """
        Returns the results as a DataFrame grouped by dataset.
        """
        return pd.DataFrame(self.summary_results).groupby("Dataset").apply(lambda x: x.reset_index(drop=True))

    def summarize_results(self):
        """
        Returns a summary DataFrame grouped by algorithm with means and 95% confidence intervals.
        """
        df = pd.DataFrame(self.summary_results)
        self.summary = df.groupby("Algorithm").agg(
            Mean_Likelihood_Score=('Likelihood Score', 'mean'),
            CI_Likelihood_Score=('Likelihood Score', lambda x: 1.96 * x.sem()),
            Mean_Time_Discounted=('Time-Discounted Score', 'mean'),
            CI_Time_Discounted=('Time-Discounted Score', lambda x: 1.96 * x.sem()),
            Mean_Victims_Found=('Victims Found (%)', 'mean'),
            CI_Victims_Found=('Victims Found (%)', lambda x: 1.96 * x.sem()),
            Mean_Detection_Dist=('Avg. Detection Distance (m)', 'mean'),
            CI_Detection_Dist=('Avg. Detection Distance (m)', lambda x: 1.96 * x.sem()),
        ).reset_index()
        return self.summary

    def plot_aggregate_bars(self, output_dir="graphs/aggregate"):
        os.makedirs(output_dir, exist_ok=True)
        metrics = [
            ("Mean_Likelihood_Score", "CI_Likelihood_Score", "Likelihood Score"),
            ("Mean_Time_Discounted", "CI_Time_Discounted", "Time-Discounted Score"),
            ("Mean_Victims_Found", "CI_Victims_Found", "Victims Found (%)"),
            ("Mean_Detection_Dist", "CI_Detection_Dist", "Avg. Detection Distance (m)"),
        ]
        for mean_col, ci_col, label in metrics:
            plt.figure(figsize=(10, 6))
            plt.bar(self.summary["Algorithm"], self.summary[mean_col], yerr=self.summary[ci_col], capsize=5, alpha=0.7)
            plt.ylabel(label)
            plt.title(f"Algorithm Comparison: {label} (All Datasets)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"aggregate_{label.replace(' ','_').replace('(%)','').replace('(m)','').lower()}_{self.evaluation_size}.png"))
            plt.close()

    def plot_combined_normalized_bars(self, output_dir="graphs/aggregate"):
        """
        Creates a grouped bar chart comparing algorithms across normalized metrics (0 to 1 scale).
        Excludes Average Detection Distance.
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics = [
            ("Mean_Likelihood_Score", "CI_Likelihood_Score", "Likelihood Score"),
            ("Mean_Time_Discounted", "CI_Time_Discounted", "Time-Discounted Score"),
            ("Mean_Victims_Found", "CI_Victims_Found", "Victims Found Score"),
        ]
        algorithms = self.summary["Algorithm"].tolist()
        n_algorithms = len(algorithms)
        n_metrics = len(metrics)
        x = np.arange(n_metrics)
        width = 0.8 / n_algorithms

        # Normalize the mean values between 0 and 1 for each metric
        normalized_means = {}
        for metric in metrics:
            values = self.summary[metric[0]].values
            min_val = values.min()
            max_val = values.max()
            if max_val - min_val == 0:
                normalized_means[metric[0]] = np.ones_like(values)
            else:
                normalized_means[metric[0]] = (values - min_val) / (max_val - min_val)

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.get_cmap('tab10', n_algorithms)

        for i, alg in enumerate(algorithms):
            means = [normalized_means[metric[0]][self.summary["Algorithm"] == alg][0] for metric in metrics]
            cis = [self.summary.loc[self.summary["Algorithm"] == alg, metric[1]].values[0] for metric in metrics]
            positions = x - 0.4 + i * width + width / 2
            ax.bar(positions, means, width, yerr=cis, capsize=5, label=alg, color=colors(i), alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in metrics])
        ax.set_ylabel('Normalized Score (0 to 1)')
        ax.set_title('Algorithm Comparison Across Normalized Metrics (All Datasets)')
        ax.legend(title='Algorithm')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'aggregate_normalized_metrics_{self.evaluation_size}.png'))
        plt.close()

    def plot_time_series_with_ci(self, output_dir="graphs/plots"):
        """
        For each algorithm, plot mean and 95% CI for time-series metrics.
        """
        os.makedirs(output_dir, exist_ok=True)

        def mean_ci(arrays):
            max_len = max(len(a) for a in arrays)
            padded = [np.pad(a, (0, max_len - len(a)), mode='edge') if len(a) < max_len else a for a in arrays]
            data = np.vstack(padded)
            mean = np.mean(data, axis=0)
            sem = stats.sem(data, axis=0)
            h = sem * stats.t.ppf((1 + 0.95) / 2., data.shape[0] - 1)
            return mean, mean - h, mean + h

        for name, results_list in self.time_series_results.items():
            if not results_list:
                continue
            combined_likelihoods = [r['combined_cumulative_likelihood'] for r in results_list]
            combined_victims = [r['combined_cumulative_victims'] for r in results_list]

            mean_likelihood, ci_low_likelihood, ci_high_likelihood = mean_ci(combined_likelihoods)
            mean_victims, ci_low_victims, ci_high_victims = mean_ci(combined_victims)

            fig, ax1 = plt.subplots(figsize=(10, 6))

            color_likelihood = 'tab:blue'
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Average Accumulated Likelihood (%)', color=color_likelihood)
            ax1.plot(100 * mean_likelihood, color=color_likelihood, label='Avg Accumulated Likelihood')
            ax1.fill_between(
                range(len(mean_likelihood)),
                100 * ci_low_likelihood, 100 * ci_high_likelihood,
                color=color_likelihood, alpha=0.3
            )
            ax1.tick_params(axis='y', labelcolor=color_likelihood)

            ax2 = ax1.twinx()
            color_victims = 'tab:red'
            ax2.set_ylabel('Average Victims Found', color=color_victims)
            ax2.plot(mean_victims, color=color_victims, label='Avg Victims Found')
            ax2.fill_between(range(len(mean_victims)), ci_low_victims, ci_high_victims, color=color_victims, alpha=0.3)
            ax2.tick_params(axis='y', labelcolor=color_victims)

            plt.title(f'Average Combined Metrics with 95% CI for {name}')
            fig.tight_layout()
            filename = os.path.join(output_dir, f'{name}_{self.evaluation_size}_average_combined_metrics.pdf')
            plt.savefig(filename)
            plt.close()


class ComparativeEvaluator:
    """
    A toolkit for running and comparing UAV pathfinding algorithms.

    This class simplifies the process of evaluating multiple pathfinding
    strategies across different datasets and visualizing the results.
    """

    def __init__(
        self,
        dataset_directory="sarenv_dataset",
        evaluation_sizes=None,
        num_drones=3,
        num_lost_persons=100,
        custom_path_generators=None,  # SHOULD BE A DICT {name: function}
    ):
        """
        Initializes the ComparativeEvaluator.

        Args:
            dataset_directory (str): The path to the sarenv dataset.
            evaluation_sizes (list, optional): A list of dataset sizes to evaluate (e.g., ["small", "medium"]).
                                                Defaults to ["small", "medium", "large", "xlarge"].
            num_drones (int): The number of drones to simulate.
            num_victims (int): The number of victim locations to generate per dataset.
            custom_path_generators (dict, optional): Dict of {name: function} for custom path generators.
                                                        Each function must accept a single dict of path parameters.
        """
        self.dataset_directory = dataset_directory
        self.evaluation_sizes = evaluation_sizes or [
            "small",
            "medium",
            "large",
            "xlarge",
        ]
        self.num_drones = num_drones
        self.num_victims = num_lost_persons
        self.loader = sarenv.DatasetLoader(dataset_directory=self.dataset_directory)
        self.environments = {}
        self.results = None

        # Drone/Path Parameters
        self.path_params = {
            "fov_deg": 45.0,
            "altitude": 80.0,
            "overlap": 0.25,
            "path_point_spacing_m": 10.0,
            "transition_distance_m": 50.0,
            "border_gap_m": 15.0,
            "num_drones": self.num_drones,
        }

        # Register baseline algorithms
        self.baseline_generators = {
            "Random": lambda args: paths.generate_random_walk_path(**args),
            "Greedy": lambda args: paths.generate_greedy_path(**args),
            "Spiral": lambda args: paths.generate_spiral_path(**args),
            "Concentric": lambda args: paths.generate_concentric_circles_path(**args),
            "Pizza": lambda args: paths.generate_pizza_zigzag_path(**args),
        }

        # Add custom path generators if provided
        if custom_path_generators is not None:
            for name, func in custom_path_generators.items():
                if callable(func):
                    self.baseline_generators[name] = lambda args, f=func: f(args)
                else:
                    log.warning(f"Custom path generator '{name}' is not callable and was skipped.")

        self.load_datasets()

    def load_datasets(self):
        """
        Loads all specified datasets and generates static victim locations.
        This prepares the evaluator for running the simulations.
        """
        log.info(f"Loading datasets for sizes: {self.evaluation_sizes}")
        for size in self.evaluation_sizes:
            item = self.loader.load_environment(size)
            
            if not item:
                log.warning(f"Could not load data for size '{size}'. Skipping.")
                continue

            data_crs = geo.get_utm_epsg(item.center_point[0], item.center_point[1])
            victim_generator = sarenv.LostPersonLocationGenerator(item)
            victim_points = [
                p
                for p in (
                    victim_generator.generate_location()
                    for _ in range(self.num_victims)
                )
                if p
            ]
            victims_gdf = (
                gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
                if victim_points
                else gpd.GeoDataFrame(columns=["geometry"], crs=data_crs)
            )

            self.environments[size] = {
                "item": item,
                "victims": victims_gdf,
                "crs": data_crs,
            }
        log.info("All datasets loaded and prepared.")

    def run_baseline_evaluations(self) -> pd.DataFrame:
        """
        Runs all baseline algorithms across all loaded datasets.

        Returns:
            pd.DataFrame: A DataFrame containing the comparative results.
        """
        if not self.environments:
            log.error("No datasets loaded. Please call 'load_datasets()' first.")
            return pd.DataFrame()

        all_results = []
        for size, env_data in self.environments.items():
            item = env_data["item"]
            victims_gdf = env_data["victims"]

            log.info(f"--- Evaluating Baselines on '{size}' dataset ---")

            evaluator = metrics.PathEvaluator(
                item.heatmap,
                item.bounds,
                victims_gdf,
                self.path_params["fov_deg"],
                self.path_params["altitude"],
                self.loader._meter_per_bin
            )
            center_proj = (
                gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
                .to_crs(env_data["crs"])
                .geometry.iloc[0]
            )

            current_path_args = self.path_params.copy()
            current_path_args.update(
                {
                    "center_x": center_proj.x,
                    "center_y": center_proj.y,
                    "max_radius": item.radius_km * 1000,
                    "probability_map": item.heatmap,
                    "bounds": item.bounds
                }
            )

            for name, generator in self.baseline_generators.items():
                log.info(f"Running {name} algorithm on '{size}' dataset...")
                generated_paths = generator(current_path_args)

                # # Find the maximum length (number of waypoints) among all generated paths
                # t_end = max(len(path) for path in generated_paths) if generated_paths else 0
                # discount_factor = (0.05) ** (1 / t_end) if t_end > 0 else 0.999

                all_metrics = evaluator.calculate_all_metrics(
                    generated_paths,
                    0.999# discount_factor
                )
                
                # Extract results from the returned dictionary
                victim_metrics = all_metrics['victim_detection_metrics']
                
                result = {
                    "Dataset": size,
                    "Algorithm": name,
                    "Likelihood Score": all_metrics['total_likelihood_score'],
                    "Time-Discounted Score": all_metrics['total_time_discounted_score'],
                    "Victims Found (%)": victim_metrics['percentage_found'],
                    "Avg. Detection Distance (m)": victim_metrics['average_detection_distance'],
                }
                
                all_results.append(result)

        self.results = pd.DataFrame(all_results)
        log.info("--- Baseline Evaluation Complete ---")
        print(self.results.to_string())
        return self.results

    def plot_results(self, results_df: pd.DataFrame = None, output_dir="graphs"):
        """
        Generates and saves plots for the evaluation results.

        Args:
            results_df (pd.DataFrame, optional): The dataframe of results to plot.
                                                 If None, uses the last stored results.
            output_dir (str): The directory to save the plots in.
        """
        if results_df is None:
            results_df = self.results

        if results_df is None or results_df.empty:
            log.error("No results to plot. Please run an evaluation first.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        metrics_to_plot = [
            "Likelihood Score",
            "Time-Discounted Score",
            "Victims Found (%)",
            "Avg. Detection Distance (m)",
        ]

        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 7))
            sns.barplot(
                data=results_df,
                x="Dataset",
                y=metric,
                hue="Algorithm",
                order=self.evaluation_sizes,
            )
            plt.title(f"Comparison of Algorithms: {metric}", fontsize=16)
            plt.ylabel(metric)
            plt.xlabel("Dataset Size")
            plt.legend(title="Algorithm")
            plt.tight_layout()

            plot_filename = os.path.join(
                output_dir, f"plot_{metric.replace(' ', '_').replace('(%)','').replace('(m)','').lower()}.png"
            )
            plt.savefig(plot_filename)
            log.info(f"Saved plot to {plot_filename}")
            plt.close()


if __name__ == "__main__":
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
        log.info("--- Toolkit execution finished successfully! ---")