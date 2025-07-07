# toolkit.py
import os
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo
from shapely.geometry import Point
from sarenv.utils.logging_setup import get_logger

log = get_logger()


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
        num_missing_persons=100,
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
        self.num_victims = num_missing_persons
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
            "discount_factor": 0.999 # Default discount factor for time-discounted score
        }

        # Register baseline algorithms
        self.baseline_generators = {
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
                }
            )

            for name, generator in self.baseline_generators.items():
                log.info(f"Running {name} algorithm on '{size}' dataset...")
                generated_paths = generator(current_path_args)

                # Call the single, optimized function to get all metrics
                all_metrics = evaluator.calculate_all_metrics(
                    generated_paths,
                    discount_factor=self.path_params["discount_factor"]
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
        num_missing_persons=100,
    )

    # 2. Run the evaluations
    baseline_results = evaluator.run_baseline_evaluations()

    # 3. Plot the results from the baseline run
    if baseline_results is not None and not baseline_results.empty:
        evaluator.plot_results(baseline_results)
        log.info("--- Toolkit execution finished successfully! ---")