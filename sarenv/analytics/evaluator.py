# toolkit.py
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString

import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo
from sarenv.utils.logging_setup import get_logger
from sarenv.utils.plot import (
    plot_aggregate_bars,
    plot_combined_normalized_bars,
    plot_time_series_with_ci,
    plot_combined_time_series_with_ci,
    plot_single_evaluation_results
)

log = get_logger()

class PathGeneratorConfig:
    """
    Configuration class for path generation parameters.
    Provides a clean interface for managing path generation parameters.
    """

    def __init__(self, **kwargs):
        """
        Initialize path generation configuration.

        Args:
            **kwargs: Keyword arguments for path generation parameters.
                num_drones (int): Number of drones to simulate. Defaults to 3.
                fov_degrees (float): Field of view in degrees. Defaults to 45.0.
                altitude_meters (float): Altitude in meters. Defaults to 80.0.
                overlap_ratio (float): Overlap ratio for systematic patterns. Defaults to 0.25.
                path_point_spacing_m (float): Spacing between path points in meters. Defaults to 10.0.
                transition_distance_m (float): Transition distance for concentric patterns. Defaults to 50.0.
                pizza_border_gap_m (float): Border gap for pizza patterns. Defaults to 15.0.
        """
        self.num_drones = kwargs.pop('num_drones', 3)
        self.fov_degrees = kwargs.pop('fov_degrees', 45.0)
        self.altitude_meters = kwargs.pop('altitude_meters', 80.0)
        self.overlap_ratio = kwargs.pop('overlap_ratio', 0)
        self.path_point_spacing_m = kwargs.pop('path_point_spacing_m', 10.0)
        self.transition_distance_m = kwargs.pop('transition_distance_m', 50.0)
        self.pizza_border_gap_m = kwargs.pop('pizza_border_gap_m', 15.0)

        # Store any additional parameters not explicitly defined
        self.additional_params = kwargs

    def get_params_dict(
        self,
        center_x: float,
        center_y: float,
        max_radius: float,
        probability_map: np.ndarray | None,
        bounds: tuple[float, float, float, float] | None,
        budget: float | None = None
    ) -> dict[str, float | int | np.ndarray | tuple | None]:
        """
        Generate a complete parameter dictionary for path generation.

        Args:
            center_x: X coordinate of the center point
            center_y: Y coordinate of the center point
            max_radius: Maximum search radius
            probability_map: Optional probability map for informed search
            bounds: Optional geographic bounds tuple
            budget: Optional budget constraint

        Returns:
            Dictionary with all parameters needed for path generation
        """
        params = {
            'center_x': center_x,
            'center_y': center_y,
            'max_radius': max_radius,
            'probability_map': probability_map,
            'bounds': bounds,
            'num_drones': self.num_drones,
            'fov_deg': self.fov_degrees,
            'altitude': self.altitude_meters,
            'overlap': self.overlap_ratio,
            'path_point_spacing_m': self.path_point_spacing_m,
            'transition_distance_m': self.transition_distance_m,
            'border_gap_m': self.pizza_border_gap_m,
        }

        if budget is not None:
            params['budget'] = budget

        # Add any additional parameters
        params.update(self.additional_params)

        return params

class PathGenerator:
    """
    Wrapper class for path generation functions that provides a consistent interface.
    Uses PathGeneratorConfig to ensure consistent parameter handling.
    """

    def __init__(self, name: str, func, path_generator_config: PathGeneratorConfig, description: str = ""):
        """
        Initialize a path generator.

        Args:
            name: Name of the generator
            func: Function that generates paths
            description: Optional description of the generator
        """
        self.name = name
        self.func = func
        self.description = description
        self.path_generator_config = path_generator_config

    def __call__(
        self,
        center_x: float,
        center_y: float,
        max_radius: float,
        probability_map: np.ndarray | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> list[LineString]:
        """
        Call the generator function with properly configured parameters.

        Args:
            center_x: X coordinate of the center point
            center_y: Y coordinate of the center point
            max_radius: Maximum radius for the search area
            probability_map: Optional probability map for informed search
            bounds: Optional geographic bounds tuple
            budget: Optional budget constraint in meters

        Returns:
            List of LineString objects representing paths for each drone
        """
        # Get all parameters from the config
        params = self.path_generator_config.get_params_dict(
            center_x=center_x,
            center_y=center_y,
            max_radius=max_radius,
            probability_map=probability_map,
            bounds=bounds,
        )

        # All parameters must be passed to ensure consistent behavior
        return self.func(**params)


def get_default_path_generators(config: PathGeneratorConfig) -> dict[str, PathGenerator]:
    """
    Get default path generators with consistent parameter handling.
    
    Args:
        config: Configuration object containing default parameters for path generation
        
    Returns:
        Dictionary of PathGenerator instances
    """
    return {
        "RandomWalk": PathGenerator(
            name="RandomWalk",
            func=paths.generate_random_walk_path,
            path_generator_config=config,
            description="Random walk path generation"
        ),
        "Greedy": PathGenerator(
            name="Greedy",
            func=paths.generate_greedy_path,
            path_generator_config=config,
            description="Greedy path generation based on probability map"
        ),
        "Spiral": PathGenerator(
            name="Spiral",
            func=paths.generate_spiral_path,
            path_generator_config=config,
            description="Spiral path generation"
        ),
        "Concentric": PathGenerator(
            name="Concentric",
            func=paths.generate_concentric_circles_path,
            path_generator_config=config,
            description="Concentric circles path generation"
        ),
        "Pizza": PathGenerator(
            name="Pizza",
            func=paths.generate_pizza_zigzag_path,
            path_generator_config=config,
            description="Pizza slice zigzag path generation"
            
        )
    }


class ComparativeDatasetEvaluator:
    """
    Evaluates multiple datasets using ComparativeEvaluator to minimize code duplication.
    Provides time-series plotting and confidence interval analysis.
    """

    def __init__(self, dataset_dirs, **kwargs):
        """
        Initializes the ComparativeDatasetEvaluator.

        Args:
            dataset_dirs (list): A list of directories for the datasets to be evaluated. \\
            **kwargs: Keyword arguments for evaluation and path generation. \\
                path_generators (dict): Dict of {name: PathGenerator} instances. \\
                num_victims (int): Number of victim locations per dataset. Defaults to 100. \\
                evaluation_size (str): Dataset size to evaluate. Defaults to "medium". \\
                discount_factor (float): Discount factor for time-based scores. Defaults to 0.999. \\
                num_drones (int): Number of drones to simulate. Defaults to 3.\\
                fov_degrees (float): Camera field of view in degrees. Defaults to 45.0.\\
                altitude_meters (float): Drone altitude in meters. Defaults to 80.0.\\
                overlap_ratio (float): Overlap ratio for patterns. Defaults to 0.25.\\
                path_point_spacing_m (float): Spacing between path points. Defaults to 10.0.\\
                transition_distance_m (float): Transition distance for concentric patterns. Defaults to 50.0.\\
                pizza_border_gap_m (float): Border gap for pizza patterns. Defaults to 15.0.\\
        """
        self.dataset_dirs = dataset_dirs

        # Extract evaluation-specific parameters from kwargs
        path_generators = kwargs.get("path_generators")
        self.num_victims = kwargs.get("num_victims", 100)
        self.evaluation_size = kwargs.get("evaluation_size", "medium")
        self.discount_factor = kwargs.get("discount_factor", 0.999)

        # Create path generator configuration from all kwargs
        self.path_generator_config = PathGeneratorConfig(**kwargs)
        self.dataset_dirs = dataset_dirs
        # Set up path generators
        if path_generators is None:
            self.path_generators = get_default_path_generators(self.path_generator_config)
        else:
            self.path_generators = {}
            for name, generator in path_generators.items():
                if isinstance(generator, PathGenerator):
                    self.path_generators[name] = generator
                else:
                    # Assume it's a function and wrap it
                    self.path_generators[name] = PathGenerator(
                        name=name,
                        func=generator,
                        path_generator_config=self.path_generator_config,
                        description=f"Custom generator: {name}"
                    )

        # Storage for summary and time-series results
        self.summary_results = []
        self.time_series_results = {name: [] for name in self.path_generators}

    def evaluate(self):
        """
        Evaluates all path generators across all datasets using ComparativeEvaluator instances.
        Now only uses data generated by ComparativeEvaluator - no direct path evaluation.
        """
        # Create a ComparativeEvaluator for each dataset
        self.evaluators = []
        for dataset_dir in self.dataset_dirs:
            # Pass necessary parameters to ComparativeEvaluator
            evaluator = ComparativeEvaluator(
                dataset_directory=dataset_dir,
                evaluation_sizes=[self.evaluation_size],
                num_lost_persons=self.num_victims,
                path_generator_config=self.path_generator_config,
                path_generators=self.path_generators,
                # Pass num_drones from the config explicitly
                num_drones=self.path_generator_config.num_drones
            )
            self.evaluators.append(evaluator)

            log.info(f"Evaluating dataset: {dataset_dir}")

            # Run evaluation for this dataset
            results_df = evaluator.run_baseline_evaluations()

            if results_df.empty:
                continue

            # Store summary results from ComparativeEvaluator
            for _, row in results_df.iterrows():
                self.summary_results.append({
                    "Algorithm": row["Algorithm"],
                    "Dataset": Path(dataset_dir).name,
                    "Likelihood Score": row["Likelihood Score"],
                    "Time-Discounted Score": row["Time-Discounted Score"],
                    "Victims Found (%)": row["Victims Found (%)"],
                    "Area Covered (km²)": row["Area Covered (km²)"],
                    "Total Path Length (km)": row["Total Path Length (km)"],
                })

            # Use time-series data generated by ComparativeEvaluator
            for algorithm_name, time_series_list in evaluator.time_series_data.items():
                if algorithm_name not in self.time_series_results:
                    self.time_series_results[algorithm_name] = []

                # Add the time-series data from this evaluator
                self.time_series_results[algorithm_name].extend(time_series_list)

    def get_results_per_dataset(self) -> pd.DataFrame:
        """
        Returns the results as a DataFrame grouped by dataset.
        """
        return pd.DataFrame(self.summary_results).groupby("Dataset").apply(lambda x: x.reset_index(drop=True))

    def summarize_results(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame grouped by algorithm with means and 95% confidence intervals.
        """
        results_df = pd.DataFrame(self.summary_results)
        self.summary = results_df.groupby("Algorithm").agg(
            Mean_Likelihood_Score=('Likelihood Score', 'mean'),
            CI_Likelihood_Score=('Likelihood Score', lambda x: 1.96 * x.sem()),
            Mean_Time_Discounted=('Time-Discounted Score', 'mean'),
            CI_Time_Discounted=('Time-Discounted Score', lambda x: 1.96 * x.sem()),
            Mean_Victims_Found=('Victims Found (%)', 'mean'),
            CI_Victims_Found=('Victims Found (%)', lambda x: 1.96 * x.sem()),
            Mean_Area_Covered=('Area Covered (km²)', 'mean'),
            CI_Area_Covered=('Area Covered (km²)', lambda x: 1.96 * x.sem()),
            Mean_Path_Length=('Total Path Length (km)', 'mean'),
            CI_Path_Length=('Total Path Length (km)', lambda x: 1.96 * x.sem()),
        ).reset_index()
        return self.summary

    def plot_aggregate_bars(self, output_dir="graphs/aggregate"):
        """Plot aggregate bar charts for each metric across all datasets."""
        plot_aggregate_bars(self.summary, self.evaluation_size, output_dir)

    def plot_combined_normalized_bars(self, output_dir="graphs/aggregate"):
        """
        Creates a grouped bar chart comparing algorithms across normalized metrics (0 to 1 scale).
        Excludes Average Detection Distance.
        """
        plot_combined_normalized_bars(self.summary, self.evaluation_size, output_dir)

    def plot_time_series_with_ci(self, output_dir="graphs/plots"):
        """
        For each algorithm, plot mean and 95% CI for time-series metrics.
        """
        plot_time_series_with_ci(self.time_series_results, self.evaluation_size, output_dir)

    def plot_combined_normalized_bars_with_ci(self, output_dir="graphs/aggregate"):
        """
        Creates a grouped bar chart comparing algorithms across normalized metrics (0 to 1 scale)
        with 95% confidence intervals.
        Excludes Average Detection Distance.
        """
        plot_combined_normalized_bars(self.summary, self.evaluation_size, output_dir)

    def plot_combined_time_series_with_ci(self, output_dir="graphs/plots"):
        """
        Plot all algorithms in a single figure with four subplots arranged vertically.
        Each subplot shows time-series metrics with 95% CI for one algorithm.
        """
        plot_combined_time_series_with_ci(self.time_series_results, self.evaluation_size, output_dir)


class ComparativeEvaluator:
    """
    A toolkit for running and comparing UAV pathfinding algorithms.

    This class simplifies the process of evaluating multiple pathfinding
    strategies across different datasets and visualizing the results.
    """

    def __init__(self, **kwargs):
        """
        Initializes the ComparativeEvaluator.

        Args:
            **kwargs: Keyword arguments for evaluation.
                dataset_directory (str): Path to the sarenv dataset. Defaults to "sarenv_dataset".
                evaluation_sizes (list): List of dataset sizes to evaluate. Defaults to ["small", "medium", "large", "xlarge"].
                num_drones (int): The number of drones to simulate. Defaults to 3.
                num_lost_persons (int): Number of victim locations to generate. Defaults to 100.
                path_generator_config (PathGeneratorConfig): Configuration for path parameters.
                path_generators (dict): Dict of {name: PathGenerator} instances.
                use_defaults (bool): Whether to use default path generators if none are provided. Defaults to True.
        """
        self.dataset_directory = kwargs.get("dataset_directory", "sarenv_dataset")
        self.evaluation_sizes = kwargs.get("evaluation_sizes") or [
            "small",
            "medium",
            "large",
            # "xlarge",
        ]
        self.num_victims = kwargs.get("num_lost_persons", 100)
        self.path_generator_config:PathGeneratorConfig = PathGeneratorConfig(**kwargs)
        self.num_drones = self.path_generator_config.num_drones if self.path_generator_config else kwargs.get("num_drones", 3)
        self.path_generators = kwargs.get("path_generators")
        use_defaults = kwargs.get("use_defaults", True)

        self.loader = sarenv.DatasetLoader(dataset_directory=self.dataset_directory)
        self.environments = {}
        self.results = None
        self.time_series_data = {}

        # Set up path generators, defaulting if none are provided
        if self.path_generators is None:
            if use_defaults:
                if self.path_generator_config:
                    self.path_generators = get_default_path_generators(self.path_generator_config)
                else:
                    config = PathGeneratorConfig(num_drones=self.num_drones)
                    self.path_generators = get_default_path_generators(config)
            else:
                self.path_generators = {}

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
        self.time_series_data = {}  # Reset time-series data for each evaluation

        for size, env_data in self.environments.items():
            item = env_data["item"]
            victims_gdf = env_data["victims"]

            log.info(f"--- Evaluating Baselines on '{size}' dataset ---")

            evaluator = metrics.PathEvaluator(
                item.heatmap,
                item.bounds,
                victims_gdf,
                self.path_generator_config.fov_degrees,
                self.path_generator_config.altitude_meters,
                self.loader._meter_per_bin
            )
            center_proj = (
                gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
                .to_crs(env_data["crs"])
                .geometry.iloc[0]
            )

            for name, generator in self.path_generators.items():
                log.info(f"Running {name} algorithm on '{size}' dataset...")
                generator : PathGenerator
                generated_paths = generator(
                    center_proj.x,
                    center_proj.y,
                    item.radius_km * 1000,
                    item.heatmap,
                    item.bounds,
                )

                all_metrics = evaluator.calculate_all_metrics(
                    generated_paths,
                    0.999
                )

                # Extract results from the returned dictionary
                victim_metrics = all_metrics['victim_detection_metrics']

                result = {
                    "Dataset": size,
                    "Algorithm": name,
                    "Likelihood Score": all_metrics['total_likelihood_score'],
                    "Time-Discounted Score": all_metrics['total_time_discounted_score'],
                    "Victims Found (%)": victim_metrics['percentage_found'],
                    "Area Covered (km²)": all_metrics['area_covered'],
                    "Total Path Length (km)": all_metrics['total_path_length'],
                }

                all_results.append(result)

                # Collect time-series data for this algorithm
                if name not in self.time_series_data:
                    self.time_series_data[name] = []

                # Calculate combined cumulative metrics for time series
                cumulative_likelihoods = all_metrics['cumulative_likelihoods']
                if cumulative_likelihoods:
                    # Handle paths with different lengths by padding shorter arrays
                    max_length_ts = max(len(cum_lik) for cum_lik in cumulative_likelihoods)

                    # Pad arrays to the same length and sum
                    padded_arrays = []
                    for cum_lik in cumulative_likelihoods:
                        if len(cum_lik) < max_length_ts:
                            # Pad with the last value (constant extrapolation)
                            padded = np.pad(cum_lik, (0, max_length_ts - len(cum_lik)), mode='edge')
                        else:
                            padded = cum_lik
                        padded_arrays.append(padded)

                    combined_cumulative_likelihood = np.sum(padded_arrays, axis=0)

                    # Estimate cumulative victims found (simple approximation)
                    # This spreads the total victims found across the time series
                    total_victims_found = victim_metrics['percentage_found'] / 100.0 * self.num_victims
                    combined_cumulative_victims = np.linspace(0, total_victims_found, max_length_ts)
                else:
                    combined_cumulative_likelihood = np.array([0])
                    combined_cumulative_victims = np.array([0])

                # Store time-series results
                self.time_series_data[name].append({
                    'combined_cumulative_likelihood': combined_cumulative_likelihood,
                    'combined_cumulative_victims': combined_cumulative_victims,
                })

        self.results = pd.DataFrame(all_results)
        log.info("--- Baseline Evaluation Complete ---")
        log.info(f"Results:\n{self.results.to_string()}")
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

        plot_single_evaluation_results(results_df, self.evaluation_sizes, output_dir)
