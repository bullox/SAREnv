# toolkit.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import sarenv
from sarenv.analytics import paths, metrics
from sarenv.utils import geo
from sarenv.utils.plot import (
    plot_aggregate_bars, 
    plot_combined_normalized_bars, 
    plot_time_series_with_ci,
    plot_combined_time_series_with_ci,
    plot_single_evaluation_results
)
from shapely.geometry import Point
from sarenv.utils.logging_setup import get_logger


log = get_logger()


class PathGeneratorConfig:
    """
    Configuration class for path generation parameters.
    Provides a clean interface for managing path generation parameters.
    """
    
    def __init__(self, 
                 num_drones: int = 3,
                 fov_degrees: float = 45.0,
                 altitude_meters: float = 80.0,
                 overlap_ratio: float = 0.25,
                 path_point_spacing_m: float = 10.0,
                 transition_distance_m: float = 50.0,
                 pizza_border_gap_m: float = 15.0,
                 **kwargs):
        """
        Initialize path generation configuration.
        
        Args:
            num_drones: Number of drones to simulate
            fov_degrees: Field of view in degrees
            altitude_meters: Altitude in meters
            overlap_ratio: Overlap ratio for systematic patterns
            path_point_spacing_m: Spacing between path points in meters
            transition_distance_m: Transition distance for concentric patterns
            pizza_border_gap_m: Border gap for pizza patterns
            **kwargs: Additional parameters for custom generators
        """
        self.num_drones = num_drones
        self.fov_degrees = fov_degrees
        self.altitude_meters = altitude_meters
        self.overlap_ratio = overlap_ratio
        self.path_point_spacing_m = path_point_spacing_m
        self.transition_distance_m = transition_distance_m
        self.pizza_border_gap_m = pizza_border_gap_m
        
        # Store additional parameters
        self.additional_params = kwargs
    
    def get_params_dict(self, center_x: float, center_y: float, max_radius: float, 
                       probability_map: np.ndarray, bounds: tuple, budget: float = None) -> dict:
        """
        Generate a complete parameter dictionary for path generation.
        
        Args:
            center_x: X coordinate of the center point
            center_y: Y coordinate of the center point
            max_radius: Maximum search radius
            probability_map: 2D probability map
            bounds: Geographic bounds tuple
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
    """
    
    def __init__(self, name: str, func, description: str = ""):
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
        
    def generate(self, config: PathGeneratorConfig, center_x: float, center_y: float, 
                max_radius: float, probability_map: np.ndarray, bounds: tuple, 
                budget: float = None):
        """
        Generate paths using the configured function.
        
        Args:
            config: PathGeneratorConfig with all parameters
            center_x: X coordinate of center point
            center_y: Y coordinate of center point
            max_radius: Maximum search radius
            probability_map: 2D probability map
            bounds: Geographic bounds tuple
            budget: Optional budget constraint
            
        Returns:
            List of LineString paths
        """
        params = config.get_params_dict(center_x, center_y, max_radius, 
                                       probability_map, bounds, budget)
        return self.func(**params)
    
    def __call__(self, *args, **kwargs):
        """Allow the generator to be called directly."""
        return self.func(*args, **kwargs)


def get_default_path_generators(config: PathGeneratorConfig) -> dict:
    """
    Get default path generators with consistent parameter handling.
    
    Args:
        config: PathGeneratorConfig with default parameters
        
    Returns:
        Dictionary of PathGenerator instances
    """
    return {
        "RandomWalk": PathGenerator(
            name="RandomWalk",
            func=paths.generate_random_walk_path,
            description="Random walk path generation"
        ),
        "Greedy": PathGenerator(
            name="Greedy",
            func=paths.generate_greedy_path,
            description="Greedy path generation based on probability map"
        ),
        "Spiral": PathGenerator(
            name="Spiral",
            func=paths.generate_spiral_path,
            description="Spiral path generation"
        ),
        "Concentric": PathGenerator(
            name="Concentric",
            func=paths.generate_concentric_circles_path,
            description="Concentric circles path generation"
        ),
        "Pizza": PathGenerator(
            name="Pizza",
            func=paths.generate_pizza_zigzag_path,
            description="Pizza slice zigzag path generation"
        )
    }


class ComparativeDatasetEvaluator:
    """
    Evaluates multiple datasets using ComparativeEvaluator to minimize code duplication.
    Provides time-series plotting and confidence interval analysis.
    """
    
    def __init__(
        self,
        dataset_dirs,
        path_generators=None,  # Can be dict of PathGenerator instances or functions
        num_victims=100,
        evaluation_size="medium",
        fov_degrees=45.0,
        altitude_meters=80.0,
        overlap_ratio=0.25,
        num_drones=3,
        path_point_spacing_m=10.0,
        transition_distance_m=50.0,
        pizza_border_gap_m=15.0,
        discount_factor=0.999,
        **additional_params
    ):
        self.dataset_dirs = dataset_dirs
        
        # Create path generator configuration
        self.path_config = PathGeneratorConfig(
            num_drones=num_drones,
            fov_degrees=fov_degrees,
            altitude_meters=altitude_meters,
            overlap_ratio=overlap_ratio,
            path_point_spacing_m=path_point_spacing_m,
            transition_distance_m=transition_distance_m,
            pizza_border_gap_m=pizza_border_gap_m,
            **additional_params
        )
        
        # Set up path generators
        if path_generators is None:
            self.path_generators = get_default_path_generators(self.path_config)
        else:
            self.path_generators = {}
            for name, generator in path_generators.items():
                if isinstance(generator, PathGenerator):
                    self.path_generators[name] = generator
                else:
                    # Assume it's a function and wrap it
                    self.path_generators[name] = PathGenerator(name, generator)
        
        self.num_victims = num_victims
        self.evaluation_size = evaluation_size
        self.discount_factor = discount_factor

        # Create a ComparativeEvaluator for each dataset
        self.evaluators = []
        for dataset_dir in dataset_dirs:
            evaluator = ComparativeEvaluator(
                dataset_directory=dataset_dir,
                evaluation_sizes=[evaluation_size],
                num_drones=num_drones,
                num_lost_persons=num_victims,
                path_config=self.path_config,
                path_generators=self.path_generators
            )
            self.evaluators.append(evaluator)

        # Storage for summary and time-series results
        self.summary_results = []
        self.time_series_results = {name: [] for name in self.path_generators}

    def evaluate(self):
        """
        Evaluates all path generators across all datasets using ComparativeEvaluator instances.
        Now only uses data generated by ComparativeEvaluator - no direct path evaluation.
        """
        for i, evaluator in enumerate(self.evaluators):
            dataset_dir = self.dataset_dirs[i]
            log.info(f"Evaluating dataset: {dataset_dir}")
            
            # Run evaluation for this dataset
            results_df = evaluator.run_baseline_evaluations()
            
            if results_df.empty:
                continue
                
            # Store summary results from ComparativeEvaluator
            for _, row in results_df.iterrows():
                self.summary_results.append({
                    "Algorithm": row["Algorithm"],
                    "Dataset": os.path.basename(dataset_dir),
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



    def __init__(
        self,
        dataset_directory="sarenv_dataset",
        evaluation_sizes=None,
        num_drones=3,
        num_lost_persons=100,
        path_config=None,  # PathGeneratorConfig instance
        path_generators=None,  # Dict of PathGenerator instances
        # Legacy parameters for backward compatibility
        custom_path_generators=None,
        use_defaults=True,
    ):
        """
        Initializes the ComparativeEvaluator.

        Args:
            dataset_directory (str): The path to the sarenv dataset.
            evaluation_sizes (list, optional): A list of dataset sizes to evaluate.
            num_drones (int): The number of drones to simulate.
            num_lost_persons (int): The number of victim locations to generate per dataset.
            path_config (PathGeneratorConfig, optional): Configuration object for path parameters.
            path_generators (dict, optional): Dict of {name: PathGenerator} instances.
            custom_path_generators (dict, optional): Legacy parameter for backward compatibility.
            use_defaults (bool): Whether to include default path generators.
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
        self.time_series_data = {}

        # Create path config if not provided
        if path_config is None:
            self.path_config = PathGeneratorConfig(num_drones=num_drones)
        else:
            self.path_config = path_config

        # Static budgets for each environment size (in meters)
        self.budget_by_size = {
            "small": 5000.0,    # 5 km
            "medium": 200000.0,  # 200 km
            "large": 200000.0,   # 200 km
            "xlarge": 100000.0,  # 500 km
        }

        # Set up path generators
        if path_generators is not None:
            self.path_generators = path_generators
        elif custom_path_generators is not None:
            # Legacy support - convert old style to new style
            self.path_generators = {}
            for name, func in custom_path_generators.items():
                if callable(func):
                    if isinstance(func, PathGenerator):
                        self.path_generators[name] = func
                    else:
                        self.path_generators[name] = PathGenerator(name, func)
                else:
                    log.warning(f"Custom path generator '{name}' is not callable and was skipped.")
        elif use_defaults:
            self.path_generators = get_default_path_generators(self.path_config)
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
                self.path_config.fov_degrees,
                self.path_config.altitude_meters,
                self.loader._meter_per_bin
            )
            center_proj = (
                gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
                .to_crs(env_data["crs"])
                .geometry.iloc[0]
            )

            current_budget = self.budget_by_size.get(size, 10000.0)

            for name, generator in self.path_generators.items():
                log.info(f"Running {name} algorithm on '{size}' dataset...")
                
                # Use the new PathGenerator interface
                generated_paths = generator.generate(
                    self.path_config,
                    center_proj.x,
                    center_proj.y,
                    item.radius_km * 1000,
                    item.heatmap,
                    item.bounds,
                    current_budget
                )

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
