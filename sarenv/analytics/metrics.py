# sarenv/analytics/metrics.py
"""
Provides the PathEvaluator class to score coverage paths against various metrics.
"""
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point, LineString, MultiLineString
from scipy import stats
import sarenv
from sarenv.utils import geo

class DatasetEvaluator:
    def __init__(self, dataset_dirs, path_generators, num_victims, evaluation_size, fov_degrees, altitude_meters, overlap_ratio, num_drones, path_point_spacing_m, transition_distance_m, pizza_border_gap_m, discount_factor):
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

    def evaluate(self):
        # Collect results for each strategy
        strategy_results = {name: [] for name in self.path_generators}

        for dataset_dir in self.dataset_dirs:
            loader = sarenv.DatasetLoader(dataset_directory=dataset_dir)
            item = loader.load_environment(self.evaluation_size)
            if not item:
                continue

            data_crs = geo.get_utm_epsg(item.center_point[0], item.center_point[1])
            victim_generator = sarenv.LostPersonLocationGenerator(item)
            victim_points = [p for p in (victim_generator.generate_location() for _ in range(self.num_victims)) if p]
            victims_gdf = gpd.GeoDataFrame(geometry=victim_points, crs=data_crs) if victim_points else gpd.GeoDataFrame(columns=['geometry'], crs=data_crs)

            evaluator = PathEvaluator(
                item.heatmap,
                item.bounds,
                victims_gdf,
                self.fov_degrees,
                self.altitude_meters,
                loader._meter_per_bin
            )
            center_proj = gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326").to_crs(data_crs).geometry.iloc[0]
            center_x, center_y = center_proj.x, center_proj.y
            max_radius_m = item.radius_km * 1000

            # Generate and evaluate all path strategies
            for name, generator in self.path_generators.items():
                paths = generator(center_x, center_y, max_radius_m, item)
                results = evaluator.calculate_all_metrics(paths, self.discount_factor)
                strategy_results[name].append(results)

        # Aggregate and plot
        for name, results_list in strategy_results.items():
            if not results_list:
                continue
            self._aggregate_and_plot(name, results_list)

    def _aggregate_and_plot(self, strategy_name, results_list):
        # Gather arrays for averaging
        combined_likelihoods = [r['combined_cumulative_likelihood'] for r in results_list]
        combined_victims = [r['combined_cumulative_victims'] for r in results_list]

        def mean_ci(arrays):
            max_len = max(len(a) for a in arrays)
            padded = [np.pad(a, (0, max_len - len(a)), mode='edge') if len(a) < max_len else a for a in arrays]
            data = np.vstack(padded)
            mean = np.mean(data, axis=0)
            sem = stats.sem(data, axis=0)
            h = sem * stats.t.ppf((1 + 0.95) / 2., data.shape[0] - 1)
            return mean, mean - h, mean + h

        mean_likelihood, ci_low_likelihood, ci_high_likelihood = mean_ci(combined_likelihoods)
        mean_victims, ci_low_victims, ci_high_victims = mean_ci(combined_victims)

        self._plot_with_ci(
            mean_likelihood, ci_low_likelihood, ci_high_likelihood,
            mean_victims, ci_low_victims, ci_high_victims,
            strategy_name
        )

    def _plot_with_ci(self, mean_likelihood, ci_low_likelihood, ci_high_likelihood, mean_victims, ci_low_victims, ci_high_victims, strategy_name):
        output_dir = 'graphs/plots'
        os.makedirs(output_dir, exist_ok=True)
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

        plt.title(f'Average Combined Metrics with 95% CI for {strategy_name}')
        fig.tight_layout()
        filename = os.path.join(output_dir, f'{strategy_name}_{self.evaluation_size}_average_combined_metrics.pdf')
        plt.savefig(filename)
        plt.close()

        

class PathEvaluator:
    """
    Evaluates coverage paths against metrics like likelihood scores, time-discounted
    scores, and victim detection probabilities, with accurate unique victim tracking.
    """
    def __init__(self, heatmap: np.ndarray, extent: tuple, victims: gpd.GeoDataFrame, fov_deg: float, altitude: float, meters_per_bin: int):
        self.heatmap = heatmap
        self.extent = extent
        self.victims = victims
        self.detection_radius = altitude * np.tan(np.radians(fov_deg / 2))
        self.interpolation_resolution = int(np.ceil(meters_per_bin / 2))

        minx, miny, maxx, maxy = self.extent
        y_range = np.linspace(miny, maxy, heatmap.shape[0])
        x_range = np.linspace(minx, maxx, heatmap.shape[1])
        self.interpolator = RegularGridInterpolator((y_range, x_range), heatmap, bounds_error=False, fill_value=0)

        # This will store, for each path, a list of sets of victim indices found up to each time step
        self.per_path_found_victim_indices = []

    def _calculate_cumulative_victims_found(self, path: LineString, distances: np.ndarray):
        """
        Calculates cumulative victims found and returns both counts and sets of victim indices.
        """
        proj_distances = self.victims.geometry.apply(lambda v: path.project(v))
        path_points = proj_distances.apply(lambda d: path.interpolate(d))
        path_points_gs = gpd.GeoSeries(path_points, crs=self.victims.crs)
        dists = self.victims.geometry.distance(path_points_gs)
        
        detected_mask = dists <= self.detection_radius
        detection_distances = np.where(detected_mask, proj_distances, np.inf)

        # Get the indices of victims sorted by their detection distance
        victim_indices = np.argsort(detection_distances)
        sorted_detection_distances = detection_distances[victim_indices]

        # Get cumulative counts efficiently
        cumulative_counts = np.searchsorted(sorted_detection_distances, distances, side='right')

        # Build the list of sets of found victim indices
        found_victim_sets = []
        current_found = set()
        idx = 0
        n_victims = len(self.victims)

        for dist in distances:
            # Add all victims detected at or before this distance
            while idx < n_victims and sorted_detection_distances[idx] <= dist:
                if sorted_detection_distances[idx] != np.inf:
                    current_found.add(victim_indices[idx])
                idx += 1
            found_victim_sets.append(set(current_found))

        return cumulative_counts, found_victim_sets

    def calculate_all_metrics(self, paths: list, discount_factor) -> dict:
        total_likelihood = 0
        total_time_discounted_score = 0

        cumulative_distances_all_paths = []
        cumulative_likelihoods_all_paths = []
        cumulative_discounted_scores_all_paths = []
        cumulative_victims_found_all_paths = []
        self.per_path_found_victim_indices = []

        for path in paths:
            if path.is_empty or path.length == 0:
                cumulative_distances_all_paths.append(np.array([0]))
                cumulative_likelihoods_all_paths.append(np.array([0]))
                cumulative_discounted_scores_all_paths.append(np.array([0]))
                cumulative_victims_found_all_paths.append(np.array([0]))
                self.per_path_found_victim_indices.append([set()])
                continue

            num_points = int(np.ceil(path.length / self.interpolation_resolution)) + 1
            distances = np.linspace(0, path.length, num_points)
            points = [path.interpolate(d) for d in distances]
            point_coords = [(p.y, p.x) for p in points]

            likelihoods = self.interpolator(point_coords)
            total_likelihood += np.sum(likelihoods)

            discounts = discount_factor ** distances
            discounted_likelihoods = likelihoods * discounts
            total_time_discounted_score += np.sum(discounted_likelihoods)

            cumulative_distances_all_paths.append(distances)
            cumulative_likelihoods_all_paths.append(np.cumsum(likelihoods))
            cumulative_discounted_scores_all_paths.append(np.cumsum(discounted_likelihoods))

            # Get both counts and sets of found victims
            cumulative_counts, found_victim_sets = self._calculate_cumulative_victims_found(path, distances)
            cumulative_victims_found_all_paths.append(cumulative_counts)
            self.per_path_found_victim_indices.append(found_victim_sets)

        victim_metrics = self._calculate_victims_found_score(paths)

        total_heatmap_prob = np.sum(self.heatmap)
        normalized_cumulative_likelihoods = [arr / total_heatmap_prob for arr in cumulative_likelihoods_all_paths]
        normalized_total_likelihood = total_likelihood / total_heatmap_prob

        combined_cumulative_likelihood, combined_cumulative_victims = self._calculate_combined_metrics(
            normalized_cumulative_likelihoods
        )

        results = {
            'total_likelihood_score': normalized_total_likelihood,
            'total_time_discounted_score': total_time_discounted_score / total_heatmap_prob,
            'victim_detection_metrics': victim_metrics,
            'cumulative_distances': cumulative_distances_all_paths,
            'cumulative_likelihoods': normalized_cumulative_likelihoods,
            'cumulative_time_discounted_scores': [arr / total_heatmap_prob for arr in cumulative_discounted_scores_all_paths],
            'cumulative_victims_found': cumulative_victims_found_all_paths,
            'combined_cumulative_likelihood': combined_cumulative_likelihood,
            'combined_cumulative_victims': combined_cumulative_victims,
        }
        return results

    def _calculate_combined_metrics(self, cumulative_likelihoods_all_paths):
        """
        Combines metrics across all paths, ensuring unique victim counts.
        """
        def pad_array(arr, length):
            if len(arr) < length:
                return np.pad(arr, (0, length - len(arr)), mode='edge')
            return arr

        # Determine the maximum length needed for padding
        max_len_like = max(len(arr) for arr in cumulative_likelihoods_all_paths) if cumulative_likelihoods_all_paths else 0
        max_len_vic = max(len(sets) for sets in self.per_path_found_victim_indices) if self.per_path_found_victim_indices else 0
        max_len = max(max_len_like, max_len_vic)

        # Pad and sum likelihoods, capping at 1.0
        padded_likelihoods = [pad_array(arr, max_len) for arr in cumulative_likelihoods_all_paths]
        combined_likelihood = np.minimum(np.sum(padded_likelihoods, axis=0), 1.0)

        # Pad victim sets and union them for each time step
        padded_victim_sets = []
        for victim_sets in self.per_path_found_victim_indices:
            if len(victim_sets) < max_len:
                last_set = victim_sets[-1] if victim_sets else set()
                padded_victim_sets.append(victim_sets + [last_set] * (max_len - len(victim_sets)))
            else:
                padded_victim_sets.append(victim_sets)

        combined_victims = []
        for t in range(max_len):
            combined_set = set()
            for victim_sets in padded_victim_sets:
                combined_set.update(victim_sets[t])
            combined_victims.append(len(combined_set))

        return combined_likelihood, np.array(combined_victims)

    def _calculate_victims_found_score(self, paths: list) -> dict:
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths or self.victims.empty:
            return {'percentage_found': 0, 'average_detection_distance': 0, 'found_victim_indices': []}

        path_collection = MultiLineString(valid_paths)
        coverage_area = path_collection.buffer(self.detection_radius)
        found_victims = self.victims[self.victims.within(coverage_area)]
        percentage_found = (len(found_victims) / len(self.victims)) * 100 if not self.victims.empty else 0

        timeliness = []
        for _, victim in found_victims.iterrows():
            min_dist = min(path.project(victim.geometry) for path in valid_paths)
            timeliness.append(min_dist)
        average_timeliness_distance = np.mean(timeliness) if timeliness else 0

        return {
            'percentage_found': percentage_found,
            'average_detection_distance': average_timeliness_distance,
            'found_victim_indices': found_victims.index.tolist()
        }
