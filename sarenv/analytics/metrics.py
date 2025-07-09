# sarenv/analytics/metrics.py
"""
Provides the PathEvaluator class to score coverage paths against various metrics.
"""
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, interp1d
from shapely.geometry import Point, LineString, MultiLineString

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

    def _resample_metric_by_distance(self, original_distances, original_values, new_distances):
        # Handles both 1D arrays and list-of-sets (for victim sets)
        if len(original_distances) == 1:
            # Path is a single point, just repeat value
            if isinstance(original_values[0], set):
                return [original_values[0] for _ in new_distances]
            else:
                return np.full_like(new_distances, original_values[0], dtype=float)
        if isinstance(original_values[0], set):
            # For sets, use the set at the last distance less than or equal to each new_distance
            idxs = np.searchsorted(original_distances, new_distances, side='right') - 1
            idxs = np.clip(idxs, 0, len(original_values) - 1)
            return [original_values[i] for i in idxs]
        else:
            interp_func = interp1d(original_distances, original_values, kind='linear', bounds_error=False, fill_value=(original_values[0], original_values[-1]))
            return interp_func(new_distances)

    def calculate_all_metrics(self, paths: list, discount_factor) -> dict:
        """
        Calculates metrics for given paths using a probability map (heatmap).
        Ensures each grid cell is only counted once for total likelihood.
        Uses the same grid mapping as in generate_greedy_path.
        """
        total_time_discounted_score = 0

        cumulative_distances_all_paths = []
        cumulative_likelihoods_all_paths = []
        cumulative_discounted_scores_all_paths = []
        cumulative_victims_found_all_paths = []
        self.per_path_found_victim_indices = []

        # --- Setup grid mapping as in generate_greedy_path ---
        probability_map = self.heatmap
        height, width = probability_map.shape
        minx, miny, maxx, maxy = self.extent

        x_map = np.linspace(minx + (maxx - minx) / (2 * width), maxx - (maxx - minx) / (2 * width), width)
        y_map = np.linspace(miny + (maxy - miny) / (2 * height), maxy - (maxy - miny) / (2 * height), height)

        def coord_to_index(coord):
            """Convert (y, x) coordinate to (row, col) index in the heatmap."""
            y, x = coord
            # Find nearest index in y_map and x_map
            row = np.argmin(np.abs(y_map - y))
            col = np.argmin(np.abs(x_map - x))
            # Clamp to grid
            row = max(0, min(row, height - 1))
            col = max(0, min(col, width - 1))
            return (row, col)

        # --- Track unique visited indices ---
        visited_indices = set()
        unique_likelihood_sum = 0

        max_path_length = 0

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

            # Interpolate likelihoods at each point using the heatmap
            likelihoods = []
            for coord in point_coords:
                row, col = coord_to_index(coord)
                likelihoods.append(probability_map[row, col])
            likelihoods = np.array(likelihoods)

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

            # --- Only count each heatmap cell once ---
            for idx, coord in enumerate(point_coords):
                row, col = coord_to_index(coord)
                if (row, col) not in visited_indices:
                    visited_indices.add((row, col))
                    unique_likelihood_sum += probability_map[row, col]

            max_path_length = max(max_path_length, path.length)

        victim_metrics = self._calculate_victims_found_score(paths)

        total_heatmap_prob = np.sum(probability_map)
        normalized_cumulative_likelihoods = [arr / total_heatmap_prob for arr in cumulative_likelihoods_all_paths]
        normalized_total_likelihood = unique_likelihood_sum / total_heatmap_prob

        combined_cumulative_likelihood, combined_cumulative_victims = self._calculate_combined_metrics(
            normalized_cumulative_likelihoods
        )

        # --- RESAMPLING TO 10-METER STEPS ---
        step = 10
        resample_distances = np.arange(0, max_path_length + step, step)

        # Per-path resampled metrics
        resampled_cumulative_likelihoods = []
        resampled_cumulative_victims_found = []
        for dists, likes, vics, sets in zip(
            cumulative_distances_all_paths,
            normalized_cumulative_likelihoods,
            cumulative_victims_found_all_paths,
            self.per_path_found_victim_indices,
        ):
            resampled_cumulative_likelihoods.append(
                self._resample_metric_by_distance(dists, likes, resample_distances)
            )
            resampled_cumulative_victims_found.append(
                self._resample_metric_by_distance(dists, vics, resample_distances)
            )

        # Combined resampled metrics
        resampled_combined_cumulative_likelihood = self._resample_metric_by_distance(
            resample_distances[:len(combined_cumulative_likelihood)], combined_cumulative_likelihood, resample_distances
        )
        resampled_combined_cumulative_victims = self._resample_metric_by_distance(
            resample_distances[:len(combined_cumulative_victims)], combined_cumulative_victims, resample_distances
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
            # New: resampled metrics at 10m steps
            'resample_distances': resample_distances,
            'resampled_cumulative_likelihoods': resampled_cumulative_likelihoods,
            'resampled_cumulative_victims_found': resampled_cumulative_victims_found,
            'resampled_combined_cumulative_likelihood': resampled_combined_cumulative_likelihood,
            'resampled_combined_cumulative_victims': resampled_combined_cumulative_victims,
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