# sarenv/analytics/metrics.py
"""
Provides the PathEvaluator class to score coverage paths against various metrics.
"""
import geopandas as gpd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, MultiLineString

class PathEvaluator:
    """
    Evaluates coverage paths against metrics like likelihood scores, time-discounted
    scores, and victim detection probabilities.
    """
    def __init__(self, heatmap: np.ndarray, extent: tuple, victims: gpd.GeoDataFrame, fov_deg: float, altitude: float, meters_per_bin: int):
        """
        Initializes the PathEvaluator.

        Args:
            heatmap (np.ndarray): The 2D numpy array representing the probability heatmap.
            extent (tuple): A tuple (minx, miny, maxx, maxy) defining the geographical bounds of the heatmap.
            victims (gpd.GeoDataFrame): A GeoDataFrame containing victim locations as points.
            fov_deg (float): The camera's field of view in degrees.
            altitude (float): The altitude of the drone in meters.
            meters_per_bin (int): The size of a heatmap cell in meters.
        """
        self.heatmap = heatmap
        self.extent = extent
        self.victims = victims
        self.detection_radius = altitude * np.tan(np.radians(fov_deg / 2))
        self.interpolation_resolution = int(np.ceil(meters_per_bin / 2))

        minx, miny, maxx, maxy = self.extent
        y_range = np.linspace(miny, maxy, heatmap.shape[0])
        x_range = np.linspace(minx, maxx, heatmap.shape[1])
        self.interpolator = RegularGridInterpolator((y_range, x_range), heatmap, bounds_error=False, fill_value=0)

    def calculate_all_metrics(self, paths: list[LineString], discount_factor) -> dict:
        """
        Calculates all metrics for a given list of paths by processing the
        points along the paths only once.

        Args:
            paths (list[LineString]): A list of Shapely LineString objects representing the drone paths.
            discount_factor (float, optional): The discount factor for time-discounted scores. Defaults to 0.999.

        Returns:
            dict: A dictionary containing all calculated metrics:
                  - 'total_likelihood_score'
                  - 'total_time_discounted_score'
                  - 'victim_detection_metrics'
                  - 'cumulative_distances' (list of arrays)
                  - 'cumulative_likelihoods' (list of arrays)
                  - 'cumulative_time_discounted_scores' (list of arrays)
        """
        total_likelihood = 0
        total_time_discounted_score = 0
        
        # Lists to store cumulative results for each path
        cumulative_distances_all_paths = []
        cumulative_likelihoods_all_paths = []
        cumulative_discounted_scores_all_paths = []

        # --- Main Loop: Process each path once ---
        for path in paths:
            if path.is_empty or path.length == 0:
                # Append empty/zero results for empty paths to maintain list correspondence
                cumulative_distances_all_paths.append(np.array([0]))
                cumulative_likelihoods_all_paths.append(np.array([0]))
                cumulative_discounted_scores_all_paths.append(np.array([0]))
                continue

            # 1. Generate points and distances ONCE
            num_points = int(np.ceil(path.length / self.interpolation_resolution)) + 1
            distances = np.linspace(0, path.length, num_points)
            points = [path.interpolate(d) for d in distances]
            point_coords = [(p.y, p.x) for p in points]

            # 2. Get likelihood values from the interpolator ONCE
            likelihoods = self.interpolator(point_coords)

            # 3. Calculate all point-based metrics using the pre-computed values

            # Aggregate Likelihood Score
            total_likelihood += np.sum(likelihoods)

            # Aggregate Time-Discounted Score
            discounts = discount_factor ** distances
            discounted_likelihoods = likelihoods * discounts
            total_time_discounted_score += np.sum(discounted_likelihoods)

            # Cumulative scores for this specific path
            cumulative_distances_all_paths.append(distances)
            cumulative_likelihoods_all_paths.append(np.cumsum(likelihoods))
            cumulative_discounted_scores_all_paths.append(np.cumsum(discounted_likelihoods))

        # --- Geospatial Metrics (handled separately for efficiency) ---
        victim_metrics = self._calculate_victims_found_score(paths)

        # 4. Assemble the final results dictionary
        results = {
            'total_likelihood_score': total_likelihood,
            'total_time_discounted_score': total_time_discounted_score,
            'victim_detection_metrics': victim_metrics,
            'cumulative_distances': cumulative_distances_all_paths,
            'cumulative_likelihoods': cumulative_likelihoods_all_paths,
            'cumulative_time_discounted_scores': cumulative_discounted_scores_all_paths,
        }

        return results

    def _calculate_victims_found_score(self, paths: list[LineString]) -> dict:
        """
        Calculates victim detection percentage and timeliness.
        This is kept as a separate internal method as its logic is geospatial,
        not point-interpolation based.
        """
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths or self.victims.empty:
            return {'percentage_found': 0, 'average_detection_distance': 0, 'found_victim_indices': []}

        # Use MultiLineString for potentially faster buffering and unioning
        path_collection = MultiLineString(valid_paths)
        coverage_area = path_collection.buffer(self.detection_radius)

        found_victims = self.victims[self.victims.within(coverage_area)]
        
        percentage_found = (len(found_victims) / len(self.victims)) * 100 if not self.victims.empty else 0
        
        timeliness = []
        # Calculate timeliness only for the victims that were found
        for _, victim in found_victims.iterrows():
            # Find the distance along the path to the closest point on any path
            min_dist = min(path.project(victim.geometry) for path in valid_paths)
            timeliness.append(min_dist)
            
        # Timeliness can be represented as the average distance travelled to find a victim
        average_timeliness_distance = np.mean(timeliness) if timeliness else 0

        return {
            'percentage_found': percentage_found, 
            'average_detection_distance': average_timeliness_distance,
            'found_victim_indices': found_victims.index.tolist()
        }