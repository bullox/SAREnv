# sarenv/analytics/metrics.py
"""
Provides the PathEvaluator class to score coverage paths against various metrics.
"""
import geopandas as gpd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString
from shapely.ops import unary_union

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

        # Calculate actual heatmap cell size for proper cell_key calculation
        minx, miny, maxx, maxy = self.extent
        self.heatmap_cell_size_x = (maxx - minx) / heatmap.shape[1]
        self.heatmap_cell_size_y = (maxy - miny) / heatmap.shape[0]

        minx, miny, maxx, maxy = self.extent
        y_range = np.linspace(miny, maxy, heatmap.shape[0])
        x_range = np.linspace(minx, maxx, heatmap.shape[1])
        self.interpolator = RegularGridInterpolator((y_range, x_range), heatmap, bounds_error=False, fill_value=0)

    def calculate_all_metrics(self, paths: list[LineString], discount_factor) -> dict:
        """
        Calculates all metrics for a given list of paths by collecting all points
        first and then calculating unique likelihood scores.

        Args:
            paths (list[LineString]): A list of Shapely LineString objects representing the drone paths.
            discount_factor (float, optional): The discount factor for time-discounted scores. Defaults to 0.999.

        Returns:
            dict: A dictionary containing all calculated metrics:
                  - 'total_likelihood_score'
                  - 'total_time_discounted_score'
                  - 'victim_detection_metrics'
                  - 'area_covered'
                  - 'total_path_length'
                  - 'cumulative_distances' (list of arrays)
                  - 'cumulative_likelihoods' (list of arrays)
                  - 'cumulative_time_discounted_scores' (list of arrays)
        """
        # First pass: collect all points from all paths with their metadata
        all_points_data = []
        path_metadata = []
        global_time_offset = 0


        for path_idx, path in enumerate(paths):
            if path.is_empty or path.length == 0:
                path_metadata.append({
                    'path_idx': path_idx,
                    'distances': np.array([0]),
                    'point_coords': [],
                    'global_distances': np.array([0]),
                    'start_idx': len(all_points_data),
                    'end_idx': len(all_points_data)
                })
                continue

            num_points = int(np.ceil(path.length / self.interpolation_resolution)) + 1
            distances = np.linspace(0, path.length, num_points)
            points = [path.interpolate(d) for d in distances]
            point_coords = [(p.y, p.x) for p in points]

            # Global distances for time-discounted scores
            global_distances = distances + global_time_offset

            # Store metadata for this path
            path_metadata.append({
                'path_idx': path_idx,
                'distances': distances,
                'point_coords': point_coords,
                'global_distances': global_distances,
                'start_idx': len(all_points_data),
                'end_idx': len(all_points_data) + len(point_coords)
            })

            # Add points to global collection with their metadata
            for i, (y, x) in enumerate(point_coords):
                # Use heatmap grid for cell_key calculation to avoid double-counting
                minx, miny, _, _ = self.extent
                heatmap_row = int((y - miny) / self.heatmap_cell_size_y)
                heatmap_col = int((x - minx) / self.heatmap_cell_size_x)

                # Ensure bounds safety
                heatmap_row = max(0, min(heatmap_row, self.heatmap.shape[0] - 1))
                heatmap_col = max(0, min(heatmap_col, self.heatmap.shape[1] - 1))

                cell_key = (heatmap_row, heatmap_col)
                all_points_data.append({
                    'coords': (y, x),
                    'cell_key': cell_key,
                    'path_idx': path_idx,
                    'point_idx': i,
                    'global_distance': global_distances[i]
                })

            global_time_offset += path.length

        # Second pass: calculate likelihoods for all unique points
        if all_points_data:
            # Get all unique cell keys and their first occurrence
            unique_cells = {}
            for i, point_data in enumerate(all_points_data):
                cell_key = point_data['cell_key']
                if cell_key not in unique_cells:
                    unique_cells[cell_key] = {
                        'coords': point_data['coords'],
                        'first_occurrence_idx': i
                    }

            # Calculate likelihoods for unique cells only
            unique_coords = [data['coords'] for data in unique_cells.values()]
            unique_likelihoods = self.interpolator(unique_coords)

            # Create mapping from cell_key to likelihood
            cell_likelihood_map = {}
            for i, cell_key in enumerate(unique_cells.keys()):
                cell_likelihood_map[cell_key] = unique_likelihoods[i]

            # Calculate total likelihood (sum of all unique cells)
            total_likelihood = np.sum(unique_likelihoods)

            # Calculate time-discounted score for all points (including duplicates)
            total_time_discounted_score = 0
            for point_data in all_points_data:
                likelihood = cell_likelihood_map[point_data['cell_key']]
                discount = discount_factor ** point_data['global_distance']
                total_time_discounted_score += likelihood * discount
        else:
            total_likelihood = 0
            total_time_discounted_score = 0

        # Third pass: generate cumulative results for each path
        # Track globally visited cells to avoid double-counting in cumulative metrics
        globally_visited_cells = set()
        cumulative_distances_all_paths = []
        cumulative_likelihoods_all_paths = []
        cumulative_discounted_scores_all_paths = []

        for meta in path_metadata:
            if not meta['point_coords']:
                cumulative_distances_all_paths.append(np.array([0]))
                cumulative_likelihoods_all_paths.append(np.array([0]))
                cumulative_discounted_scores_all_paths.append(np.array([0]))
                continue

            # Get likelihoods for this path's points, only counting new cells
            path_likelihoods = []
            path_discounted_likelihoods = []

            for i, (y, x) in enumerate(meta['point_coords']):
                # Use same heatmap grid calculation as above
                minx, miny, _, _ = self.extent
                heatmap_row = int((y - miny) / self.heatmap_cell_size_y)
                heatmap_col = int((x - minx) / self.heatmap_cell_size_x)

                # Ensure bounds safety
                heatmap_row = max(0, min(heatmap_row, self.heatmap.shape[0] - 1))
                heatmap_col = max(0, min(heatmap_col, self.heatmap.shape[1] - 1))

                cell_key = (heatmap_row, heatmap_col)

                # Only count likelihood if this cell hasn't been visited by previous paths
                if cell_key not in globally_visited_cells:
                    likelihood = cell_likelihood_map[cell_key]
                    globally_visited_cells.add(cell_key)
                else:
                    likelihood = 0.0  # Already counted by a previous path

                discount = discount_factor ** meta['distances'][i]

                path_likelihoods.append(likelihood)
                path_discounted_likelihoods.append(likelihood * discount)

            cumulative_distances_all_paths.append(meta['distances'])
            cumulative_likelihoods_all_paths.append(np.cumsum(path_likelihoods))
            cumulative_discounted_scores_all_paths.append(np.cumsum(path_discounted_likelihoods))

        # # --- Geospatial Metrics (handled separately for efficiency) ---
        # victim_metrics = self._calculate_victims_found_score(paths)
        # area_covered = self._calculate_area_covered(paths)
        total_path_length = self._calculate_total_path_length(paths)

        # 4. Assemble the final results dictionary
        return {
            'total_likelihood_score': total_likelihood,
            'total_time_discounted_score': total_time_discounted_score,
            'victim_detection_metrics': {'percentage_found': 0, 'found_victim_indices': []},
            'area_covered': 0,
            'total_path_length': total_path_length,
            'cumulative_distances': cumulative_distances_all_paths,
            'cumulative_likelihoods': cumulative_likelihoods_all_paths,
            'cumulative_time_discounted_scores': cumulative_discounted_scores_all_paths,
        }

    def _calculate_victims_found_score(self, paths: list[LineString]) -> dict:
        """
        Calculates victim detection percentage and timeliness.
        This is kept as a separate internal method as its logic is geospatial,
        not point-interpolation based.
        """
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths or self.victims.empty:
            return {'percentage_found': 0, 'found_victim_indices': []}

        # Memory-efficient approach: use unary_union to avoid intermediate geometry accumulation
        buffered_paths = [p.buffer(self.detection_radius) for p in valid_paths]
        coverage_area = unary_union(buffered_paths)

        # Clear the buffered_paths list to free memory
        del buffered_paths

        found_victims = self.victims[self.victims.within(coverage_area)]

        percentage_found = (len(found_victims) / len(self.victims)) * 100 if not self.victims.empty else 0

        return {
            'percentage_found': percentage_found,
            'found_victim_indices': found_victims.index.tolist()
        }

    def _calculate_area_covered(self, paths: list[LineString]) -> float:
        """
        Calculates the area covered by the paths within the detection radius.
        Handles overlapping paths by computing the union of all buffered areas.

        Args:
            paths (list[LineString]): A list of Shapely LineString objects representing the drone paths.

        Returns:
            float: The total area covered by the paths in square kilometers, considering the detection radius,
                   with no double-counting of overlapping areas.
        """
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths:
            return 0.0

        # Memory-efficient approach: use unary_union to avoid intermediate geometry accumulation
        buffered_paths = [path.buffer(self.detection_radius) for path in valid_paths]
        combined_coverage = unary_union(buffered_paths)

        # Clear the buffered_paths list to free memory
        del buffered_paths

        return combined_coverage.area / 1_000_000  # Convert from m² to km²

    def _calculate_total_path_length(self, paths: list[LineString]) -> float:
        """
        Calculates the total length of all agent paths.

        Args:
            paths (list[LineString]): A list of Shapely LineString objects representing the drone paths.

        Returns:
            float: The total length of all paths in kilometers.
        """
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths:
            return 0.0

        total_length = sum(path.length for path in valid_paths)
        return total_length / 1000  # Convert from meters to kilometers
