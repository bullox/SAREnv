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
        Calculates all metrics for a given list of paths using a view model that considers
        the detection radius to determine which grid cells are visible from each position.

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
        # Track globally observed cells using the view model
        globally_observed_cells = set()
        all_position_data = []
        path_metadata = []
        global_time_offset = 0

        # First pass: collect all positions and their visible cells
        for path_idx, path in enumerate(paths):
            if path.is_empty or path.length == 0:
                path_metadata.append({
                    'path_idx': path_idx,
                    'distances': np.array([0]),
                    'positions': [],
                    'global_distances': np.array([0]),
                    'visible_cells_per_position': []
                })
                continue

            num_points = int(np.ceil(path.length / self.interpolation_resolution)) + 1
            distances = np.linspace(0, path.length, num_points)
            points = [path.interpolate(d) for d in distances]
            positions = [(p.x, p.y) for p in points]

            # Global distances for time-discounted scores
            global_distances = distances + global_time_offset

            # Calculate visible cells for each position
            visible_cells_per_position = []
            for x, y in positions:
                visible_cells = self.get_visible_cells(x, y)
                visible_cells_per_position.append(visible_cells)

            # Store metadata for this path
            path_metadata.append({
                'path_idx': path_idx,
                'distances': distances,
                'positions': positions,
                'global_distances': global_distances,
                'visible_cells_per_position': visible_cells_per_position
            })

            # Add positions to global collection
            for i, ((x, y), visible_cells) in enumerate(zip(positions, visible_cells_per_position, strict=True)):
                all_position_data.append({
                    'position': (x, y),
                    'visible_cells': visible_cells,
                    'path_idx': path_idx,
                    'position_idx': i,
                    'global_distance': global_distances[i]
                })

            global_time_offset += path.length

        # Second pass: calculate metrics using the view model
        if all_position_data:
            # Calculate total likelihood by tracking all observed cells
            all_observed_cells = set()
            for position_data in all_position_data:
                all_observed_cells.update(position_data['visible_cells'])

            # Calculate total likelihood (sum of all unique observed cells)
            total_likelihood = sum(self.heatmap[row, col] for row, col in all_observed_cells)

            # Calculate time-discounted score considering all visible cells at each position
            total_time_discounted_score = 0
            for position_data in all_position_data:
                position_score = sum(self.heatmap[row, col] for row, col in position_data['visible_cells'])
                discount = discount_factor ** position_data['global_distance']
                total_time_discounted_score += position_score * discount
        else:
            total_likelihood = 0
            total_time_discounted_score = 0

        # Third pass: generate cumulative results for each path
        cumulative_distances_all_paths = []
        cumulative_likelihoods_all_paths = []
        cumulative_discounted_scores_all_paths = []

        for meta in path_metadata:
            if not meta['positions']:
                cumulative_distances_all_paths.append(np.array([0]))
                cumulative_likelihoods_all_paths.append(np.array([0]))
                cumulative_discounted_scores_all_paths.append(np.array([0]))
                continue

            # Track cells observed by this path (without double-counting within the same path)
            path_observed_cells = set()
            path_likelihoods = []
            path_discounted_likelihoods = []

            for i, visible_cells in enumerate(meta['visible_cells_per_position']):
                # Only count cells not yet observed in this path
                new_cells = visible_cells - path_observed_cells
                path_observed_cells.update(new_cells)

                # Calculate likelihood for new cells only
                position_likelihood = sum(self.heatmap[row, col] for row, col in new_cells)
                discount = discount_factor ** meta['distances'][i]

                path_likelihoods.append(position_likelihood)
                path_discounted_likelihoods.append(position_likelihood * discount)

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

    def get_visible_cells(self, x: float, y: float) -> set[tuple[int, int]]:
        """
        Get all grid cells that are visible from a given position based on the detection radius.
        
        Args:
            x (float): X coordinate in world space
            y (float): Y coordinate in world space
            
        Returns:
            set[tuple[int, int]]: Set of (row, col) tuples representing visible grid cells
        """
        minx, miny, maxx, maxy = self.extent
        
        # Convert detection radius to grid cells
        radius_in_cells_x = int(np.ceil(self.detection_radius / self.heatmap_cell_size_x))
        radius_in_cells_y = int(np.ceil(self.detection_radius / self.heatmap_cell_size_y))
        
        # Get the center cell position
        center_col = int((x - minx) / self.heatmap_cell_size_x)
        center_row = int((y - miny) / self.heatmap_cell_size_y)
        
        visible_cells = set()
        
        # Check all cells within the radius
        for row in range(max(0, center_row - radius_in_cells_y),
                        min(self.heatmap.shape[0], center_row + radius_in_cells_y + 1)):
            for col in range(max(0, center_col - radius_in_cells_x),
                            min(self.heatmap.shape[1], center_col + radius_in_cells_x + 1)):
                
                # Calculate the world coordinates of this cell's center
                cell_x = minx + (col + 0.5) * self.heatmap_cell_size_x
                cell_y = miny + (row + 0.5) * self.heatmap_cell_size_y
                
                # Check if this cell is within the detection radius
                distance = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)
                if distance <= self.detection_radius:
                    visible_cells.add((row, col))
        
        return visible_cells

    def calculate_view_score_at_position(self, x: float, y: float, visited_cells: set[tuple[int, int]]) -> float:
        """
        Calculate the total likelihood score for all visible cells from a position,
        excluding already visited cells.
        
        Args:
            x (float): X coordinate in world space
            y (float): Y coordinate in world space
            visited_cells (set): Set of already visited (row, col) tuples
            
        Returns:
            float: Total likelihood score for unvisited visible cells
        """
        visible_cells = self.get_visible_cells(x, y)
        total_score = 0.0
        
        for row, col in visible_cells:
            if (row, col) not in visited_cells:
                total_score += self.heatmap[row, col]
        
        return total_score
