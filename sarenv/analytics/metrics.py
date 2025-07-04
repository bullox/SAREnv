# sarenv/analytics/metrics.py
"""
Provides the PathEvaluator class to score coverage paths against various metrics.
"""
import geopandas as gpd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString


class PathEvaluator:
    def __init__(self, heatmap: np.ndarray, extent: tuple, victims: gpd.GeoDataFrame, fov_deg: float, altitude: float):
        self.heatmap = heatmap
        self.extent = extent
        self.victims = victims
        self.detection_radius = altitude * np.tan(np.radians(fov_deg / 2))
        minx, miny, maxx, maxy = self.extent
        y_range = np.linspace(miny, maxy, heatmap.shape[0])
        x_range = np.linspace(minx, maxx, heatmap.shape[1])
        self.interpolator = RegularGridInterpolator((y_range, x_range), heatmap, bounds_error=False, fill_value=0)

    def calculate_likelihood_score(self, paths: list[LineString]) -> float:
        total_likelihood = 0
        for path in paths:
            if not path.is_empty and path.length > 0:
                points = [path.interpolate(d) for d in np.linspace(0, path.length, int(np.ceil(path.length)))]
                point_coords = [(p.y, p.x) for p in points]
                total_likelihood += np.sum(self.interpolator(point_coords))
        return total_likelihood

    def calculate_time_discounted_score(self, paths: list[LineString], discount_factor: float = 0.999) -> float:
        total_score = 0
        for path in paths:
            if not path.is_empty and path.length > 0:
                distances = np.linspace(0, path.length, int(np.ceil(path.length)))
                points = [path.interpolate(d) for d in distances]
                point_coords = [(p.y, p.x) for p in points]
                likelihoods = self.interpolator(point_coords)
                discounts = discount_factor ** distances
                total_score += np.sum(likelihoods * discounts)
        return total_score

    def calculate_victims_found_score(self, paths: list[LineString]) -> dict:
        # ... (no changes needed here)
        valid_paths = [p for p in paths if not p.is_empty]
        if not valid_paths or self.victims.empty: return {'percentage_found': 0, 'detection_timeliness': 0}
        coverage_area = gpd.GeoSeries(valid_paths, crs=self.victims.crs).buffer(self.detection_radius).union_all()
        found_victims = self.victims[self.victims.within(coverage_area)]
        percentage_found = (len(found_victims) / len(self.victims)) * 100 if not self.victims.empty else 0
        timeliness = []
        for _, victim in found_victims.iterrows():
            min_dist = min(path.project(victim.geometry) for path in valid_paths)
            avg_path_length = np.mean([p.length for p in valid_paths])
            if avg_path_length > 0: timeliness.append(min_dist / avg_path_length)
        return {'percentage_found': percentage_found, 'detection_timeliness': np.mean(timeliness) if timeliness else 0}

    def get_cumulative_likelihood_over_distance(self, path: LineString):
        """Calculates the cumulative likelihood score as a function of distance along a path."""
        if path.is_empty or path.length == 0:
            return np.array([0]), np.array([0])
        
        distances = np.linspace(0, path.length, int(np.ceil(path.length)))
        points = [path.interpolate(d) for d in distances]
        point_coords = [(p.y, p.x) for p in points]
        
        likelihoods = self.interpolator(point_coords)
        cumulative_likelihoods = np.cumsum(likelihoods)
        
        return distances, cumulative_likelihoods

    def get_cumulative_time_discounted_score_over_distance(self, path: LineString, discount_factor: float = 0.999):
        """Calculates the cumulative time-discounted score as a function of distance."""
        if path.is_empty or path.length == 0:
            return np.array([0]), np.array([0])

        distances = np.linspace(0, path.length, int(np.ceil(path.length)))
        points = [path.interpolate(d) for d in distances]
        point_coords = [(p.y, p.x) for p in points]
        
        likelihoods = self.interpolator(point_coords)
        discounts = discount_factor ** distances
        
        discounted_scores = likelihoods * discounts
        cumulative_discounted_scores = np.cumsum(discounted_scores)
        
        return distances, cumulative_discounted_scores

