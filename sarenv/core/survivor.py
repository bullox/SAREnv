# sarenv/core/survivor.py
"""
Generates plausible survivor locations based on geographic features.
"""
import random
import geopandas as gpd
from shapely.geometry import Point, Polygon

from sarenv.core.loading import SARDatasetItem
from sarenv.utils.logging_setup import get_logger

log = get_logger()

class SurvivorLocationGenerator:
    """
    Generates a plausible survivor location based on geographic features.
    """
    def __init__(self, dataset_item: SARDatasetItem):
        self.dataset_item = dataset_item
        self.features = dataset_item.features.copy()
        self.type_probabilities = {}
        self._calculate_weights()

    def _calculate_weights(self):
        if self.features.empty or 'area_probability' not in self.features.columns:
            log.warning("Features are empty or missing 'area_probability'. Cannot calculate weights.")
            return

        total_prob_in_subset = self.features['area_probability'].sum()
        if total_prob_in_subset > 0:
            self.features['renormalized_prob'] = self.features['area_probability'] / total_prob_in_subset
        else:
            self.features['renormalized_prob'] = 0
        
        type_weights = self.features.groupby('feature_type')['renormalized_prob'].sum()
        self.type_probabilities = type_weights.to_dict()
        log.info(f"Calculated feature type probabilities: {self.type_probabilities}")

    def _generate_random_point_in_polygon(self, poly: Polygon) -> Point:
        min_x, min_y, max_x, max_y = poly.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if poly.contains(random_point):
                return random_point

    def generate_location(self) -> Point | None:
        if not self.type_probabilities:
            log.error("No feature probabilities available. Cannot generate location.")
            return None

        chosen_type = random.choices(list(self.type_probabilities.keys()), weights=list(self.type_probabilities.values()), k=1)[0]
        type_gdf = self.features[self.features['feature_type'] == chosen_type]
        
        if type_gdf.empty or type_gdf['renormalized_prob'].sum() == 0:
            return None

        chosen_feature = type_gdf.sample(n=1, weights='renormalized_prob').iloc[0]
        feature_buffer = chosen_feature.geometry.buffer(30)
        
        center_proj = gpd.GeoDataFrame(geometry=[Point(self.dataset_item.center_point)], crs="EPSG:4326").to_crs(self.features.crs).geometry.iloc[0]
        main_search_circle = center_proj.buffer(self.dataset_item.radius_km * 1000)
        
        final_search_area = feature_buffer.intersection(main_search_circle)

        if final_search_area.is_empty:
             return None

        return self._generate_random_point_in_polygon(final_search_area)