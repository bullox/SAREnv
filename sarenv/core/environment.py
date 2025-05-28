# sarenv/core/environment.py
import concurrent.futures
import json
import time
from collections import defaultdict

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np

# from scipy.__config__ import show # This import seems unused and specific to scipy source
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
from PIL import Image  # Ensure Pillow is in requirements.txt
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Point, mapping
import geopandas as gpd  # Ensure geopandas is in requirements.txt
import cv2  # Ensure opencv-python is in requirements.txt
from shapely import is_empty, plotting as shplt
from skimage.draw import (
    polygon as ski_polygon,
)  # Ensure scikit-image is in requirements.txt

# Relative imports for modules within the same package
from ..utils import (
    logging_setup,
)  # Assuming get_logger is exposed in logging_setup.__init__ or directly
from ..utils import plot_utils  # Assuming plot_basemap is exposed here
from .geometries import GeoMultiPolygon, GeoMultiTrajectory, GeoPolygon, GeoPoint
from ..io.osm_query import query_features  # Corrected path

log = logging_setup.get_logger()


class EnvironmentBuilder:
    def __init__(self):
        self.polygon_file = None
        self.sample_distance = 1
        self.meter_per_bin = 3
        self.buffer = 0
        self.tags = {}

    def set_polygon_file(self, polygon_file):
        self.polygon_file = polygon_file
        return self

    def set_sample_distance(self, sample_distance):
        self.sample_distance = sample_distance
        return self

    def set_meter_per_bin(self, meter_per_bin):
        self.meter_per_bin = meter_per_bin
        return self

    def set_buffer(self, buffer):
        self.buffer = buffer
        return self

    def set_features(self, features):  # Renamed for consistency
        if not isinstance(features, dict):
            raise ValueError("Features must be a dictionary")
        self.tags.update(features)
        return self

    def set_feature(self, name, tags):
        self.tags[name] = tags
        return self

    def build(self):
        if self.polygon_file is None:
            raise ValueError(
                "Polygon file must be set before building the environment."
            )
        return Environment(
            self.polygon_file,
            self.sample_distance,
            self.meter_per_bin,
            self.buffer,
            self.tags,
        )


def image_to_world(
    x, y, meters_per_bin, minx, miny, buffer_val
):  # Renamed buffer to buffer_val
    x_world = x * meters_per_bin + minx - buffer_val
    y_world = y * meters_per_bin + miny - buffer_val
    return x_world, y_world


def world_to_image(
    x, y, meters_per_bin, minx, miny, buffer_val
):  # Renamed buffer to buffer_val
    x_img = (x - minx + buffer_val) / meters_per_bin
    y_img = (y - miny + buffer_val) / meters_per_bin
    return int(x_img), int(y_img)


class Environment:
    def __init__(
        self, polygon_file, sample_distance, meter_per_bin, buffer_val, tags
    ):  # Renamed buffer to buffer_val
        self.polygon_file = polygon_file
        self.tags = tags
        self.sample_distance = sample_distance
        self.meter_per_bin = meter_per_bin
        self.buffer_val = buffer_val  # Renamed buffer
        self.polygon: GeoPolygon | None = None
        self.xedges: np.ndarray | None = None
        self.yedges: np.ndarray | None = None
        self.heatmaps: dict[str, np.ndarray | None] = {}
        self.features: dict[str, gpd.GeoDataFrame | None] = (
            {}
        )  # Store features as GeoDataFrames

        with open(self.polygon_file, "r") as f:
            data = json.load(f)

        # Assuming the first feature in the GeoJSON is the boundary polygon
        # and its coordinates are in WGS84 (lon, lat)
        boundary_coords_lon_lat = data["features"][0]["geometry"]["coordinates"][0]
        query_region_wgs84 = shapely.Polygon(boundary_coords_lon_lat)
        log.info("Query region bounds (WGS84): %s", query_region_wgs84.bounds)

        # Initialize GeoPolygon with WGS84, then convert to projected CRS
        self.polygon = GeoPolygon(
            query_region_wgs84, crs="EPSG:4326"
        )  # Explicitly WGS84
        self.polygon.set_crs(
            "EPSG:2197"
        )  # Example projected CRS, choose appropriate one
        log.info(f"Environment polygon CRS set to: {self.polygon.crs}")

        self.area = self.polygon.geometry.area
        log.info(
            "Area of the polygon: %s m² (approx. %.2f km²)", self.area, self.area / 1e6
        )
        self.minx, self.miny, self.maxx, self.maxy = self.polygon.geometry.bounds

        # Ensure buffer_val doesn't make bins negative if bounds are small
        num_bins_x = int(
            abs(self.maxx - self.minx + 2 * self.buffer_val) / self.meter_per_bin
        )
        num_bins_y = int(
            abs(self.maxy - self.miny + 2 * self.buffer_val) / self.meter_per_bin
        )

        if num_bins_x <= 0:
            num_bins_x = 1
        if num_bins_y <= 0:
            num_bins_y = 1

        log.info("Number of bins x: %i y: %i", num_bins_x, num_bins_y)

        self.xedges = np.linspace(
            self.minx - self.buffer_val, self.maxx + self.buffer_val, num_bins_x + 1
        )
        self.yedges = np.linspace(
            self.miny - self.buffer_val, self.maxy + self.buffer_val, num_bins_y + 1
        )

        self._load_features()

    def _load_features(self):
        # Prepare a GeoPolygon in WGS84 for querying OSMnx
        query_polygon_wgs84 = GeoPolygon(self.polygon.geometry, crs=self.polygon.crs)
        query_polygon_wgs84.set_crs("EPSG:4326")  # Convert to WGS84 for query

        def process_feature_osm(key_val_pair):
            key, tag_dict = key_val_pair
            # query_features now expects GeoPolygon directly
            osm_geometries = query_features(query_polygon_wgs84, tag_dict)

            if osm_geometries is None:
                log.warning(
                    f"No geometries returned from OSM query for features: {key}"
                )
                return key, None

            feature_collection = []
            for (
                tag_specific_geom
            ) in osm_geometries.values():  # Iterate through dict values
                if tag_specific_geom is None or is_empty(tag_specific_geom):
                    continue
                if hasattr(tag_specific_geom, "geoms"):  # MultiGeometry
                    feature_collection.extend(
                        geom
                        for geom in tag_specific_geom.geoms
                        if geom is not None and not is_empty(geom)
                    )
                elif isinstance(tag_specific_geom, shapely.geometry.base.BaseGeometry):
                    if not is_empty(tag_specific_geom):
                        feature_collection.append(tag_specific_geom)

            if not feature_collection:
                log.info(
                    f"No valid geometries found for feature type '{key}' after filtering empty ones."
                )
                return key, None

            gdf_wgs84 = gpd.GeoDataFrame(geometry=feature_collection, crs="EPSG:4326")
            gdf_projected = gdf_wgs84.to_crs(
                self.polygon.crs
            )  # Convert to environment's CRS
            log.info(
                f"Processed {len(gdf_projected)} geometries for feature type '{key}'"
            )
            return key, gdf_projected

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # executor.map will return results in the order of submission
            results = executor.map(process_feature_osm, self.tags.items())
            for key, feature_gdf in results:
                self.features[key] = feature_gdf
                if feature_gdf is not None:
                    log.info(
                        f"Stored {len(feature_gdf)} features for '{key}' in CRS {feature_gdf.crs}"
                    )
                else:
                    log.info(f"No features stored for '{key}'")

    def image_to_world(self, x, y):
        return image_to_world(
            x, y, self.meter_per_bin, self.minx, self.miny, self.buffer_val
        )

    def world_to_image(self, x, y):
        return world_to_image(
            x, y, self.meter_per_bin, self.minx, self.miny, self.buffer_val
        )

    def interpolate_line(self, line, distance):
        # Ensure distance is positive
        if distance <= 0:
            # Return start and end points if distance is non-positive
            return [shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])]

        num_points = int(line.length / distance)
        points = []
        if (
            num_points > 0
        ):  # Changed from num_points != 0 to handle line.length < distance
            points = [
                line.interpolate(float(i) / num_points, normalized=True)
                for i in range(num_points + 1)
            ]
        else:  # If line is shorter than distance, or num_points is 0
            points = [
                shapely.Point(line.coords[0]),
                shapely.Point(line.coords[-1]),
            ]  # Start and end
        return points

    def generate_heatmaps(self):
        log.info("Generating heatmaps for all features...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {
                executor.submit(
                    self.generate_heatmap,
                    key,
                    feature_gdf.geometry,
                    self.sample_distance,
                ): key
                for key, feature_gdf in self.features.items()
                if feature_gdf is not None and not feature_gdf.empty
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    heatmap = future.result()
                    self.heatmaps[key] = heatmap
                    log.info(f"Generated heatmap for {key}")
                except Exception as exc:
                    log.error(f"Error generating heatmap for {key}: {exc}")
                    self.heatmaps[key] = None  # Store None if error
        log.info("Heatmap generation complete.")

    def generate_heatmap(
        self,
        feature_key: str,
        geometry_series: gpd.GeoSeries,
        sample_distance: float,
        infill_geometries=True,
    ):
        log.debug(
            f"Generating heatmap for feature: {feature_key} with {len(geometry_series)} geometries."
        )
        # Ensure xedges and yedges are initialized
        if self.xedges is None or self.yedges is None:
            log.error("Heatmap edges not initialized.")
            raise ValueError("Heatmap edges (xedges, yedges) must be initialized.")

        heatmap = np.zeros(
            (len(self.yedges) - 1, len(self.xedges) - 1), dtype=float
        )  # Flipped for (row, col) / (y,x)

        for geometry in geometry_series:
            if geometry is None or geometry.is_empty:
                continue

            interpolated_img_coords_x = []
            interpolated_img_coords_y = []

            if isinstance(geometry, LineString):
                points_on_line = self.interpolate_line(geometry, sample_distance)
                for p in points_on_line:
                    ix, iy = self.world_to_image(p.x, p.y)
                    interpolated_img_coords_x.append(ix)
                    interpolated_img_coords_y.append(iy)

            elif isinstance(geometry, shapely.geometry.Polygon):
                if infill_geometries:  # Infill polygon
                    # Convert exterior coordinates to image space for ski_polygon
                    ext_coords_world = np.array(list(geometry.exterior.coords))
                    ext_coords_img_x, ext_coords_img_y = self.world_to_image(
                        ext_coords_world[:, 0], ext_coords_world[:, 1]
                    )

                    # skimage.draw.polygon expects (row, col) which is (y, x) for image
                    rr, cc = ski_polygon(
                        ext_coords_img_y, ext_coords_img_x, shape=heatmap.shape
                    )
                    interpolated_img_coords_y.extend(rr)
                    interpolated_img_coords_x.extend(cc)
                else:  # Only outline
                    points_on_exterior = self.interpolate_line(
                        geometry.exterior, sample_distance
                    )
                    for p in points_on_exterior:
                        ix, iy = self.world_to_image(p.x, p.y)
                        interpolated_img_coords_x.append(ix)
                        interpolated_img_coords_y.append(iy)

                # Process interiors (as outlines)
                for interior in geometry.interiors:
                    points_on_interior = self.interpolate_line(
                        interior, sample_distance
                    )
                    for p in points_on_interior:
                        ix, iy = self.world_to_image(p.x, p.y)
                        interpolated_img_coords_x.append(ix)
                        interpolated_img_coords_y.append(iy)
            else:
                log.warning(
                    f"Unsupported geometry type for heatmap: {type(geometry)} for feature {feature_key}"
                )
                continue

            # Filter coordinates to be within heatmap bounds
            valid_indices = [
                i
                for i, (x, y) in enumerate(
                    zip(interpolated_img_coords_x, interpolated_img_coords_y)
                )
                if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]
            ]

            if valid_indices:
                valid_x = np.array(interpolated_img_coords_x)[valid_indices]
                valid_y = np.array(interpolated_img_coords_y)[valid_indices]
                # Increment heatmap cells; ensures each geometry contributes at most 1 to a cell it covers
                # For lines this is fine. For filled polygons, this effectively marks presence.
                heatmap[valid_y, valid_x] = 1

        # Normalize if there's any sum to avoid division by zero
        # Note: this normalization makes it a probability distribution *over the bins*.
        # If you want density (value per unit area), normalization needs to consider bin area.
        current_sum = np.sum(heatmap)
        if current_sum > 0:
            heatmap /= current_sum
        else:
            log.warning(
                f"Heatmap for {feature_key} is all zeros, no normalization performed."
            )
        return heatmap

    def get_combined_heatmap(self, sigma_features=None, alpha_features=None):
        # Ensure individual heatmaps are generated if not already present
        if not self.heatmaps or any(h is None for h in self.heatmaps.values()):
            self.generate_heatmaps()  # This now populates self.heatmaps

        if self.xedges is None or self.yedges is None:
            log.error("Cannot combine heatmaps, xedges or yedges not initialized.")
            return None  # Or raise an error

        # Initialize combined heatmap based on xedges and yedges
        # Swapped dimensions to (rows, cols) which is (y_bins, x_bins)
        combined_heatmap = np.zeros(
            (len(self.yedges) - 1, len(self.xedges) - 1), dtype=float
        )

        if sigma_features is None:
            sigma_features = {key: 1.0 for key in self.tags.keys()}
        if alpha_features is None:
            alpha_features = {key: 1.0 for key in self.tags.keys()}

        for key, individual_heatmap in self.heatmaps.items():
            if individual_heatmap is None:
                log.warning(
                    f"Skipping feature '{key}' in combined heatmap as its individual heatmap is None."
                )
                continue
            if individual_heatmap.shape != combined_heatmap.shape:
                log.error(
                    f"Shape mismatch for '{key}': {individual_heatmap.shape} vs {combined_heatmap.shape}"
                )
                continue

            sigma = sigma_features.get(key, 1.0)
            alpha = alpha_features.get(key, 1.0)

            # Apply Gaussian filter (if sigma > 0) and alpha weighting
            # Ensure individual_heatmap is float for gaussian_filter
            filtered_heatmap_part = individual_heatmap.astype(float) * alpha
            if sigma > 0:
                filtered_heatmap_part = gaussian_filter(
                    filtered_heatmap_part, sigma=sigma
                )

            combined_heatmap += filtered_heatmap_part
        log.info("Combined heatmap generated.")
        return combined_heatmap

    def binary_cut(
        self, lines: list[LineString], max_length: float
    ) -> list[LineString]:
        result = []
        processing_lines = list(lines)  # Create a copy to modify
        while processing_lines:
            line = processing_lines.pop(0)
            if line.length > max_length:
                # self.cut returns two LineStrings or (LineString, empty LineString)
                part1, part2 = self.cut(line, line.length / 2)
                if part1 and not part1.is_empty:
                    processing_lines.append(part1)
                if part2 and not part2.is_empty:
                    processing_lines.append(part2)
            else:
                if line and not line.is_empty:
                    result.append(line)
        return result

    def modulus_cut(
        self, lines: list[LineString], max_length: float
    ) -> list[LineString]:
        result = []
        processing_lines = list(lines)  # Create a copy
        while processing_lines:
            line = processing_lines.pop(0)
            if line.length > max_length:
                num_segments = int(np.ceil(line.length / max_length))
                if (
                    num_segments <= 1
                ):  # Avoid issues if max_length is very large or line is short
                    if line and not line.is_empty:
                        result.append(line)
                    continue

                segment_length = line.length / num_segments
                current_line = line
                for _ in range(num_segments - 1):  # Cut num_segments - 1 times
                    part1, part2 = self.cut(current_line, segment_length)
                    if part1 and not part1.is_empty:
                        result.append(part1)
                    current_line = part2
                    if not current_line or current_line.is_empty:
                        break
                if current_line and not current_line.is_empty:  # Add the last segment
                    result.append(current_line)
            else:
                if line and not line.is_empty:
                    result.append(line)
        return result

    def cut(
        self, line: LineString, distance: float
    ) -> tuple[LineString | None, LineString | None]:
        if not isinstance(line, LineString) or line.is_empty:
            return None, None
        if distance <= 0 or distance >= line.length:  # Cut at start or beyond end
            return line if distance <= 0 else None, None if distance <= 0 else line

        coords = list(line.coords)
        current_dist = 0.0
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            segment = LineString([p1, p2])
            if current_dist + segment.length >= distance:
                # Cut point is on this segment
                remaining_dist_on_segment = distance - current_dist
                cut_point_geom = segment.interpolate(remaining_dist_on_segment)

                first_part_coords = coords[: i + 1] + [
                    (cut_point_geom.x, cut_point_geom.y)
                ]
                second_part_coords = [(cut_point_geom.x, cut_point_geom.y)] + coords[
                    i + 1 :
                ]

                line1 = (
                    LineString(first_part_coords)
                    if len(first_part_coords) >= 2
                    else None
                )
                line2 = (
                    LineString(second_part_coords)
                    if len(second_part_coords) >= 2
                    else None
                )
                return line1, line2
            current_dist += segment.length
        return line, None  # Should not be reached if distance < line.length

    def shrink_polygon(
        self, p: shapely.Polygon | shapely.MultiPolygon, buffer_size: float
    ) -> list[shapely.Polygon | shapely.MultiPolygon]:
        if p.is_empty:
            return []

        shrunken_polygon = p.buffer(
            -buffer_size, join_style="mitre", single_sided=True
        )  # Using mitre for sharper corners sometimes

        if shrunken_polygon.is_empty:
            # Try a smaller buffer if the first attempt failed
            shrunken_polygon_half = p.buffer(
                -buffer_size * 0.5, join_style="mitre", single_sided=True
            )
            if not shrunken_polygon_half.is_empty:
                # If polygon is too thin, it might disappear. Return original and half-shrunken.
                return (
                    [p, *self.shrink_polygon(shrunken_polygon_half, buffer_size * 0.5)]
                    if buffer_size * 0.5 > 1
                    else [p]
                )  # Avoid infinite recursion
            else:
                return [p]  # Cannot shrink further, return original

        # Recursively shrink the new polygon
        # Ensure buffer_size for recursion is meaningful or stop
        if (
            buffer_size > 1
        ):  # Threshold to prevent excessive recursion with tiny buffers
            return [p, *self.shrink_polygon(shrunken_polygon, buffer_size)]
        else:
            return [p, shrunken_polygon] if not shrunken_polygon.is_empty else [p]

    def create_polygons_from_contours(
        self, contours: list[np.ndarray], hierarchy: np.ndarray, min_area_world: float
    ) -> list[shapely.Polygon]:
        # min_area_world is area in world coordinates (e.g., m^2)
        all_polygons = []
        if (
            hierarchy is None or hierarchy.ndim != 2 or hierarchy.shape[0] != 1
        ):  # OpenCV returns hierarchy as [[[next, prev, child, parent]]]
            log.warning("Unexpected hierarchy format or no hierarchy found.")
            # Process all contours as external if no hierarchy
            for cnt in contours:
                if cnt.shape[0] < 3:
                    continue  # Not enough points for a polygon
                world_coords = np.array(
                    [self.image_to_world(pt[0][0], pt[0][1]) for pt in cnt]
                )
                poly = shapely.Polygon(world_coords)
                if poly.area >= min_area_world:
                    all_polygons.append(poly)
            return all_polygons

        # Valid hierarchy: hierarchy is a 3D array with shape (1, num_contours, 4)
        # For each contour i, hierarchy[0][i] is [Next, Previous, First_Child, Parent]
        # Create a mapping of parent_idx to list of its child contours' points
        parent_to_children_coords = defaultdict(list)
        for i, h_info in enumerate(hierarchy[0]):
            parent_idx = h_info[3]  # Parent index
            if parent_idx != -1:  # If it has a parent, it's an interior (hole)
                child_contour_img = contours[i]
                if child_contour_img.shape[0] < 3:
                    continue
                child_coords_world = np.array(
                    [
                        self.image_to_world(pt[0][0], pt[0][1])
                        for pt in child_contour_img
                    ]
                )
                # Check if child contour forms a valid polygon and has some area before considering it a hole
                temp_hole_poly = shapely.Polygon(child_coords_world)
                if (
                    temp_hole_poly.is_valid and temp_hole_poly.area > EPS
                ):  # EPS is a small epsilon
                    parent_to_children_coords[parent_idx].append(child_coords_world)

        # Create polygons: iterate through contours, if it's not a child (i.e., it's an exterior), form polygon with its holes
        for i, cnt_ext_img in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # It's an exterior contour (has no parent)
                if cnt_ext_img.shape[0] < 3:
                    continue

                exterior_coords_world = np.array(
                    [self.image_to_world(pt[0][0], pt[0][1]) for pt in cnt_ext_img]
                )
                holes_coords_world = parent_to_children_coords.get(
                    i, []
                )  # Get holes for this exterior

                try:
                    poly = shapely.Polygon(
                        shell=exterior_coords_world, holes=holes_coords_world
                    )
                    if poly.is_valid and poly.area >= min_area_world:
                        all_polygons.append(poly)
                except Exception as e:
                    log.error(
                        f"Error creating polygon from contour {i}: {e}. Exterior: {len(exterior_coords_world)} pts, Holes: {len(holes_coords_world)} lists of pts."
                    )
        return all_polygons

    def informative_coverage(
        self,
        heatmap: np.ndarray,
        sensor_radius_world: float = 8.0,  # Sensor radius in world units (meters)
        max_path_length_world: float = 500.0,  # Max length of a single path segment in world units
        contour_smoothing_world: float = 1.0,  # Simplification tolerance in world units
        contour_threshold: float = 0.00001,  # Threshold on the heatmap values
    ) -> list[LineString]:
        start_time = time.time()
        if heatmap is None or np.sum(heatmap) == 0:
            log.warning(
                "Heatmap is empty or all zeros. Cannot generate informative coverage."
            )
            return []

        # 1. Threshold the heatmap to create a binary mask
        # Heatmap is (y_bins, x_bins) which is (rows, cols)
        mask = np.zeros_like(heatmap, dtype=np.uint8)
        mask[heatmap > contour_threshold] = 255  # Use 255 for findContours

        # 2. Find contours in the binary mask (image coordinates)
        # Input to findContours should be (rows, cols) or (height, width)
        # OpenCV expects mask.T if heatmap was (x_bins, y_bins)
        # If heatmap is already (y_bins, x_bins), no transpose needed for findContours
        contours_img, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE  # Using mask directly
        )
        if not contours_img:
            log.warning("No contours found in the heatmap above the threshold.")
            return []
        log.info(f"Found {len(contours_img)} raw contours in the heatmap.")

        # 3. Convert contours to world coordinate polygons, considering hierarchy for holes
        # Min area for a contour to be considered, in world units squared (e.g., m^2)
        # This helps filter out tiny noise contours. Consider sensor footprint.
        min_contour_area_world = (
            np.pi * (sensor_radius_world * 0.5) ** 2
        )  # e.g., area of a circle with half sensor radius

        all_polygons_world = self.create_polygons_from_contours(
            contours_img, hierarchy, min_contour_area_world
        )
        log.info(f"Created {len(all_polygons_world)} valid polygons from contours.")
        if not all_polygons_world:
            return []

        # 4. Simplify and shrink polygons to create coverage paths
        coverage_rings_world = []  # These will be LineStrings (polygon exteriors)
        for poly_world in all_polygons_world:
            if poly_world.is_empty:
                continue
            # Simplify the polygon in world coordinates
            simplified_poly = poly_world.simplify(
                contour_smoothing_world, preserve_topology=True
            )
            if simplified_poly.is_empty or not isinstance(
                simplified_poly, (shapely.Polygon, shapely.MultiPolygon)
            ):
                continue

            # Handle MultiPolygons that might result from simplification
            polys_to_shrink = []
            if isinstance(simplified_poly, shapely.Polygon):
                polys_to_shrink.append(simplified_poly)
            elif isinstance(simplified_poly, shapely.MultiPolygon):
                polys_to_shrink.extend(list(simplified_poly.geoms))

            for p_shrink in polys_to_shrink:
                if p_shrink.area < min_contour_area_world / 2:
                    continue  # Filter small remnants after simplify
                # Shrink polygon iteratively to get multiple coverage rings (offsetting)
                # The shrink_polygon method needs to be robust.
                # It should return a list of polygons, from original to most shrunken.
                # The "rings" are then the exteriors of these.
                shrunk_polygons_list = self.shrink_polygon(
                    p_shrink, sensor_radius_world * 2.0
                )  # Buffer by diameter
                for sp in shrunk_polygons_list:
                    if (
                        sp
                        and not sp.is_empty
                        and isinstance(sp, shapely.Polygon)
                        and sp.exterior
                    ):
                        coverage_rings_world.append(
                            sp.exterior
                        )  # Add exterior LineString
        log.info(
            f"Generated {len(coverage_rings_world)} coverage rings after shrinking."
        )
        if not coverage_rings_world:
            return []

        # 5. Orient, sort, and cut coverage rings into manageable path segments
        paths_world = []
        for ring in coverage_rings_world:
            if ring.length < EPS:
                continue
            # Ensure consistent orientation (e.g., CCW for exterior-like paths)
            oriented_ring = LineString(
                list(ring.coords)
            )  # Re-create to ensure simple LineString
            # if not oriented_ring.is_ccw: # This check is for Polygons, not LineStrings directly for path order
            #    oriented_ring = LineString(list(oriented_ring.coords)[::-1])

            # Reorder vertices of the ring so the "start" is consistent (e.g., closest to origin or min X/Y)
            # This helps in making the cutting more deterministic if needed.
            coords = list(oriented_ring.coords)
            if not coords:
                continue
            # Sort by min-X, then min-Y to have a somewhat consistent start for cutting
            # This is a simple heuristic.
            min_idx = min(
                range(len(coords) - 1), key=lambda i: (coords[i][0], coords[i][1])
            )
            # Create a closed ring for cutting, then open it up
            # For cutting, an open line is better. The last point is same as first.
            reordered_coords = (
                coords[min_idx:-1] + coords[: min_idx + 1]
            )  # Forms a closed loop starting at min_idx
            open_line_for_cutting = LineString(reordered_coords)

            if open_line_for_cutting.length > max_path_length_world:
                # Cut the line into segments of max_path_length_world
                # modulus_cut expects a list of lines.
                paths_world.extend(
                    self.modulus_cut([open_line_for_cutting], max_path_length_world)
                )
            else:
                paths_world.append(open_line_for_cutting)

        # 6. Final processing: offset paths slightly (first segment often too close to start)
        # This was in your original code: cut sensor_radius from the start of each line.
        final_paths = []
        for line in paths_world:
            if line.length > sensor_radius_world:
                _, path_segment = self.cut(
                    line, sensor_radius_world
                )  # Get the second part
                if path_segment and not path_segment.is_empty:
                    final_paths.append(path_segment)
            # elif line.length > EPS: # Keep very short lines if they are meaningful
            #     final_paths.append(line)

        end_time = time.time()
        log.info(
            f"Informative coverage: {end_time - start_time:.2f}s, {len(final_paths)} paths."
        )
        return final_paths

    def plot(
        self,
        show_basemap=True,
        show_features=False,
        show_heatmap=True,
        show_coverage=False,
        combined_heatmap_override=None,  # Allow passing a pre-computed heatmap
        coverage_paths_override=None,  # Allow passing pre-computed paths
    ):
        fig, ax = plt.subplots(figsize=(12, 12))  # Larger figure
        ax.set_xlim(self.minx - self.buffer_val, self.maxx + self.buffer_val)
        ax.set_ylim(self.miny - self.buffer_val, self.maxy + self.buffer_val)
        ax.set_aspect("equal", adjustable="box")  # Ensure correct aspect ratio

        current_alpha = 1.0  # Default alpha

        if show_basemap:
            if self.polygon and self.polygon.crs:
                plot_utils.plot_basemap(
                    ax=ax,
                    source=cx.providers.OpenStreetMap.Mapnik,
                    crs=self.polygon.crs,
                )
                current_alpha = 0.5  # Make subsequent layers somewhat transparent
            else:
                log.warning("Cannot show basemap, environment polygon or CRS not set.")

        if show_features:
            if self.polygon:
                self.polygon.plot(
                    ax=ax,
                    linestyle="--",
                    facecolor="none",
                    edgecolor="black",
                    alpha=0.8,
                    linewidth=1.5,
                )
            feature_color_map = plt.cm.get_cmap(
                "viridis", len(self.features)
            )  # Example colormap
            for i, (key, feature_gdf) in enumerate(self.features.items()):
                if feature_gdf is not None and not feature_gdf.empty:
                    # Differentiate lines and polygons for plotting
                    lines_gdf = feature_gdf[
                        feature_gdf.geom_type.isin(["LineString", "MultiLineString"])
                    ]
                    polys_gdf = feature_gdf[
                        feature_gdf.geom_type.isin(["Polygon", "MultiPolygon"])
                    ]

                    if not lines_gdf.empty:
                        lines_gdf.plot(
                            ax=ax,
                            color=feature_color_map(i),
                            linestyle="-",
                            linewidth=1,
                            label=f"{key} (lines)",
                            alpha=current_alpha * 0.8,
                        )
                    if not polys_gdf.empty:
                        polys_gdf.plot(
                            ax=ax,
                            color=feature_color_map(i),
                            edgecolor=feature_color_map(i),
                            linewidth=0.5,
                            label=f"{key} (polys)",
                            alpha=current_alpha * 0.6,
                        )

        heatmap_to_plot = combined_heatmap_override
        if show_heatmap and heatmap_to_plot is None:
            # Default sigma and alpha, make these configurable if needed
            sigma_features = {key: 3.0 for key in self.tags.keys()}
            alpha_features = {key: 1.0 for key in self.tags.keys()}
            heatmap_to_plot = self.get_combined_heatmap(sigma_features, alpha_features)

        if show_heatmap and heatmap_to_plot is not None and np.sum(heatmap_to_plot) > 0:
            colors = [
                (1, 1, 1, 0),
                (0, 0, 1, 0.2),
                (0, 1, 1, 0.4),
                (0, 1, 0, 0.6),
                (1, 1, 0, 0.8),
                (1, 0, 0, 1),
            ]  # White (transparent) to Red
            custom_cmap = LinearSegmentedColormap.from_list("custom_fade", colors)

            extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]
            # Ensure heatmap is oriented correctly for imshow (origin='lower' means (0,0) is bottom-left)
            # If heatmap is (y_bins, x_bins), no transpose is needed.
            im = ax.imshow(
                heatmap_to_plot,
                extent=extent,
                origin="lower",
                cmap=custom_cmap,
                alpha=current_alpha,
                interpolation="bilinear",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar
            cbar.set_label("Probability Density")

        paths_to_plot = coverage_paths_override
        if show_coverage and paths_to_plot is None and heatmap_to_plot is not None:
            # Default sensor radius, make this configurable
            sensor_radius = 8.0
            paths_to_plot = self.informative_coverage(
                heatmap_to_plot, sensor_radius_world=sensor_radius
            )

        if show_coverage and paths_to_plot:
            coverage_multiline = GeoMultiTrajectory(
                paths_to_plot, crs=self.polygon.crs if self.polygon else "EPSG:2197"
            )
            coverage_multiline.plot(
                ax=ax, color="magenta", linewidth=1.5, alpha=0.9, label="Coverage Paths"
            )

            # Optional: Show sensor footprint along paths
            # This can be computationally intensive for many/long paths
            # for line in paths_to_plot:
            #    buffered_path = line.buffer(sensor_radius, cap_style='round')
            #    gpd.GeoSeries([buffered_path]).plot(ax=ax, color='magenta', alpha=0.1, edgecolor='none')

        ax.set_title("SAR Environment Overview")
        ax.set_xlabel("X Coordinate (meters)")
        ax.set_ylabel("Y Coordinate (meters)")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Legend outside plot
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend
        plt.grid(True, linestyle=":", alpha=0.5)
        # Save before showing if needed
        # plt.savefig("sarenv_plot.png", bbox_inches="tight", dpi=300)
        plt.show()

    def visualise_environment(self):
        """Visualize the environment with different color coding for each feature type"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal", adjustable="box")

        if self.polygon:
            self.polygon.plot(
                ax=ax,
                facecolor="lightgrey",
                edgecolor="black",
                alpha=0.5,
                label="Search Boundary",
            )

        # Define color mapping for each feature type
        # Using a colormap might be better for many feature types
        num_features = len(self.features)
        colors = plt.cm.get_cmap(
            "tab20", num_features if num_features > 0 else 1
        )  # tab20 is good for distinct colors

        legend_handles = []

        for i, (feature_type, gdf) in enumerate(self.features.items()):
            if gdf is not None and not gdf.empty:
                color = colors(
                    i / num_features if num_features > 0 else 0.5
                )  # Normalize index for colormap
                gdf.plot(
                    ax=ax,
                    label=feature_type.capitalize(),
                    color=color,
                    alpha=0.6,
                    linewidth=0.8,
                )
                legend_handles.append(
                    Patch(
                        facecolor=color,
                        edgecolor=color,  # Use same color for edge for simplicity
                        label=feature_type.capitalize(),
                        alpha=0.6,
                    )
                )
        if self.polygon:  # Add boundary to legend if plotted
            legend_handles.append(
                Patch(
                    facecolor="lightgrey",
                    edgecolor="black",
                    label="Search Boundary",
                    alpha=0.5,
                )
            )

        ax.set_title("Environment Features Visualization")
        ax.set_xlabel("X Coordinate (meters)")
        ax.set_ylabel("Y Coordinate (meters)")
        if legend_handles:
            ax.legend(
                handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1)
            )

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for legend
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.show()

    def plot_heatmap_interactive(  # Renamed from plot_heatmap
        self,
        initial_heatmap: np.ndarray,  # Pass the initially computed heatmap
        show_basemap=True,
        show_features=False,
        export_final_image=False,  # Changed from 'export'
        show_coverage_paths=True,  # Changed from 'show_coverage'
    ):
        fig, ax = plt.subplots(figsize=(12, 10))  # Main plot axis
        fig.subplots_adjust(left=0.1, bottom=0.35)  # Adjust for sliders

        ax.set_xlim(self.minx - self.buffer_val, self.maxx + self.buffer_val)
        ax.set_ylim(self.miny - self.buffer_val, self.maxy + self.buffer_val)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Interactive Heatmap and Coverage")

        alpha_basemap = 1.0
        if show_basemap and self.polygon and self.polygon.crs:
            plot_utils.plot_basemap(
                ax=ax, source=cx.providers.OpenStreetMap.Mapnik, crs=self.polygon.crs
            )
            alpha_basemap = 0.6  # Make heatmap/features semi-transparent over basemap

        if show_features and self.polygon:
            self.polygon.plot(
                ax=ax,
                linestyle="--",
                facecolor="none",
                edgecolor="black",
                alpha=alpha_basemap * 0.9,
                linewidth=1,
            )
            # Simplified feature plotting for interactive view
            # Consider plotting only a few key features or using very light colors
            for key, feature_gdf in self.features.items():
                if feature_gdf is not None and not feature_gdf.empty:
                    feature_gdf.plot(
                        ax=ax, label=key, alpha=alpha_basemap * 0.5, linewidth=0.5
                    )

        # Heatmap display setup
        colors = [
            (1, 1, 1, 0),
            (0, 0, 1, 0.2),
            (0, 1, 1, 0.4),
            (0, 1, 0, 0.6),
            (1, 1, 0, 0.8),
            (1, 0, 0, 1),
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom_fade", colors)
        extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

        heatmap_display_img = ax.imshow(
            initial_heatmap,  # Display the passed initial heatmap
            extent=extent,
            origin="lower",
            cmap=custom_cmap,
            alpha=alpha_basemap,
            interpolation="bilinear",
        )
        cbar = fig.colorbar(heatmap_display_img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Probability Density")
        if np.sum(initial_heatmap) > 0:
            cbar.mappable.set_clim(
                vmin=initial_heatmap.min(), vmax=initial_heatmap.max()
            )

        # Coverage path display (initially empty or based on initial heatmap)
        coverage_plot_elements = (
            []
        )  # To store artists for coverage paths for easy removal

        # Sliders setup
        slider_axes = []
        sigma_sliders = {}
        alpha_sliders = {}
        slider_start_y = 0.25
        slider_height = 0.02
        slider_spacing = 0.03

        if not self.heatmaps:
            self.generate_heatmaps()  # Ensure individual heatmaps exist

        sorted_feature_keys = sorted(self.heatmaps.keys())

        for i, key in enumerate(sorted_feature_keys):
            ax_sigma = fig.add_axes(
                [0.15, slider_start_y - i * slider_spacing, 0.3, slider_height]
            )
            sigma_sliders[key] = Slider(
                ax_sigma, f"{key} σ", 0.0, 10.0, valinit=3.0, valstep=0.1
            )
            slider_axes.append(ax_sigma)

            ax_alpha = fig.add_axes(
                [0.55, slider_start_y - i * slider_spacing, 0.3, slider_height]
            )
            alpha_sliders[key] = Slider(
                ax_alpha, f"{key} α", 0.0, 5.0, valinit=1.0, valstep=0.1
            )
            slider_axes.append(ax_alpha)

        ax_sensor_slider = fig.add_axes(
            [
                0.15,
                slider_start_y - (len(sorted_feature_keys)) * slider_spacing,
                0.3,
                slider_height,
            ]
        )
        sensor_radius_slider = Slider(
            ax_sensor_slider, "Sensor Radius (m)", 1.0, 50.0, valinit=8.0, valstep=0.5
        )
        slider_axes.append(ax_sensor_slider)

        ax_pathlen_slider = fig.add_axes(
            [
                0.55,
                slider_start_y - (len(sorted_feature_keys)) * slider_spacing,
                0.3,
                slider_height,
            ]
        )
        max_path_len_slider = Slider(
            ax_pathlen_slider,
            "Max Path Len (m)",
            50.0,
            2000.0,
            valinit=500.0,
            valstep=10,
        )
        slider_axes.append(ax_pathlen_slider)

        def update_plot(val=None):  # val is unused but required by on_changed
            nonlocal coverage_plot_elements  # Allow modification of this list

            current_sigmas = {key: s.val for key, s in sigma_sliders.items()}
            current_alphas = {key: a.val for key, a in alpha_sliders.items()}
            current_sensor_radius = sensor_radius_slider.val
            current_max_path_len = max_path_len_slider.val

            # Recalculate combined heatmap
            updated_combined_heatmap = self.get_combined_heatmap(
                current_sigmas, current_alphas
            )
            if updated_combined_heatmap is None:
                return  # Should not happen if self.heatmaps is good

            # Update heatmap display
            heatmap_display_img.set_data(
                updated_combined_heatmap
            )  # Transpose if needed based on your heatmap orientation
            if np.sum(updated_combined_heatmap) > 0:
                heatmap_display_img.set_clim(
                    vmin=updated_combined_heatmap.min(),
                    vmax=updated_combined_heatmap.max(),
                )
            else:  # Handle all-zero heatmap case for clim
                heatmap_display_img.set_clim(vmin=0, vmax=1)

            # Clear previous coverage paths
            for artist in coverage_plot_elements:
                artist.remove()
            coverage_plot_elements.clear()

            if show_coverage_paths:
                new_paths = self.informative_coverage(
                    updated_combined_heatmap,
                    sensor_radius_world=current_sensor_radius,
                    max_path_length_world=current_max_path_len,
                )
                if new_paths:
                    # Plotting GeoMultiTrajectory directly might add it to axes permanently
                    # Instead, plot each line and store the artists
                    for line in new_paths:
                        gpd_line = gpd.GeoSeries(
                            [line], crs=self.polygon.crs if self.polygon else None
                        )
                        # This returns a list of Line2D objects
                        plotted_artists = gpd_line.plot(
                            ax=ax, color="fuchsia", linewidth=1.2, alpha=0.9
                        )
                        coverage_plot_elements.extend(
                            plotted_artists
                        )  # Assuming plot returns a list of artists

            fig.canvas.draw_idle()

        for s in list(sigma_sliders.values()) + list(alpha_sliders.values()):
            s.on_changed(update_plot)
        sensor_radius_slider.on_changed(update_plot)
        max_path_len_slider.on_changed(update_plot)

        # Initial plot update
        update_plot()

        if export_final_image:
            ax_save_button = fig.add_axes([0.8, 0.01, 0.1, 0.04])
            save_button = plt.Button(ax_save_button, "Save Plot")

            def save_current_plot(event):
                # Temporarily hide sliders for cleaner save
                for sa in slider_axes:
                    sa.set_visible(False)
                ax_save_button.set_visible(False)
                fig.canvas.draw_idle()
                plt.savefig("interactive_plot_output.png", dpi=300, bbox_inches="tight")
                log.info("Plot saved to interactive_plot_output.png")
                for sa in slider_axes:
                    sa.set_visible(True)  # Restore
                ax_save_button.set_visible(True)
                fig.canvas.draw_idle()

            save_button.on_clicked(save_current_plot)

        plt.show()


def get_filter_sigma(filter_width_meters: float, env_instance: Environment) -> float:
    # Ensure env_instance is a valid Environment object with meter_per_bin
    if not isinstance(env_instance, Environment) or env_instance.meter_per_bin <= 0:
        raise ValueError(
            "Valid Environment instance with positive meter_per_bin is required."
        )

    filter_width_pixels = filter_width_meters / env_instance.meter_per_bin

    # Calculate the sigma value for the Gaussian filter
    # This formula relates Full Width at Half Maximum (FWHM) to sigma: FWHM = 2 * sqrt(2 * ln(2)) * sigma
    # Assuming filter_width_meters is intended to be something like FWHM.
    if filter_width_pixels <= 0:
        return 0.0  # No blur if width is zero or negative
    sigma = filter_width_pixels / (2 * np.sqrt(2 * np.log(2)))
    return sigma
