# sarenv/core/environment.py
import concurrent.futures
import json
import os
import time
from collections import defaultdict

import contextily as cx
import cv2  # Ensure opencv-python is in requirements.txt
import geopandas as gpd  # Ensure geopandas is in requirements.txt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Point
from skimage.draw import (
    polygon as ski_polygon,
)

from ..io.osm_query import query_features
from ..utils import (
    logging_setup,
    plot_utils,
)
from .geometries import GeoMultiTrajectory, GeoPolygon

log = logging_setup.get_logger()
EPS = 1e-9

class EnvironmentBuilder:
    def __init__(self):
        self.polygon = None
        self.meter_per_bin = 3
        self.sample_distance = 1
        self.buffer = 0
        self.tags = {}
        self.projected_crs = None # Add projected_crs attribute

    def set_polygon(self, polygon):
        self.polygon = polygon
        return self

    def set_sample_distance(self, sample_distance):
        self.sample_distance = sample_distance
        return self

    def set_meter_per_bin(self, meter_per_bin):
        self.meter_per_bin = meter_per_bin
        return self

    def set_projected_crs(self, crs: str):
        self.projected_crs = crs
        return self

    def set_buffer(self, buffer_val):
        self.buffer = buffer_val
        return self

    def set_features(self, features):
        if not isinstance(features, dict):
            raise ValueError("Features must be a dictionary")
        self.tags.update(features)
        return self

    def set_feature(self, name, tags):
        self.tags[name] = tags
        return self

    def build(self):
        if self.polygon is None:
            raise ValueError("Polygon must be set before building the environment.")
        if self.projected_crs is None:
            raise ValueError("Projected CRS must be set before building.")

        return Environment(
            self.polygon,
            self.sample_distance,
            self.meter_per_bin,
            self.buffer,
            self.tags,
            self.projected_crs, # Pass the CRS
        )
# Corrected global helper function
def world_to_image(x, y, meters_per_bin, minx, miny, buffer_val):
    # x and y can be NumPy arrays
    x_img = (x - minx + buffer_val) / meters_per_bin
    y_img = (y - miny + buffer_val) / meters_per_bin
    # Convert to int array, then return.
    # If x_img, y_img are single scalars, astype(int) also works.
    return x_img.astype(int), y_img.astype(int)


# This function is not strictly needed if Environment.world_to_image calls the global one directly
# but kept for structural consistency if preferred.
def image_to_world(x_img, y_img, meters_per_bin, minx, miny, buffer_val):
    x_world = x_img * meters_per_bin + minx - buffer_val
    y_world = y_img * meters_per_bin + miny - buffer_val
    return x_world, y_world


class Environment:
    def __init__(self, bounding_polygon, sample_distance, meter_per_bin, buffer_val, tags, projected_crs):
        self.tags = tags
        self.sample_distance = sample_distance
        self.meter_per_bin = meter_per_bin
        self.buffer_val = buffer_val
        self.projected_crs = projected_crs # Store the projected CRS

        self.polygon: GeoPolygon | None = None
        self.xedges: np.ndarray | None = None
        self.yedges: np.ndarray | None = None
        self.heatmaps: dict[str, np.ndarray | None] = {}
        self.features: dict[str, gpd.GeoDataFrame | None] = {}

        self.polygon = GeoPolygon(bounding_polygon, crs="EPSG:4326") # make sure it is WGS84
        self.polygon.set_crs(self.projected_crs) # Use the dynamically provided projected CRS
        log.info(f"Environment polygon CRS set to: {self.polygon.crs}")

        self.area = self.polygon.geometry.area
        log.info("Area of the polygon: %s m² (approx. %.2f km²)", self.area, self.area / 1e6)
        self.minx, self.miny, self.maxx, self.maxy = self.polygon.geometry.bounds

        num_bins_x = int(abs(self.maxx - self.minx + 2 * self.buffer_val) / self.meter_per_bin)
        num_bins_y = int(abs(self.maxy - self.miny + 2 * self.buffer_val) / self.meter_per_bin)

        if num_bins_x <= 0: num_bins_x = 1
        if num_bins_y <= 0: num_bins_y = 1

        log.info("Number of bins x: %i y: %i", num_bins_x, num_bins_y)

        self.xedges = np.linspace(
            self.minx - self.buffer_val, self.maxx + self.buffer_val, num_bins_x + 1
        )
        self.yedges = np.linspace(
            self.miny - self.buffer_val, self.maxy + self.buffer_val, num_bins_y + 1
        )

        self._load_features()


    def _load_features(self):
        query_polygon_wgs84 = GeoPolygon(self.polygon.geometry, crs=self.polygon.crs)
        query_polygon_wgs84.set_crs("EPSG:4326")  # Ensure the query polygon is in WGS84

        def process_feature_osm(key_val_pair):
            key, tag_dict = key_val_pair
            osm_geometries_dict = query_features(query_polygon_wgs84, tag_dict)


            if osm_geometries_dict is None: # query_features now returns a dict or None
                log.warning(f"No geometries returned from OSM query for features: {key}")
                return key, None # Return key and None for the GeoDataFrame

            all_geoms_for_key = []
            for geom in osm_geometries_dict.values():
                if geom is not None and not geom.is_empty:
                    if hasattr(geom, 'geoms'): # MultiGeometry
                        all_geoms_for_key.extend(g for g in geom.geoms if g is not None and not g.is_empty)
                    else:
                        all_geoms_for_key.append(geom)

            if not all_geoms_for_key:
                log.info(f"No valid geometries found for feature type '{key}' after filtering empty ones.")
                return key, None

            gdf_wgs84 = gpd.GeoDataFrame(geometry=all_geoms_for_key, crs="EPSG:4326")
            gdf_projected = gdf_wgs84.to_crs(self.polygon.crs)
            log.info(f"Processed {len(gdf_projected)} geometries for feature type '{key}'")
            return key, gdf_projected


        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {executor.submit(process_feature_osm, item): item[0] for item in self.tags.items()}
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    _, feature_gdf = future.result() # process_feature_osm returns (key, gdf)
                    self.features[key] = feature_gdf
                    if feature_gdf is not None:
                        log.info(f"Stored {len(feature_gdf)} features for '{key}' in CRS {feature_gdf.crs}")
                    else:
                        log.info(f"No features stored for '{key}'")
                except Exception as exc:
                    log.error(f"Error processing feature {key}: {exc}", exc_info=True)
                    self.features[key] = None


    # Instance method calling the global helper
    def image_to_world(self, x_img, y_img):
        return image_to_world(
            x_img, y_img, self.meter_per_bin, self.minx, self.miny, self.buffer_val
        )

    # Instance method calling the global helper
    def world_to_image(self, x_world, y_world):
        return world_to_image(
            x_world, y_world, self.meter_per_bin, self.minx, self.miny, self.buffer_val
        )

    def interpolate_line(self, line, distance):
        if distance <= 0:
            return [shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])]

        points = []
        for i in range(len(line.coords) - 1):
            segment = LineString([line.coords[i], line.coords[i + 1]])
            segment_length = segment.length
            num_points = max(1, int(segment_length / distance))
            points.extend(
                segment.interpolate(float(j) / num_points * segment_length)
                for j in range(num_points)
            )
        points.append(shapely.Point(line.coords[-1]))  # Ensure the last point is included
        return points

    def generate_heatmaps(self):
        log.info("Generating heatmaps for all features...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {
                # Pass feature_gdf.geometry (which is a GeoSeries)
                executor.submit(self.generate_heatmap, key, feature_gdf.geometry, self.sample_distance): key
                for key, feature_gdf in self.features.items()
                if feature_gdf is not None and not feature_gdf.empty
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    heatmap_result = future.result()
                    self.heatmaps[key] = heatmap_result
                    log.info(f"Generated heatmap for {key}")
                except Exception as exc:
                    log.error(f"Error generating heatmap for {key}: {exc}", exc_info=True) # Log full traceback
                    self.heatmaps[key] = None
        log.info("Heatmap generation complete.")


    def generate_heatmap(
        self, feature_key: str, geometry_series: gpd.GeoSeries, sample_distance: float, infill_geometries=True
    ):
        log.debug(f"Generating heatmap for feature: {feature_key} with {len(geometry_series)} geometries.")
        if self.xedges is None or self.yedges is None or self.meter_per_bin <=0:
            log.error("Heatmap edges or meter_per_bin not correctly initialized.")
            raise ValueError("Heatmap edges (xedges, yedges) and meter_per_bin must be initialized and positive.")

        heatmap = np.zeros((len(self.yedges) - 1, len(self.xedges) - 1), dtype=float)

        for geometry in geometry_series: # geometry_series is a GeoSeries of Shapely geometries
            if geometry is None or geometry.is_empty:
                continue

            current_geom_img_coords_x = []
            current_geom_img_coords_y = []

            if isinstance(geometry, LineString):
                points_on_line = self.interpolate_line(geometry, sample_distance)
                if points_on_line:
                    world_x = [p.x for p in points_on_line]
                    world_y = [p.y for p in points_on_line]
                    img_x, img_y = self.world_to_image(np.array(world_x), np.array(world_y))
                    current_geom_img_coords_x.extend(img_x)
                    current_geom_img_coords_y.extend(img_y)

            elif isinstance(geometry, shapely.geometry.Polygon):
                if infill_geometries:
                    ext_coords_world = np.array(list(geometry.exterior.coords))
                    ext_coords_img_x_arr, ext_coords_img_y_arr = self.world_to_image(
                        ext_coords_world[:,0], ext_coords_world[:,1]
                    )
                    # skimage.draw.polygon expects (row, col) which is (y, x)
                    rr, cc = ski_polygon(ext_coords_img_y_arr, ext_coords_img_x_arr, shape=heatmap.shape)
                    current_geom_img_coords_y.extend(rr)
                    current_geom_img_coords_x.extend(cc)
                else: # Only outline
                    points_on_exterior = self.interpolate_line(geometry.exterior, sample_distance)
                    if points_on_exterior:
                        world_x = [p.x for p in points_on_exterior]
                        world_y = [p.y for p in points_on_exterior]
                        img_x, img_y = self.world_to_image(np.array(world_x), np.array(world_y))
                        current_geom_img_coords_x.extend(img_x)
                        current_geom_img_coords_y.extend(img_y)

                for interior in geometry.interiors:
                    interior_coords_world = np.array(list(interior.coords))
                    interior_coords_img_x, interior_coords_img_y = self.world_to_image(
                        interior_coords_world[:, 0], interior_coords_world[:, 1]
                    )
                    for ix, iy in zip(interior_coords_img_x, interior_coords_img_y):
                        if ix in current_geom_img_coords_x and iy in current_geom_img_coords_y:
                            idx = current_geom_img_coords_x.index(ix)
                            if current_geom_img_coords_y[idx] == iy:
                                current_geom_img_coords_x.pop(idx)
                                current_geom_img_coords_y.pop(idx)
            else:
                log.warning(f"Unsupported geometry type for heatmap: {type(geometry)} for feature {feature_key}")
                continue

            if current_geom_img_coords_x: # If any points were generated
                valid_indices = [
                    i for i, (x, y) in enumerate(zip(current_geom_img_coords_x, current_geom_img_coords_y))
                    if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]
                ]
                if valid_indices:
                    valid_x = np.array(current_geom_img_coords_x)[valid_indices]
                    valid_y = np.array(current_geom_img_coords_y)[valid_indices]
                    heatmap[valid_y, valid_x] = 1 # Mark presence

        return heatmap

    def get_combined_heatmap(self, sigma_features=None, alpha_features=None):
        if not self.heatmaps: # If heatmaps dict is empty
             log.info("Individual heatmaps not generated yet. Generating them now.")
             self.generate_heatmaps()
        # Check if any heatmaps were actually generated
        if not any(h is not None for h in self.heatmaps.values()):
            log.warning("No individual heatmaps available to combine. Returning zero map.")
            if self.xedges is None or self.yedges is None: return None # Cannot determine shape
            return np.zeros((len(self.yedges) - 1, len(self.xedges) - 1), dtype=float)


        if self.xedges is None or self.yedges is None:
            log.error("Cannot combine heatmaps, xedges or yedges not initialized.")
            return None

        combined_heatmap = np.zeros((len(self.yedges) - 1, len(self.xedges) - 1), dtype=float)
        if alpha_features is None:
            alpha_features = {
            "linear": 0.25,
            "field": 0.14,
            "structure": 0.13,
            "road": 0.13,
            "drainage": 0.12,
            "water": 0.08,
            "brush": 0.02,
            "scrub": 0.03,
            "woodland": 0.07,
            "rock": 0.04,
            }
        default_sigma = sigma_features if isinstance(sigma_features, (int, float)) else 0
        default_alpha = alpha_features if isinstance(alpha_features, (int, float)) else 0

        sigma_map = sigma_features if isinstance(sigma_features, dict) else {key: default_sigma for key in self.tags.keys()}
        alpha_map = alpha_features if isinstance(alpha_features, dict) else {key: default_alpha for key in self.tags.keys()}

        for key, individual_heatmap in self.heatmaps.items():
            if individual_heatmap is None:
                log.warning(f"Skipping feature '{key}' in combined heatmap as its individual heatmap is None.")
                continue
            if individual_heatmap.shape != combined_heatmap.shape:
                log.error(f"Shape mismatch for '{key}': {individual_heatmap.shape} vs {combined_heatmap.shape}")
                continue

            sigma = sigma_map.get(key, default_sigma)
            alpha = alpha_map.get(key, default_alpha)

            filtered_heatmap_part = individual_heatmap.astype(float) * alpha
            if sigma > 0:
                filtered_heatmap_part = gaussian_filter(filtered_heatmap_part, sigma=sigma)

            combined_heatmap = np.maximum(combined_heatmap, filtered_heatmap_part)
        return combined_heatmap

    def binary_cut(self, lines: list[LineString], max_length: float) -> list[LineString]:
        result = []
        processing_lines = list(lines)
        while processing_lines:
            line = processing_lines.pop(0)
            if line.length > max_length:
                part1, part2 = self.cut(line, line.length / 2)
                if part1 and not part1.is_empty: processing_lines.append(part1)
                if part2 and not part2.is_empty: processing_lines.append(part2)
            else:
                if line and not line.is_empty: result.append(line)
        return result

    def modulus_cut(self, lines: list[LineString], max_length: float) -> list[LineString]:
        result = []
        processing_lines = list(lines)
        while processing_lines:
            line = processing_lines.pop(0)
            if line.length > max_length:
                num_segments = int(np.ceil(line.length / max_length))
                if num_segments <= 1:
                    if line and not line.is_empty: result.append(line)
                    continue

                segment_length = line.length / num_segments
                current_line = line
                for _ in range(num_segments -1):
                    part1, part2 = self.cut(current_line, segment_length)
                    if part1 and not part1.is_empty: result.append(part1)
                    current_line = part2
                    if not current_line or current_line.is_empty: break
                if current_line and not current_line.is_empty:
                    result.append(current_line)
            else:
                if line and not line.is_empty: result.append(line)
        return result

    def cut(self, line: LineString, distance: float) -> tuple[LineString | None, LineString | None]:
        if not isinstance(line, LineString) or line.is_empty:
            return None, None
        if distance <= 1e-6 : # effectively zero or negative distance
            return None, line
        if distance >= line.length - 1e-6: # distance is at or beyond the line length
            return line, None

        coords = list(line.coords)
        current_dist_along_line = 0.0
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            segment = LineString([p1, p2])
            segment_len = segment.length

            if current_dist_along_line + segment_len >= distance - 1e-6: # Use tolerance
                # Cut point is on this segment
                remaining_dist_on_segment = distance - current_dist_along_line
                cut_point_geom = segment.interpolate(remaining_dist_on_segment)

                first_part_coords = coords[:i+1] + [(cut_point_geom.x, cut_point_geom.y)]
                second_part_coords = [(cut_point_geom.x, cut_point_geom.y)] + coords[i+1:]

                line1 = LineString(first_part_coords) if len(first_part_coords) >= 2 else None
                line2 = LineString(second_part_coords) if len(second_part_coords) >= 2 else None
                return line1, line2
            current_dist_along_line += segment_len
        # Should be covered by distance checks at start, but as a fallback:
        return line, None

    def shrink_polygon(self, p: shapely.Polygon | shapely.MultiPolygon, buffer_size: float) -> list[shapely.Polygon | shapely.MultiPolygon]:
        if p.is_empty or buffer_size <=0:
            return [p] if not p.is_empty else []

        shrunken_polygon = p.buffer(-buffer_size, join_style="mitre", single_sided=True)

        if shrunken_polygon.is_empty:
            return [p] # Cannot shrink, return original

        return [p, shrunken_polygon]

    def create_polygons_from_contours(self, contours: list[np.ndarray], hierarchy: np.ndarray, min_area_world: float) -> list[shapely.Polygon]:
        all_polygons = []
        if hierarchy is None or hierarchy.ndim != 2 or hierarchy.shape[0] != 1:
            log.warning("Unexpected hierarchy format or no hierarchy found when creating polygons from contours.")
            for cnt in contours:
                if len(cnt) < 3: continue
                world_coords = np.array([self.image_to_world(pt[0][0], pt[0][1]) for pt in cnt])
                poly = shapely.Polygon(world_coords)
                if poly.is_valid and poly.area >= min_area_world:
                    all_polygons.append(poly)
            return all_polygons

        parent_to_children_coords = defaultdict(list)
        for i, h_info in enumerate(hierarchy[0]):
            parent_idx = h_info[3]
            if parent_idx != -1:
                child_contour_img = contours[i]
                if len(child_contour_img) < 3: continue
                child_coords_world = np.array([self.image_to_world(pt[0][0], pt[0][1]) for pt in child_contour_img])
                temp_hole_poly = shapely.Polygon(child_coords_world) # Check validity of hole
                if temp_hole_poly.is_valid and temp_hole_poly.area > 1e-6: # Min area for a hole
                    parent_to_children_coords[parent_idx].append(child_coords_world)

        for i, cnt_ext_img in enumerate(contours):
            if hierarchy[0][i][3] == -1: # Exterior contour
                if len(cnt_ext_img) < 3: continue
                exterior_coords_world = np.array([self.image_to_world(pt[0][0], pt[0][1]) for pt in cnt_ext_img])
                holes_coords_world_list = parent_to_children_coords.get(i, [])
                try:
                    poly = shapely.Polygon(shell=exterior_coords_world, holes=holes_coords_world_list)
                    if not poly.is_valid: poly = poly.buffer(0) # Try to fix invalid geometry
                    if poly.is_valid and poly.area >= min_area_world:
                        all_polygons.append(poly)
                except Exception as e:
                    log.error(f"Error creating polygon from contour {i} (world coords): {e}")
        return all_polygons

    def informative_coverage(
        self,
        heatmap: np.ndarray,
        sensor_radius_world: float = 8.0,
        max_path_length_world: float = 500.0,
        contour_smoothing_world: float = 1.0,
        contour_threshold: float = 0.00001,
    ) -> list[LineString]:
        start_time = time.time()
        if heatmap is None or np.sum(heatmap) < 1e-9: # Check for near-zero sum too
            log.warning("Heatmap is empty or effectively all zeros. Cannot generate informative coverage.")
            return []

        mask = np.zeros_like(heatmap, dtype=np.uint8)
        mask[heatmap > contour_threshold] = 255

        # findContours expects (image_height, image_width) or (rows, cols)
        # If heatmap is (y_bins, x_bins), it's correct.
        contours_img, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if not contours_img:
            log.warning("No contours found in the heatmap above the threshold.")
            return []
        log.info(f"Found {len(contours_img)} raw contours in the heatmap.")

        min_contour_area_world = np.pi * (sensor_radius_world * 0.25)**2 # Filter smaller than quarter sensor disk

        all_polygons_world = self.create_polygons_from_contours(contours_img, hierarchy, min_contour_area_world)
        log.info(f"Created {len(all_polygons_world)} valid polygons from contours.")
        if not all_polygons_world: return []

        coverage_rings_world = []
        for poly_world_initial in all_polygons_world:
            if poly_world_initial.is_empty: continue

            poly_world = poly_world_initial
            if not poly_world.is_valid: poly_world = poly_world.buffer(0) # Attempt to fix
            if poly_world.is_empty or not poly_world.is_valid: continue

            simplified_poly = poly_world.simplify(contour_smoothing_world, preserve_topology=True)
            if simplified_poly.is_empty: continue

            polys_to_process_shrink = []
            if isinstance(simplified_poly, shapely.Polygon):
                polys_to_process_shrink.append(simplified_poly)
            elif isinstance(simplified_poly, shapely.MultiPolygon):
                polys_to_process_shrink.extend(list(simplified_poly.geoms))

            for p_to_shrink in polys_to_process_shrink:
                if p_to_shrink.is_empty or p_to_shrink.area < min_contour_area_world / 4: continue # Stricter filter

                # Iterative shrinking logic:
                current_poly_to_shrink = p_to_shrink
                while current_poly_to_shrink and not current_poly_to_shrink.is_empty and current_poly_to_shrink.area > min_contour_area_world / 2:
                    if current_poly_to_shrink.exterior and current_poly_to_shrink.exterior.length > EPS:
                        coverage_rings_world.append(LineString(current_poly_to_shrink.exterior.coords)) # Ensure it's a LineString copy

                    next_shrunk = current_poly_to_shrink.buffer(-sensor_radius_world * 2.0, join_style="round")
                    if next_shrunk.is_empty or not next_shrunk.is_valid:
                        # Try smaller shrink if full fails, then break
                        next_shrunk_half = current_poly_to_shrink.buffer(-sensor_radius_world, join_style="round")
                        if next_shrunk_half.is_empty or not next_shrunk_half.is_valid: break
                        current_poly_to_shrink = next_shrunk_half
                    else:
                        current_poly_to_shrink = next_shrunk

        log.info(f"Generated {len(coverage_rings_world)} coverage rings after shrinking.")
        if not coverage_rings_world: return []

        paths_world = []
        for ring_exterior in coverage_rings_world:
            if ring_exterior.length < sensor_radius_world / 2 : continue # Ignore very short rings

            coords = list(ring_exterior.coords)
            if not coords or len(coords) < 2 : continue

            # Make start point consistent for cutting (e.g. min x, then min y)
            min_idx = min(range(len(coords)), key=lambda i: (coords[i][0], coords[i][1]))
            # Ensure it's an open line for cutting by removing duplicate end if it's closed
            if Point(coords[0]).equals_exact(Point(coords[-1]), 1e-6):
                final_coords_for_line = coords[min_idx:-1] + coords[:min_idx+1] # Create one full loop
            else: # Already open or not properly closed
                final_coords_for_line = coords[min_idx:] + coords[:min_idx]


            if len(final_coords_for_line) < 2 : continue
            open_line_to_cut = LineString(final_coords_for_line)

            if open_line_to_cut.length > max_path_length_world:
                paths_world.extend(self.modulus_cut([open_line_to_cut], max_path_length_world))
            elif open_line_to_cut.length > EPS : # Add if it has some length
                paths_world.append(open_line_to_cut)

        final_paths = []
        for line_seg in paths_world:
            if line_seg.length > sensor_radius_world: # Ensure line is longer than sensor radius before cutting start
                _, path_main_segment = self.cut(line_seg, sensor_radius_world)
                if path_main_segment and not path_main_segment.is_empty and path_main_segment.length > EPS:
                    final_paths.append(path_main_segment)
            elif line_seg.length > EPS: # Keep shorter valid segments
                 final_paths.append(line_seg)

        end_time = time.time()
        log.info(f"Informative coverage: {end_time - start_time:.2f}s, {len(final_paths)} paths generated.")
        return final_paths

    def plot(
        self,
        show_basemap=True,
        show_features=False,
        show_heatmap=True,
        show_coverage=False,
        combined_heatmap_override=None,
        coverage_paths_override=None
    ):
        fig, ax = plt.subplots(figsize=(12,12))
        if self.polygon is None:
            log.error("Cannot plot: Environment polygon not initialized.")
            return
        ax.set_xlim(self.minx - self.buffer_val, self.maxx + self.buffer_val)
        ax.set_ylim(self.miny - self.buffer_val, self.maxy + self.buffer_val)
        ax.set_aspect('equal', adjustable='box')

        current_alpha = 1.0

        if show_basemap:
            if self.polygon.crs:
                plot_utils.plot_basemap(ax=ax, source=cx.providers.OpenStreetMap.Mapnik, crs=self.polygon.crs)
                current_alpha = 0.6 # Make subsequent layers somewhat transparent
            else:
                log.warning("Cannot show basemap, environment polygon CRS not set.")


        if show_features:
            self.polygon.plot(ax=ax, linestyle="--", facecolor="none", edgecolor="black", alpha=0.8, linewidth=1.5, label="Boundary")
            num_feature_cats = len(self.features)
            feature_color_map = plt.cm.get_cmap('tab20', num_feature_cats if num_feature_cats > 0 else 1)

            plotted_feature_labels = set()
            for i, (key, feature_gdf) in enumerate(self.features.items()):
                if feature_gdf is not None and not feature_gdf.empty:
                    label_key = f"{key.capitalize()} Features"
                    if label_key not in plotted_feature_labels: # Add label only once per category
                        feature_gdf.plot(ax=ax, color=feature_color_map(i), linewidth=1, label=label_key, alpha=current_alpha*0.7)
                        plotted_feature_labels.add(label_key)
                    else:
                         feature_gdf.plot(ax=ax, color=feature_color_map(i), linewidth=1, alpha=current_alpha*0.7)


        heatmap_to_display = combined_heatmap_override
        if show_heatmap and heatmap_to_display is None:
            sigma_features = {key: 3.0 for key in self.tags.keys()}
            alpha_features = {key: 1.0 for key in self.tags.keys()}
            heatmap_to_display = self.get_combined_heatmap(sigma_features, alpha_features)

        if show_heatmap and heatmap_to_display is not None and np.sum(heatmap_to_display) > 1e-9:
            colors = [(1,1,1,0), (0.1,0.1,1,0.1), (0,0.5,1,0.3), (0,1,1,0.5), (0.7,1,0,0.7), (1,1,0,0.85), (1,0,0,1)]
            custom_cmap = LinearSegmentedColormap.from_list("custom_fade_more_transparent", colors)

            extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

            im = ax.imshow(
                heatmap_to_display, extent=extent, origin="lower", cmap=custom_cmap, alpha=current_alpha, interpolation='bilinear',
                vmin=0, vmax=np.percentile(heatmap_to_display[heatmap_to_display > 0], 99) if np.any(heatmap_to_display > 0) else 1 # Robust vmax
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
            cbar.set_label('Probability Density')


        paths_to_display = coverage_paths_override
        if show_coverage and paths_to_display is None and heatmap_to_display is not None:
            sensor_radius = 8.0
            paths_to_display = self.informative_coverage(heatmap_to_display, sensor_radius_world=sensor_radius)

        if show_coverage and paths_to_display:
            # Check if paths_to_display is a list of LineString, if not, wrap it
            if not isinstance(paths_to_display, list) or not all(isinstance(p, LineString) for p in paths_to_display):
                 log.warning("coverage_paths_override is not a list of LineStrings. Cannot plot.")
            else:
                coverage_multiline = GeoMultiTrajectory(paths_to_display, crs=self.polygon.crs)
                coverage_multiline.plot(ax=ax, color='fuchsia', linewidth=1.2, alpha=0.9, label="Coverage Paths")

        ax.set_title("SAR Environment Overview")
        ax.set_xlabel(f"X Coordinate ({self.polygon.crs})")
        ax.set_ylabel(f"Y Coordinate ({self.polygon.crs})")

        handles, labels = ax.get_legend_handles_labels()
        if handles: # Avoid empty legend warning
            ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.02, 1), title="Legend")

        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.show()

    def visualise_environment(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        if self.polygon is None:
            log.error("Cannot visualize environment: polygon not initialized.")
            return
        ax.set_aspect('equal', adjustable='box')

        self.polygon.plot(ax=ax, facecolor="lightgrey", edgecolor="black", alpha=0.5, label="Search Boundary")

        num_feature_cats = len(self.features)
        colors_cm = plt.cm.get_cmap('tab20', num_feature_cats if num_feature_cats > 0 else 1)
        legend_handles = [Patch(facecolor="lightgrey", edgecolor="black", label="Search Boundary", alpha=0.5)]

        for i, (feature_type, gdf) in enumerate(self.features.items()):
            if gdf is not None and not gdf.empty:
                color_val = colors_cm(i / num_feature_cats if num_feature_cats > 0 else 0.5)
                gdf.plot(ax=ax, label=feature_type.capitalize(), color=color_val, alpha=0.6, linewidth=0.8)
                legend_handles.append(
                    Patch(facecolor=color_val, edgecolor=color_val, label=feature_type.capitalize(), alpha=0.6)
                )

        ax.set_title("Environment Features Visualization")
        ax.set_xlabel(f"X Coordinate ({self.polygon.crs})")
        ax.set_ylabel(f"Y Coordinate ({self.polygon.crs})")
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.pause(0.1)  # Allow GUI to update

    def plot_heatmap_simple(
        self,
        heatmap: np.ndarray,
        show_basemap=True,
        show_features=False,
        export_final_image=False,
    ):
        if self.polygon is None or self.xedges is None or self.yedges is None:
            log.error("Environment not fully initialized for plotting.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        ax.set_xlim(self.minx - self.buffer_val, self.maxx + self.buffer_val)
        ax.set_ylim(self.miny - self.buffer_val, self.maxy + self.buffer_val)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Heatmap Visualization")

        alpha_val_basemap = 1.0
        if show_basemap and self.polygon.crs:
            plot_utils.plot_basemap(ax=ax, source=cx.providers.OpenStreetMap.Mapnik, crs=self.polygon.crs)
            alpha_val_basemap = 0.6

        if show_features:
            self.polygon.plot(ax=ax, linestyle="--", facecolor="none", edgecolor="darkgray", alpha=alpha_val_basemap, linewidth=0.8)
            # TODO plot all features

        colors_heatmap = [(1, 1, 1, 0), (0, 0, 1, 0.2), (0, 0.5, 1, 0.4), (0, 1, 1, 0.6), (0.5, 1, 0, 0.8), (1, 1, 0, 0.9), (1, 0, 0, 1)]
        custom_cmap_heatmap = LinearSegmentedColormap.from_list("custom_fade_gradual", colors_heatmap, N=256)
        extent_heatmap = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

        heatmap_display_artist = ax.imshow(
            heatmap,
            extent=extent_heatmap,
            origin="lower",
            cmap=custom_cmap_heatmap,
            alpha=alpha_val_basemap,
            # interpolation='nearest',  # Ensure no interpolation
        )
        cbar_heatmap = fig.colorbar(heatmap_display_artist, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
        cbar_heatmap.set_label('Probability Density')
        if np.any(heatmap > 0):
            cbar_heatmap.mappable.set_clim(vmin=0, vmax=np.percentile(heatmap[heatmap > 0], 99.5))

        if export_final_image:
            plt.savefig("heatmap_visualization.png", dpi=300, bbox_inches='tight')
            log.info("Plot saved to heatmap_visualization.png")

        plt.show()

    def plot_heatmap_interactive(
        self,
        initial_heatmap: np.ndarray,
        show_basemap=True,
        show_features=False, # Typically false for interactive clarity on heatmap
        export_final_image=False,
        show_coverage_paths=True,
    ):
        if self.polygon is None or self.xedges is None or self.yedges is None:
            log.error("Environment not fully initialized for interactive plotting.")
            return

        fig, ax = plt.subplots(figsize=(12,10))
        fig.subplots_adjust(left=0.1, bottom=0.40) # Increased bottom margin for more sliders

        ax.set_xlim(self.minx - self.buffer_val, self.maxx + self.buffer_val)
        ax.set_ylim(self.miny - self.buffer_val, self.maxy + self.buffer_val)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Interactive Heatmap and Coverage")

        alpha_val_basemap = 1.0 # Default alpha for layers over basemap
        if show_basemap and self.polygon.crs:
            plot_utils.plot_basemap(ax=ax, source=cx.providers.OpenStreetMap.Mapnik, crs=self.polygon.crs)
            alpha_val_basemap = 0.6

        if show_features:
            # TODO show the features in the image coordinates
            pass
        colors_heatmap = [(1,1,1,0), (0.1,0.1,1,0.1), (0,0.5,1,0.3), (0,1,1,0.5), (0.7,1,0,0.7), (1,1,0,0.85), (1,0,0,1)]
        custom_cmap_heatmap = LinearSegmentedColormap.from_list("custom_fade_interactive", colors_heatmap)
        extent_heatmap = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

        heatmap_display_artist = ax.imshow(
            initial_heatmap,
            extent=extent_heatmap, origin="lower", cmap=custom_cmap_heatmap, alpha=alpha_val_basemap, interpolation='bilinear'
        )
        cbar_heatmap = fig.colorbar(heatmap_display_artist, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
        cbar_heatmap.set_label('Probability Density')
        if np.any(initial_heatmap > 0): # Avoid error if all zero
             cbar_heatmap.mappable.set_clim(vmin=0, vmax=np.percentile(initial_heatmap[initial_heatmap > 0], 99.5))

        coverage_plot_artists_list = []

        # Ensure self.heatmaps is populated for slider creation based on available features
        if not self.heatmaps or not any(h is not None for h in self.heatmaps.values()):
            log.info("Individual heatmaps for features not found, generating them now for slider setup.")
            self.generate_heatmaps() # Crucial for sliders to know which features exist

        sorted_feature_keys_for_sliders = sorted([k for k,v in self.heatmaps.items() if v is not None]) # Only for generated heatmaps
        if not sorted_feature_keys_for_sliders:
            log.warning("No features with generated heatmaps available for interactive sliders.")
            # Show plot without sliders if no features
            plt.show()
            return

        slider_start_y_pos = 0.30
        slider_element_height = 0.02
        slider_vertical_spacing = 0.025 # Spacing between slider elements (sigma + alpha for one feature)

        sigma_sliders_dict = {}
        alpha_sliders_dict = {}

        # Dynamically create sliders based on available heatmaps
        for i, key in enumerate(sorted_feature_keys_for_sliders):
            current_y_pos = slider_start_y_pos - i * slider_vertical_spacing * 2 # *2 for sigma and alpha pair

            ax_sigma_slider = fig.add_axes([0.15, current_y_pos, 0.3, slider_element_height])
            sigma_sliders_dict[key] = Slider(ax_sigma_slider, f"{key[:10]} σ", 0.0, 10.0, valinit=3.0, valstep=0.1)

            ax_alpha_slider = fig.add_axes([0.55, current_y_pos, 0.3, slider_element_height])
            alpha_sliders_dict[key] = Slider(ax_alpha_slider, f" α", 0.0, 5.0, valinit=1.0, valstep=0.1) # Shorter label for alpha

        # Sliders for coverage parameters, placed below feature sliders
        coverage_params_y_start = slider_start_y_pos - (len(sorted_feature_keys_for_sliders)) * slider_vertical_spacing * 2

        ax_sensor_radius_slider = fig.add_axes([0.15, coverage_params_y_start - slider_vertical_spacing, 0.3, slider_element_height])
        sensor_radius_slider_widget = Slider(ax_sensor_radius_slider, "Sensor Rad.", 1.0, 50.0, valinit=8.0, valstep=0.5)

        ax_max_path_len_slider = fig.add_axes([0.55, coverage_params_y_start - slider_vertical_spacing, 0.3, slider_element_height])
        max_path_len_slider_widget = Slider(ax_max_path_len_slider, "Max Path Len.", 50.0, 2000.0, valinit=500.0, valstep=10)

        all_slider_widgets = list(sigma_sliders_dict.values()) + list(alpha_sliders_dict.values()) + [sensor_radius_slider_widget, max_path_len_slider_widget]

        def update_interactive_plot(val=None):
            nonlocal coverage_plot_artists_list

            current_sigmas_from_sliders = {key: s.val for key, s in sigma_sliders_dict.items()}
            current_alphas_from_sliders = {key: a.val for key, a in alpha_sliders_dict.items()}
            current_sensor_radius_val = sensor_radius_slider_widget.val
            current_max_path_len_val = max_path_len_slider_widget.val

            updated_heatmap = self.get_combined_heatmap(current_sigmas_from_sliders, current_alphas_from_sliders)
            if updated_heatmap is None: return

            heatmap_display_artist.set_data(updated_heatmap)
            if np.any(updated_heatmap > 0):
                 heatmap_display_artist.set_clim(vmin=0, vmax=np.percentile(updated_heatmap[updated_heatmap > 0], 99.5))
            else:
                 heatmap_display_artist.set_clim(vmin=0, vmax=1)


            for artist in coverage_plot_artists_list: artist.remove()
            coverage_plot_artists_list.clear()

            if show_coverage_paths:
                new_coverage_paths = self.informative_coverage(
                    updated_heatmap,
                    sensor_radius_world=current_sensor_radius_val,
                    max_path_length_world=current_max_path_len_val
                )
                if new_coverage_paths:
                    for line_geom in new_coverage_paths:
                        gpd_series_line = gpd.GeoSeries([line_geom], crs=self.polygon.crs if self.polygon else None)
                        # plot returns list of artists, extend our list
                        coverage_plot_artists_list.extend(gpd_series_line.plot(ax=ax, color='darkviolet', linewidth=1.0, alpha=0.9))
            fig.canvas.draw_idle()

        for slider_widget_instance in all_slider_widgets:
            slider_widget_instance.on_changed(update_interactive_plot)
        update_interactive_plot() # Initial call to draw


        if export_final_image:
            ax_save_plot_button = fig.add_axes([0.85, 0.01, 0.1, 0.04]) # Positioned bottom right
            save_plot_button_widget = plt.Button(ax_save_plot_button, 'Save Plot')
            def on_save_plot_clicked(event):
                slider_axes_to_hide = [s.ax for s in all_slider_widgets] + [ax_save_plot_button]
                for slider_ax_item in slider_axes_to_hide: slider_ax_item.set_visible(False)
                fig.canvas.draw_idle()
                plt.savefig("interactive_plot_output.png", dpi=300, bbox_inches='tight')
                log.info("Plot saved to interactive_plot_output.png")
                for slider_ax_item in slider_axes_to_hide: slider_ax_item.set_visible(True)
                fig.canvas.draw_idle()
            save_plot_button_widget.on_clicked(on_save_plot_clicked)

        plt.show()

class DataGenerator:
    """
    Generates and exports SAR environment data.
    Uses an efficient 'query once, clip many' strategy.
    """

    def __init__(self):
        self.tags_mapping = {
            "structure": {"building": True, "man_made": True, "bridge": True, "tunnel": True,},
            "road": {"highway": True, "tracktype": True},
            "linear": {"railway": True, "barrier": True, "fence": True, "wall": True, "pipeline": True,},
            "drainage": {"waterway": ["drain", "ditch", "culvert", "canal"]},
            "water": {"natural": ["water", "wetland"], "water": True, "wetland": True, "reservoir": True,},
            "brush": {"landuse": ["grass"]},
            "scrub": {"natural": "scrub"},
            "woodland": {"landuse": ["forest", "wood"], "natural": "wood"},
            "field": {"landuse": ["farmland", "farm", "meadow"]},
            "rock": {"natural": ["rock", "bare_rock", "scree", "cliff"]},
        }

        self._builder = EnvironmentBuilder()
        for feature_category, osm_tags in self.tags_mapping.items():
            self._builder.set_feature(feature_category, osm_tags)

        self.quantile_radii_km = {
            'q1': 0.6,
            'median': 1.8,
            'q3': 3.2,
            'q95': 9.9,
        }

    def _get_utm_epsg(self, lon: float, lat: float) -> str:
        """Calculates the appropriate UTM zone EPSG code for a given point."""
        zone = int((lon + 180) / 6) + 1
        epsg_code = f"326{zone}" if lat >= 0 else f"327{zone}"
        log.info(f"Determined UTM zone for point ({lon}, {lat}) as EPSG:{epsg_code}")
        return f"EPSG:{epsg_code}"

    def _create_circular_polygon(self, center_lon: float, center_lat: float, radius_km: float) -> dict:
        """Creates a GeoJSON dictionary for a circular polygon centered at a point."""
        point = shapely.Point(center_lon, center_lat)
        gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")

        projected_crs = self._get_utm_epsg(center_lon, center_lat)
        gdf_proj = gdf.to_crs(projected_crs)

        radius_meters = radius_km * 1000
        circle_proj = gdf_proj.buffer(radius_meters).iloc[0]

        circle_gdf_proj = gpd.GeoDataFrame([1], geometry=[circle_proj], crs=projected_crs)
        circle_gdf_wgs84 = circle_gdf_proj.to_crs("EPSG:4326")

        return circle_gdf_wgs84.geometry.iloc[0]

    def generate(self, center_point: tuple[float, float], quantile: str, meter_per_bin: int = 5) -> 'Environment | None':
        """Generates a single Environment object for a given location and quantile."""
        center_lon, center_lat = center_point
        radius_km = self.quantile_radii_km[quantile]
        projected_crs = self._get_utm_epsg(center_lon, center_lat)

        log.info(f"Generating data for center ({center_lon:.4f}, {center_lat:.4f}) with quantile '{quantile}' (radius: {radius_km} km).")
        geojson_poly = self._create_circular_polygon(center_lon, center_lat, radius_km)

        try:
            env = self._builder.set_polygon(geojson_poly).set_projected_crs(projected_crs).set_meter_per_bin(meter_per_bin).build()
            return env
        except Exception as e:
            log.error(f"An error occurred during environment generation: {e}", exc_info=True)
            return None


    def export_all_quantiles(self, center_point: tuple[float, float], output_directory: str, meter_per_bin: int = 5):
        """
        Efficiently generates and exports data for all predefined quantiles. The center point
        is embedded within each GeoJSON file.
        """
        log.info(f"--- Starting efficient batch export to '{output_directory}' ---")
        os.makedirs(output_directory, exist_ok=True)
        center_lon, center_lat = center_point

        # 1. Generate the single, largest environment (q95)
        log.info("--- Generating master environment for the largest radius (q95) ---")
        master_env = self.generate(center_point, 'q95', meter_per_bin)
        if not master_env:
            log.error("Failed to generate the master environment. Aborting export.")
            return

        # 2. Combine all features from the master environment into one GeoDataFrame
        master_features_list = []
        for key, gdf in master_env.features.items():
            if gdf is not None and not gdf.empty:
                temp_gdf = gdf.copy()
                temp_gdf['feature_type'] = key
                master_features_list.append(temp_gdf)

        if not master_features_list:
            log.warning("No features found in the master environment. Skipping feature export.")
            master_features_gdf = None
        else:
            master_features_gdf = pd.concat(master_features_list, ignore_index=True)

        # 3. Generate the single, master heatmap
        log.info("--- Generating master heatmap for the largest radius ---")
        master_heatmap = master_env.get_combined_heatmap()

        # Plotting
        # master_env.plot(show_basemap=True, show_features=False, show_heatmap=True, show_coverage=False, combined_heatmap_override=master_heatmap)

        # 4. Loop through all quantiles to clip and export subsets
        for quantile_name, radius_km in self.quantile_radii_km.items():
            log.info(f"--- Processing and exporting subset for quantile: {quantile_name} ---")

            clipping_point_wgs84 = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs="EPSG:4326")
            clipping_point_proj = clipping_point_wgs84.to_crs(master_env.projected_crs)
            clipping_circle_proj = clipping_point_proj.buffer(radius_km * 1000).iloc[0]

            # 4a. Clip features and export
            if master_features_gdf is not None:
                clipped_features_proj = gpd.clip(master_features_gdf, clipping_circle_proj)
                clipped_features_wgs84 = clipped_features_proj.to_crs("EPSG:4326")

                geojson_path = os.path.join(output_directory, f"features_{quantile_name}.geojson")

                # Manually create the GeoJSON dictionary to add custom metadata
                geojson_dict = clipped_features_wgs84.__geo_interface__
                geojson_dict['center_point'] = center_point # Embed the center point

                with open(geojson_path, 'w') as f:
                    json.dump(geojson_dict, f)

                log.info(f"Exported {len(clipped_features_wgs84)} features to {geojson_path}")

            # 4b. Crop heatmap and export (no changes here)
            if master_heatmap is not None:
                bounds = clipping_circle_proj.bounds
                min_x, min_y, max_x, max_y = bounds

                img_min_x, img_min_y = master_env.world_to_image(np.array([min_x]), np.array([min_y]))
                img_max_x, img_max_y = master_env.world_to_image(np.array([max_x]), np.array([max_y]))

                cropped_heatmap = master_heatmap[img_min_y[0]:img_max_y[0], img_min_x[0]:img_max_x[0]]

                heatmap_path = os.path.join(output_directory, f"heatmap_{quantile_name}.npy")
                np.save(heatmap_path, cropped_heatmap)
                log.info(f"Exported heatmap matrix (shape: {cropped_heatmap.shape}) to {heatmap_path}")

        log.info("--- Batch export completed. ---")
