# sarenv/planning/path_generators.py
# sarenv/core/environment.py
import concurrent.futures
import json
import os
import time
from collections import defaultdict

import cv2  # Ensure opencv-python is in requirements.txt
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from skimage.draw import (
    polygon as ski_polygon,
)

from ..io.osm_query import query_features
from ..utils import (
    logging_setup,
    plot_utils,
)

log = logging_setup.get_logger()
EPS = 1e-9
# Placeholder function, to be implemented
def generate_search_tasks(geometries):
    """
    Generates search tasks based on input geometries.
    This is a placeholder and needs to be implemented.
    """
    # Example:
    # sweep_paths = generate_lawnmower_paths(geometries['roi'], ...)
    # road_following_paths = generate_road_paths(geometries['roads'], ...)
    # return combined_paths
    print("generate_search_tasks is a placeholder.")
    pass
    
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
