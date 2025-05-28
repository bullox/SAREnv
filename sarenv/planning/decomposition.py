# sarenv/planning/decomposition.py
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.affinity import rotate
from shapely.ops import unary_union

# --- Constants ---
EPS = 1e-9  # Epsilon for floating point comparisons

# --- Type Aliases ---
Coord = Tuple[float, float]
Chain = List[Coord]
OpenCell = Dict[
    str, Any
]  # Keys: 'lower_chain', 'upper_chain', 'lower_segment', 'upper_segment'


# --- Helper Functions ---
def feq(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) < eps


def get_polygon_segments(polygon: Polygon) -> List[LineString]:
    """Extracts all unique segments from a polygon's exterior and interiors."""
    segments_wkt = set() # Use WKT for uniqueness to handle floating point objects

    def add_segments_from_coords(coords_list: List[Coord], segment_set_wkt: set):
        for i in range(len(coords_list) - 1):
            p1 = coords_list[i]
            p2 = coords_list[i + 1]
            ls = LineString([p1, p2])
            if ls.length > EPS:
                segment_set_wkt.add(ls.wkt)

    add_segments_from_coords(list(polygon.exterior.coords), segments_wkt)
    for interior in polygon.interiors:
        add_segments_from_coords(list(interior.coords), segments_wkt)

    return [LineString(wkt_seg) for wkt_seg in segments_wkt]


def get_segment_y_at_x(
    segment: LineString,
    x_coord: float,
    default_y_if_no_intersect: Optional[float] = None,
) -> Optional[float]:
    """Calculates the y-coordinate of a segment at a given x_coord."""
    min_x, min_y_seg, max_x, max_y_seg = segment.bounds # Renamed min_y, max_y to avoid conflict

    # Check if x_coord is within the segment's x-range (with tolerance)
    if not (min_x - EPS <= x_coord <= max_x + EPS):
        # Special handling for vertical segments at their x-coordinate
        if feq(segment.coords[0][0], x_coord) and feq(segment.coords[1][0], x_coord): # Vertical segment
             if min_x - EPS <= x_coord <= max_x + EPS : # If x_coord is the x of vertical segment
                return (segment.coords[0][1] + segment.coords[1][1]) / 2.0 # Midpoint or one of the y's
        return default_y_if_no_intersect


    # If segment is vertical and x_coord matches its x
    if feq(segment.coords[0][0], x_coord) and feq(segment.coords[1][0], x_coord):
        # For a vertical segment, any y between its min_y and max_y is valid.
        # This function usually expects a single y. For BCD, we typically take
        # one of the endpoints if x_coord matches an endpoint's x.
        # Or, if it's a truly vertical sweep line intersecting a vertical segment,
        # this case might need specific handling based on BCD algorithm phase.
        # For sorting active edges, an endpoint y is often used.
        # Let's return the lower y for consistency if it's vertical at x_coord.
        return min(segment.coords[0][1], segment.coords[1][1])


    # Create a vertical line for intersection that spans well beyond the segment's y-bounds
    # Use a sufficiently large range for the vertical line
    vertical_line_min_y = min_y_seg - abs(max_y_seg - min_y_seg) - 100
    vertical_line_max_y = max_y_seg + abs(max_y_seg - min_y_seg) + 100
    vertical_line = LineString([(x_coord, vertical_line_min_y), (x_coord, vertical_line_max_y)])
    intersection = segment.intersection(vertical_line)

    if isinstance(intersection, Point):
        return intersection.y
    elif isinstance(intersection, LineString) and not intersection.is_empty:
        # Horizontal segment overlapping the vertical line at x_coord
        # All points on this part of the segment have the same y.
        return intersection.coords[0][1]
    # Handle other intersection types like MultiPoint if necessary, though unlikely for BCD context
    return default_y_if_no_intersect


def sort_active_edges_L(L: List[LineString], current_x: float):
    """Sorts active edges in L by their y-intersection at current_x."""
    if not L:
        return

    def sort_key(segment: LineString):
        y_val = get_segment_y_at_x(segment, current_x)
        # If y_val is None, it means segment doesn't cross current_x, should not happen for "active" edges
        # Or it's perfectly vertical. For vertical, use min y.
        if y_val is None:
            if feq(segment.bounds[0], current_x) and feq(segment.bounds[2], current_x): # Vertical segment at current_x
                return segment.bounds[1] # min_y of vertical segment
            return float('inf') # Should not happen if logic is correct
        return y_val

    L.sort(key=sort_key)


def add_point_to_chain(chain: Chain, point_coord: Coord):
    """Adds a point to a chain, avoiding consecutive duplicates."""
    if not chain or not (
        feq(chain[-1][0], point_coord[0]) and feq(chain[-1][1], point_coord[1])
    ):
        chain.append(point_coord)


def form_polygon_from_chains(
    lower_chain: Chain, upper_chain: Chain, final_cleanup_tolerance=EPS / 100
) -> Optional[Polygon]:
    """Forms a Shapely Polygon from a lower and upper chain of coordinates."""
    if not lower_chain or not upper_chain or len(lower_chain) < 2 or len(upper_chain) < 2:
        # Each chain needs at least two points to form a segment of the polygon boundary
        return None

    # Polygon coords: lower chain, then reversed upper chain
    poly_coords_raw = lower_chain + upper_chain[::-1]

    if len(poly_coords_raw) < 3:
        return None

    # Remove consecutive duplicates from the combined chain
    final_ring_coords = []
    if poly_coords_raw:
        final_ring_coords.append(poly_coords_raw[0])
        for i in range(1, len(poly_coords_raw)):
            # Check distance instead of feq for robustness with very close points
            if Point(poly_coords_raw[i]).distance(Point(final_ring_coords[-1])) > EPS/10:
                final_ring_coords.append(poly_coords_raw[i])

    # Ensure closure for Shapely Polygon constructor (first and last point same)
    if len(final_ring_coords) >= 3:
        if Point(final_ring_coords[0]).distance(Point(final_ring_coords[-1])) > EPS/10 :
            final_ring_coords.append(final_ring_coords[0])
    else: # Not enough unique points
        return None


    if len(final_ring_coords) < 4:  # e.g., A,B,C,A needs 4 points in list
        return None

    try:
        cell_poly = Polygon(final_ring_coords)
        # A very small buffer can sometimes fix minor self-intersections or invalid topology
        # Apply only if needed, as it changes geometry slightly.
        if not cell_poly.is_valid:
             cell_poly = cell_poly.buffer(final_cleanup_tolerance).buffer(-final_cleanup_tolerance)
             if not cell_poly.is_valid: # Still invalid
                  # Try zero buffer trick
                  cell_poly = cell_poly.buffer(0)


        # Check area again after potential buffer(0) which might create empty or tiny polygons
        if cell_poly.is_valid and cell_poly.area > EPS * 100:  # Stricter area check
            return cell_poly
    except Exception: # Error during polygon creation (e.g., from invalid topology)
        # print(f"Debug: Failed to form polygon. Coords: {final_ring_coords}")
        return None
    return None


def boustrophedon_decomposition(boundary_polygon: Polygon, sweep_angle_deg: float = 0.0) -> List[Polygon]:
    if not boundary_polygon.is_valid or boundary_polygon.is_empty:
        return []

    effective_rotation_angle = -sweep_angle_deg
    # Rotation origin: Using (0,0) is simple. If polygons are far from origin,
    # consider rotating around centroid: rotation_origin = boundary_polygon.centroid
    rotation_origin = Point(0,0)
    poly_to_sweep = rotate(boundary_polygon, effective_rotation_angle, origin=rotation_origin, use_radians=False)

    if not poly_to_sweep.is_valid:
        poly_to_sweep = poly_to_sweep.buffer(0) # Attempt to fix validity
        if not poly_to_sweep.is_valid:
            # print("Warning: Polygon became invalid after rotation and buffer(0).")
            return [] # Or raise an error

    # Ensure consistent orientation (exterior CCW, interior CW for Shapely)
    # This is important for some geometric operations.
    # Shapely's Polygon constructor handles this if coordinates are ordered correctly.
    # If `poly_to_sweep` might have incorrect orientation from `rotate`, re-orient:
    if hasattr(poly_to_sweep, 'exterior') and poly_to_sweep.exterior.is_ccw is False:
        poly_to_sweep = Polygon(list(poly_to_sweep.exterior.coords)[::-1],
                                [list(hole.coords)[::-1] if hole.is_ccw is True else list(hole.coords) for hole in poly_to_sweep.interiors])


    all_coords_tuples = []
    all_coords_tuples.extend(list(poly_to_sweep.exterior.coords))
    for interior in poly_to_sweep.interiors:
        all_coords_tuples.extend(list(interior.coords))

    # Get unique x-coordinates of vertices as event points, sorted.
    event_x_coords = sorted(list(set([coord[0] for coord in all_coords_tuples])))

    if not event_x_coords: return []

    # Get all unique segments of the (rotated) polygon
    # This ensures we process each geometric edge only once.
    all_polygon_segments = get_polygon_segments(poly_to_sweep)


    L_active_edges: List[LineString] = []
    open_cells: List[OpenCell] = []
    closed_polygons: List[Polygon] = []

    previous_x = event_x_coords[0] # Initialize with the first event x-coordinate

    # For contains checks, a slightly buffered polygon can be more robust near boundaries.
    # Make this buffer very small.
    tolerant_poly_for_contains = poly_to_sweep.buffer(EPS * 10)


    for idx_event_x, current_x in enumerate(event_x_coords):
        # --- 3.A. Extend existing open cells' chains to current_x ---
        if idx_event_x > 0 and not feq(current_x, previous_x): # Only if x has advanced
            for cell_to_extend in open_cells:
                lower_s = cell_to_extend['lower_segment']
                upper_s = cell_to_extend['upper_segment']
                y_l_curr = get_segment_y_at_x(lower_s, current_x)
                y_u_curr = get_segment_y_at_x(upper_s, current_x)
                if y_l_curr is not None: add_point_to_chain(cell_to_extend['lower_chain'], (current_x, y_l_curr))
                if y_u_curr is not None: add_point_to_chain(cell_to_extend['upper_chain'], (current_x, y_u_curr))

        # --- 3.B. Update L_active_edges ---
        # Remove edges from L that end before or at current_x
        L_active_edges = [seg for seg in L_active_edges if seg.bounds[2] > current_x + EPS] # max_x > current_x

        # Add edges from all_polygon_segments that start at current_x and go to the right
        for seg_candidate in all_polygon_segments:
            # Check if segment starts near current_x and extends to the right
            if feq(seg_candidate.bounds[0], current_x) and seg_candidate.bounds[2] > current_x + EPS:
                 # Avoid adding duplicates if an edge was already carried over
                is_already_in_L = any(existing_seg.equals_exact(seg_candidate, EPS) for existing_seg in L_active_edges)
                if not is_already_in_L:
                    L_active_edges.append(seg_candidate)

        # --- 3.C. Sort L_active_edges by y-intersection at current_x ---
        sort_active_edges_L(L_active_edges, current_x)

        # --- 3.D. Reconcile open_cells with the new L_active_edges ---
        next_iteration_open_cells: List[OpenCell] = []
        # Keep track of which old cells are "matched" to new L_active_edges pairs
        old_cell_matched_flags = [False] * len(open_cells)


        for i in range(len(L_active_edges) - 1): # Iterate through pairs of adjacent active edges
            seg_L_lower_current = L_active_edges[i]
            seg_L_upper_current = L_active_edges[i+1]

            # Try to find if this (seg_L_lower_current, seg_L_upper_current) pair corresponds to an existing open cell
            matched_old_cell_idx = -1
            for k, old_cell_candidate in enumerate(open_cells):
                if not old_cell_matched_flags[k]: # If not already matched
                    # Check if segments are geometrically identical (within tolerance)
                    # Using equals_exact might be too strict due to floating points from rotation.
                    # A custom comparison or WKT comparison might be needed if equals() is not robust enough.
                    if old_cell_candidate['lower_segment'].equals(seg_L_lower_current) and \
                       old_cell_candidate['upper_segment'].equals(seg_L_upper_current):
                        matched_old_cell_idx = k
                        break
            
            if matched_old_cell_idx != -1:
                # This pair continues an existing cell
                next_iteration_open_cells.append(open_cells[matched_old_cell_idx])
                old_cell_matched_flags[matched_old_cell_idx] = True
            else:
                # This is a new cell opening up
                y_lower_at_curr_x = get_segment_y_at_x(seg_L_lower_current, current_x)
                y_upper_at_curr_x = get_segment_y_at_x(seg_L_upper_current, current_x)

                if y_lower_at_curr_x is not None and y_upper_at_curr_x is not None and \
                   y_upper_at_curr_x > y_lower_at_curr_x + EPS: # Ensure distinct upper/lower and valid gap
                    
                    # Check if the midpoint of this new potential cell is inside the polygon
                    mid_y_of_new_slice = (y_lower_at_curr_x + y_upper_at_curr_x) / 2.0
                    test_point_for_new_cell = Point(current_x, mid_y_of_new_slice)
                    
                    if tolerant_poly_for_contains.contains(test_point_for_new_cell):
                        newly_opened_cell: OpenCell = {
                            'lower_chain': [(current_x, y_lower_at_curr_x)],
                            'upper_chain': [(current_x, y_upper_at_curr_x)],
                            'lower_segment': seg_L_lower_current,
                            'upper_segment': seg_L_upper_current
                        }
                        next_iteration_open_cells.append(newly_opened_cell)

        # Close any old cells that were not matched and continued
        for k, old_cell_to_potentially_close in enumerate(open_cells):
            if not old_cell_matched_flags[k]:
                # Chains for this cell were extended to current_x in step 3.A (if x advanced)
                # or they end at previous_x if current_x == previous_x (first event)
                formed_poly = form_polygon_from_chains(
                    old_cell_to_potentially_close['lower_chain'],
                    old_cell_to_potentially_close['upper_chain']
                )
                if formed_poly:
                    closed_polygons.append(formed_poly)
        
        open_cells = next_iteration_open_cells
        previous_x = current_x # Update previous_x for the next iteration

    # --- 4. Finalization: If sweep finished before polygon's max_x, extend and close cells ---
    # This step might be covered if event_x_coords includes polygon.bounds[2]
    # However, explicit closing for any remaining open_cells is good.
    # The last event x might be the max_x of the polygon.
    # If open_cells remain, their chains should end at the x where their segments end, or polygon max_x.
    # This part is tricky. `form_polygon_from_chains` should handle chains of varying lengths.

    for cell_to_finalize in open_cells: # Any cells still open must be finalized
        # Their chains should ideally already extend to the point where one of their bounding segments ends.
        # If not, they might need one last extension to polygon.bounds[2] if applicable.
        # However, the event processing should handle this.
        formed_poly_final = form_polygon_from_chains(
            cell_to_finalize['lower_chain'],
            cell_to_finalize['upper_chain']
        )
        if formed_poly_final:
            closed_polygons.append(formed_poly_final)
    
    # --- 5. Rotate cells back and return ---
    final_decomposed_cells: List[Polygon] = []
    if closed_polygons:
        unrotation_angle = sweep_angle_deg # Rotate back by the original angle
        for cell_poly_rotated in closed_polygons:
            if not cell_poly_rotated.is_valid: # Try to fix if any became invalid
                cell_poly_rotated = cell_poly_rotated.buffer(0)
            
            if cell_poly_rotated.is_valid and cell_poly_rotated.area > EPS * 10: # Filter tiny/invalid
                rotated_back_cell = rotate(cell_poly_rotated, unrotation_angle, origin=rotation_origin, use_radians=False)
                if rotated_back_cell.is_valid and rotated_back_cell.area > EPS * 10:
                     # Ensure the cell is actually within the original boundary_polygon (post-rotation)
                     # Intersection can slightly change geometry, so check containment or significant overlap.
                     # A small positive buffer on original helps with floating point issues for containment.
                    if boundary_polygon.buffer(EPS*100).contains(rotated_back_cell.representative_point()):
                         final_decomposed_cells.append(rotated_back_cell)
                    # else:
                    #     print(f"Debug: Cell {rotated_back_cell.wkt[:50]} not contained in original after un-rotation.")
    return final_decomposed_cells

