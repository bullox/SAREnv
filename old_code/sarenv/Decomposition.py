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
    segments = set()  # Use a set to store string representations to ensure uniqueness

    def add_segments_from_coords(coords_list: List[Coord], segment_set: set):
        for i in range(len(coords_list) - 1):
            p1 = coords_list[i]
            p2 = coords_list[i + 1]
            # Normalize segment representation (e.g., sort points) to avoid duplicates like (A,B) and (B,A)
            # For BCD, direction might matter if not handled by edge properties (min_x, max_x)
            # However, Shapely LineStrings are ordered. We'll rely on their properties later.
            # To avoid issues with floating point in set keys, use wkt for uniqueness check
            # but store the actual LineString objects.
            # We need to ensure segments are not duplicated if they are part of multiple rings or shared.
            # For BCD, we usually consider directed edges as they appear on boundary.
            ls = LineString([p1, p2])
            if ls.length > EPS:  # Avoid zero-length segments
                # Store by WKT to ensure uniqueness of geometry, not object instance
                segment_set.add(ls.wkt)

    add_segments_from_coords(list(polygon.exterior.coords), segments)
    for interior in polygon.interiors:
        add_segments_from_coords(list(interior.coords), segments)

    # Convert WKT back to LineString objects
    return [LineString(wkt_seg) for wkt_seg in segments]


def get_segment_y_at_x(
    segment: LineString,
    x_coord: float,
    default_y_if_no_intersect: Optional[float] = None,
) -> Optional[float]:
    """Calculates the y-coordinate of a segment at a given x_coord."""
    min_x, min_y, max_x, max_y = segment.bounds

    if not (
        min_x - EPS <= x_coord <= max_x + EPS
    ):  # x_coord outside segment's x-range (approx)
        if feq(min_x, x_coord):  # x_coord is at start of segment's x-range
            # Check if segment is vertical or starts/ends here
            if feq(segment.coords[0][0], x_coord):
                return segment.coords[0][1]
            if feq(segment.coords[1][0], x_coord):
                return segment.coords[1][1]
        elif feq(max_x, x_coord):  # x_coord is at end of segment's x-range
            if feq(segment.coords[0][0], x_coord):
                return segment.coords[0][1]
            if feq(segment.coords[1][0], x_coord):
                return segment.coords[1][1]
        return default_y_if_no_intersect

    # If segment is vertical at x_coord
    if feq(segment.coords[0][0], x_coord) and feq(segment.coords[1][0], x_coord):
        return (
            segment.coords[0][1] + segment.coords[1][1]
        ) / 2.0  # Midpoint for vertical segment (or min/max)

    # Create a long vertical line for intersection
    vertical_line = LineString([(x_coord, min_y - 1000), (x_coord, max_y + 1000)])
    intersection = segment.intersection(vertical_line)

    if isinstance(intersection, Point):
        return intersection.y
    elif not intersection.is_empty:  # E.g. LineString overlap (horizontal segment)
        return intersection.bounds[1]  # y-coordinate of the horizontal segment

    return default_y_if_no_intersect


def sort_active_edges_L(L: List[LineString], current_x: float):
    """Sorts active edges in L by their y-intersection at current_x."""
    if not L:
        return

    def sort_key(segment: LineString):
        y_val = get_segment_y_at_x(segment, current_x)
        return (
            y_val if y_val is not None else float("inf")
        )  # Put non-intersecting last (should not happen)

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
    if not lower_chain or not upper_chain:
        return None

    # Ensure chains are not just single points if they are supposed to form a line
    # For a valid cell, each chain should ideally have at least 2 points if it spans some x distance.
    if (
        len(lower_chain) < 1 or len(upper_chain) < 1
    ):  # Allow single point if chains start/end there
        # This condition might need adjustment based on how chains are built.
        # A cell must have some extent.
        pass

    # Polygon coords: lower chain, then reversed upper chain
    # The first point of lower_chain should connect to last point of reversed_upper_chain (which is first of upper_chain)
    # The last point of lower_chain should connect to first point of reversed_upper_chain (which is last of upper_chain)

    poly_coords = lower_chain + upper_chain[::-1]

    if len(poly_coords) < 3:  # Need at least 3 unique points for a polygon
        return None

    # Remove consecutive duplicates from the combined chain
    final_ring_coords = []
    if poly_coords:
        final_ring_coords.append(poly_coords[0])
        for i in range(1, len(poly_coords)):
            if not (
                feq(poly_coords[i][0], final_ring_coords[-1][0])
                and feq(poly_coords[i][1], final_ring_coords[-1][1])
            ):
                final_ring_coords.append(poly_coords[i])

    # Ensure closure for Shapely Polygon constructor (first and last point same)
    if len(final_ring_coords) >= 3:
        if not (
            feq(final_ring_coords[0][0], final_ring_coords[-1][0])
            and feq(final_ring_coords[0][1], final_ring_coords[-1][1])
        ):
            final_ring_coords.append(final_ring_coords[0])

    if len(final_ring_coords) < 4:  # e.g., A,B,C,A needs 4 points
        return None

    try:
        cell_poly = Polygon(final_ring_coords)
        # A very small buffer can sometimes fix minor self-intersections or invalid topology
        # cell_poly = cell_poly.buffer(final_cleanup_tolerance).buffer(-final_cleanup_tolerance) # Optional
        # cell_poly = cell_poly.simplify(EPS/10, preserve_topology=True)

        if (
            cell_poly.is_valid and cell_poly.area > EPS * 10
        ):  # More stringent area check
            return cell_poly
    except Exception:
        return None  # Error during polygon creation
    return None


def boustrophedon_decomposition(boundary_polygon: Polygon, sweep_angle_deg: float = 0.0) -> List[Polygon]:
    if not boundary_polygon.is_valid or boundary_polygon.is_empty:
        return []

    effective_rotation_angle = -sweep_angle_deg
    # Ensure origin is a Point for shapely.rotate
    # Using centroid might be problematic if centroid is outside for weird shapes,
    # but it's standard. (0,0) is another option if geometry is centered or relative.
    # For BCD, the absolute coordinates matter for the sweep, so (0,0) might be more consistent
    # if the polygon isn't always centered. Let's assume (0,0) for rotation.
    rotation_origin = Point(0,0) 
    poly_to_sweep = rotate(boundary_polygon, effective_rotation_angle, origin=rotation_origin, use_radians=False)
    
    # It's crucial that poly_to_sweep is valid after rotation for .contains() to work reliably.
    if not poly_to_sweep.is_valid:
        poly_to_sweep = poly_to_sweep.buffer(0) # Try to fix validity
        if not poly_to_sweep.is_valid:
            # print("Warning: Polygon became invalid after rotation and buffer(0).")
            return []


    if not poly_to_sweep.exterior.is_ccw:
         ext_coords = list(poly_to_sweep.exterior.coords)[::-1]
         holes = []
         for hole_ring in poly_to_sweep.interiors:
             hr_coords = list(hole_ring.coords)
             if Polygon(hr_coords).exterior.is_ccw : # check if hole is ccw
                 holes.append(hr_coords[::-1])
             else:
                 holes.append(hr_coords)
         poly_to_sweep = Polygon(ext_coords, holes)


    all_coords = list(poly_to_sweep.exterior.coords)
    for interior in poly_to_sweep.interiors:
        all_coords.extend(list(interior.coords))
    
    unique_vertices = sorted(list(set(all_coords)), key=lambda p: (p[0], p[1]))
    event_x_coords = sorted(list(set([coord[0] for coord in unique_vertices])))

    all_polygon_segments = []
    ext_coords_list = list(poly_to_sweep.exterior.coords)
    for i in range(len(ext_coords_list) - 1):
        seg = LineString([ext_coords_list[i], ext_coords_list[i+1]])
        if seg.length > EPS: all_polygon_segments.append(seg)
    for hole in poly_to_sweep.interiors:
        hole_coords_list = list(hole.coords)
        for i in range(len(hole_coords_list) - 1):
            seg = LineString([hole_coords_list[i], hole_coords_list[i+1]])
            if seg.length > EPS: all_polygon_segments.append(seg)

    L_active_edges: List[LineString] = []
    open_cells: List[OpenCell] = []
    closed_polygons: List[Polygon] = []

    if not event_x_coords: return []
    # Initialize previous_x carefully. If first event_x is poly_min_x, this is fine.
    # Using the first event x allows initialization of L correctly at the first step.
    previous_x = event_x_coords[0] 
    
    # Buffer poly_to_sweep slightly for contains check to be more tolerant to boundary points
    # This is a common technique but adjust EPS_buffer carefully.
    EPS_BUFFER_CONTAINS = EPS * 10 
    tolerant_poly_to_sweep = poly_to_sweep.buffer(EPS_BUFFER_CONTAINS)


    for idx_event_x, current_x in enumerate(event_x_coords):
        # --- 3.A. Extend existing open cells' chains to current_x ---
        if idx_event_x > 0 and not feq(current_x, previous_x) : # Only extend if x has advanced
            for cell in open_cells:
                lower_seg = cell['lower_segment']
                upper_seg = cell['upper_segment']
                if lower_seg and upper_seg:
                    y_lower_curr = get_segment_y_at_x(lower_seg, current_x)
                    y_upper_curr = get_segment_y_at_x(upper_seg, current_x)
                    if y_lower_curr is not None: add_point_to_chain(cell['lower_chain'], (current_x, y_lower_curr))
                    if y_upper_curr is not None: add_point_to_chain(cell['upper_chain'], (current_x, y_upper_curr))
        
        # --- 3.B. Update L_active_edges based on segments active at current_x ---
        # Rebuild L at each step for robustness, or use careful incremental add/remove
        # Current incremental logic:
        temp_L = []
        # Keep segments from old L that continue past current_x
        for seg in L_active_edges:
            if seg.bounds[2] > current_x + EPS: # max_x of segment > current_x
                temp_L.append(seg)
        L_active_edges = temp_L

        # Add segments from all_polygon_segments that start at current_x
        for seg in all_polygon_segments:
            min_sx, _, max_sx, _ = seg.bounds
            if feq(min_sx, current_x) and max_sx > current_x + EPS: # Starts at current_x, goes right
                is_already_in_L = any(l_seg.equals_exact(seg, EPS) for l_seg in L_active_edges)
                if not is_already_in_L:
                    L_active_edges.append(seg)
        
        # --- 3.C. Sort L_active_edges ---
        sort_active_edges_L(L_active_edges, current_x)

        # --- 3.D. Reconcile open_cells with the new L_active_edges ---
        next_round_open_cells: List[OpenCell] = []
        processed_old_cells_indices = set()

        for i in range(len(L_active_edges) - 1):
            seg_L_lower = L_active_edges[i]
            seg_L_upper = L_active_edges[i+1]
            
            matched_old_cell = None
            for cell_idx, old_cell in enumerate(open_cells):
                if cell_idx in processed_old_cells_indices: continue
                if old_cell['lower_segment'] == seg_L_lower and old_cell['upper_segment'] == seg_L_upper:
                    matched_old_cell = old_cell
                    processed_old_cells_indices.add(cell_idx)
                    break
            
            if matched_old_cell:
                next_round_open_cells.append(matched_old_cell)
            else:
                y_L_lower_curr = get_segment_y_at_x(seg_L_lower, current_x)
                y_L_upper_curr = get_segment_y_at_x(seg_L_upper, current_x)

                if y_L_lower_curr is not None and y_L_upper_curr is not None and \
                   y_L_upper_curr > y_L_lower_curr + EPS:
                    
                    mid_y_of_slice = (y_L_lower_curr + y_L_upper_curr) / 2.0
                    test_point_in_slice = Point(current_x, mid_y_of_slice)

                    # Use the slightly buffered polygon for the contains check
                    if tolerant_poly_to_sweep.contains(test_point_in_slice):
                        new_cell: OpenCell = {
                            'lower_chain': [(current_x, y_L_lower_curr)],
                            'upper_chain': [(current_x, y_L_upper_curr)],
                            'lower_segment': seg_L_lower,
                            'upper_segment': seg_L_upper
                        }
                        next_round_open_cells.append(new_cell)
        
        for cell_idx, old_cell_to_close in enumerate(open_cells):
            if cell_idx not in processed_old_cells_indices:
                # Chains for old_cell_to_close were extended to current_x in step 3.A
                poly = form_polygon_from_chains(old_cell_to_close['lower_chain'], old_cell_to_close['upper_chain'])
                if poly: closed_polygons.append(poly)
        
        open_cells = next_round_open_cells
        if idx_event_x < len(event_x_coords) : # Check to prevent index out of bounds if loop structure changes
            previous_x = current_x # Update previous_x for the next iteration's chain extension

    # --- 4. Finalization: Close any remaining open cells ---
    final_x = poly_to_sweep.bounds[2]
    if event_x_coords and not feq(final_x, previous_x) and final_x > previous_x + EPS : # If there's a final segment to sweep
        for cell in open_cells: # Extend chains to final_x
            lower_seg, upper_seg = cell['lower_segment'], cell['upper_segment']
            if lower_seg and upper_seg:
                y_l_final = get_segment_y_at_x(lower_seg, final_x)
                y_u_final = get_segment_y_at_x(upper_seg, final_x)
                if y_l_final is not None: add_point_to_chain(cell['lower_chain'], (final_x, y_l_final))
                if y_u_final is not None: add_point_to_chain(cell['upper_chain'], (final_x, y_u_final))

    for cell in open_cells: # Now form polygons for all that were open
        poly = form_polygon_from_chains(cell['lower_chain'], cell['upper_chain'])
        if poly: closed_polygons.append(poly)
    
    # --- 5. Rotate cells back and return ---
    final_decomposed_cells: List[Polygon] = []
    if closed_polygons:
        unrotation_angle = sweep_angle_deg 
        for cell_poly in closed_polygons:
            # cell_poly should be valid from form_polygon_from_chains
            rotated_back_poly = rotate(cell_poly, unrotation_angle, origin=rotation_origin, use_radians=False)
            if rotated_back_poly.is_valid and rotated_back_poly.area > EPS:
                 final_decomposed_cells.append(rotated_back_poly)
    
    return final_decomposed_cells

from shapely.geometry import Polygon, Point

# Assuming the boustrophedon_decomposition function and its helpers
# (EPS, VertexData, OpenCell, Coord, Chain, feq, get_polygon_segments,
#  get_segment_y_at_x, sort_active_edges_L, add_point_to_chain,
#  form_polygon_from_chains, rotate, etc.) are defined above this example.

# --- Example Usage ---

if __name__ == "__main__":
    print("Boustrophedon Decomposition Example")

    # --- Example 1: Simple Square ---
    print("\n--- Example 1: Simple Square ---")
    square_coords = [(0, 0), (0, 5), (5, 5), (5, 0)]
    square_polygon = Polygon(square_coords)

    print(f"Input Polygon WKT: {square_polygon.wkt}")
    decomposed_cells_square = boustrophedon_decomposition(
        square_polygon, sweep_angle_deg=0.0
    )

    print(f"Number of decomposed cells: {len(decomposed_cells_square)}")
    total_area_decomposed = 0
    for i, cell in enumerate(decomposed_cells_square):
        print(
            f"  Cell {i}: Area = {cell.area:.2f}, WKT = {cell.wkt[:70]}..."
        )  # Print partial WKT
        if not cell.is_valid:
            print(f"    WARNING: Cell {i} is invalid!")
        total_area_decomposed += cell.area
    print(
        f"Original Area: {square_polygon.area:.2f}, Total Decomposed Area: {total_area_decomposed:.2f}"
    )

    # --- Example 2: U-Shaped Polygon ---
    print("\n--- Example 2: U-Shaped Polygon ---")
    u_shape_coords = [(0, 0), (0, 6), (2, 6), (2, 2), (4, 2), (4, 6), (6, 6), (6, 0)]
    u_shape_polygon = Polygon(u_shape_coords)

    print(f"Input Polygon WKT: {u_shape_polygon.wkt}")
    decomposed_cells_u_shape = boustrophedon_decomposition(
        u_shape_polygon, sweep_angle_deg=0.0
    )

    print(f"Number of decomposed cells: {len(decomposed_cells_u_shape)}")
    total_area_decomposed = 0
    for i, cell in enumerate(decomposed_cells_u_shape):
        print(f"  Cell {i}: Area = {cell.area:.2f}, WKT = {cell.wkt[:70]}...")
        if not cell.is_valid:
            print(f"    WARNING: Cell {i} is invalid!")
        total_area_decomposed += cell.area
    print(
        f"Original Area: {u_shape_polygon.area:.2f}, Total Decomposed Area: {total_area_decomposed:.2f}"
    )

    # --- Example 3: Polygon with a Hole ---
    print("\n--- Example 3: Polygon with a Hole ---")
    exterior_coords = [(0, 0), (0, 9), (10, 10), (10, 0)]
    interior_coords = [(2, 2), (2, 4), (4, 5), (4, 2)]  # Clockwise for Shapely hole
    polygon_with_hole = Polygon(exterior_coords, [interior_coords])

    print(f"Input Polygon WKT: {polygon_with_hole.wkt}")
    if not polygon_with_hole.is_valid:
        print("  WARNING: Input polygon with hole is initially invalid!")

    decomposed_cells_hole = boustrophedon_decomposition(
        polygon_with_hole, sweep_angle_deg=0.0
    )

    print(f"Number of decomposed cells: {len(decomposed_cells_hole)}")
    total_area_decomposed = 0
    for i, cell in enumerate(decomposed_cells_hole):
        print(f"  Cell {i}: Area = {cell.area:.2f}, WKT = {cell.wkt[:70]}...")
        if not cell.is_valid:
            print(f"    WARNING: Cell {i} is invalid!")
        total_area_decomposed += cell.area
    print(
        f"Original Area: {polygon_with_hole.area:.2f}, Total Decomposed Area: {total_area_decomposed:.2f}"
    )

    # --- Example 4: U-Shaped Polygon with a different sweep angle (e.g., 45 degrees) ---
    print("\n--- Example 4: U-Shaped Polygon with 45-degree sweep ---")
    # Using the same u_shape_polygon from Example 2
    print(f"Input Polygon WKT: {u_shape_polygon.wkt}")
    decomposed_cells_u_shape_angled = boustrophedon_decomposition(
        u_shape_polygon, sweep_angle_deg=45.0
    )

    print(
        f"Number of decomposed cells (45-deg sweep): {len(decomposed_cells_u_shape_angled)}"
    )
    total_area_decomposed = 0
    for i, cell in enumerate(decomposed_cells_u_shape_angled):
        print(f"  Cell {i}: Area = {cell.area:.2f}, WKT = {cell.wkt[:70]}...")
        if not cell.is_valid:
            print(f"    WARNING: Cell {i} is invalid!")
        total_area_decomposed += cell.area
    print(
        f"Original Area: {u_shape_polygon.area:.2f}, Total Decomposed Area (45-deg sweep): {total_area_decomposed:.2f}"
    )

    # --- Example 5: A more complex shape that might test edge cases ---
    print("\n--- Example 5: More Complex Polygon ---")
    complex_coords = [
        (0, 0),
        (0, 1),
        (1, 2),
        (2, 1),
        (3, 2),
        (4, 1),
        (4, 0),
        (3, 0.5),
        (2, 0),
        (1, 0.5),
    ]
    complex_polygon = Polygon(complex_coords)
    print(f"Input Polygon WKT: {complex_polygon.wkt}")
    decomposed_cells_complex = boustrophedon_decomposition(
        complex_polygon, sweep_angle_deg=0.0
    )

    print(f"Number of decomposed cells: {len(decomposed_cells_complex)}")
    total_area_decomposed = 0
    for i, cell in enumerate(decomposed_cells_complex):
        print(f"  Cell {i}: Area = {cell.area:.2f}, WKT = {cell.wkt[:70]}...")
        if not cell.is_valid:
            print(f"    WARNING: Cell {i} is invalid!")
        total_area_decomposed += cell.area
    print(
        f"Original Area: {complex_polygon.area:.2f}, Total Decomposed Area: {total_area_decomposed:.2f}"
    )

    # To visualize (optional, requires matplotlib or other plotting library):
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_polygon, plot_points, plot_line

    fig, ax = plt.subplots()
    # plot_polygon(polygon_with_hole, ax=ax, add_points=False, color='lightgray', alpha=0.7)
    for i, cell in enumerate(decomposed_cells_u_shape_angled):
        plot_polygon(cell, ax=ax, add_points=False, alpha=0.5, label=f"Cell {i}")
    ax.set_title("Boustrophedon Decomposition (Polygon with Hole)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
