# sarenv/analytics/paths.py
"""
Collection of coverage path generation algorithms for drones.
"""
import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring

def split_path_for_drones(path: LineString, num_drones: int) -> list[LineString]:
    if num_drones <= 1 or path.is_empty or path.length == 0:
        return [path]
    segments = []
    segment_length = path.length / num_drones
    for i in range(num_drones):
        segments.append(substring(path, i * segment_length, (i + 1) * segment_length))
    return segments

# Add **kwargs to accept and ignore unused arguments
def generate_spiral_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, **kwargs) -> list[LineString]:
    loop_spacing = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    a = loop_spacing / (2 * np.pi)
    num_rotations = max_radius / loop_spacing
    theta_max = num_rotations * 2 * np.pi
    approx_path_length = 0.5 * a * (theta_max * np.sqrt(1 + theta_max**2) + np.log(theta_max + np.sqrt(1 + theta_max**2))) if theta_max > 0 else 0
    num_points = int(approx_path_length / path_point_spacing_m) if path_point_spacing_m > 0 else 2000
    theta = np.linspace(0, theta_max, max(2, num_points))
    radius = np.clip(a * theta, 0, max_radius)
    full_path = LineString(zip(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)))
    return split_path_for_drones(full_path, num_drones)

# Add **kwargs here as well for consistency
def generate_concentric_circles_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, transition_distance_m: float, **kwargs) -> list[LineString]:
    radius_increment = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    path_points, current_radius = [], radius_increment
    while current_radius <= max_radius:
        transition_angle_rad = transition_distance_m / current_radius if current_radius > 0 else np.radians(45)
        arc_length = current_radius * (2 * np.pi - transition_angle_rad)
        num_points_circle = max(2, int(arc_length / path_point_spacing_m))
        theta = np.linspace(0, 2 * np.pi - transition_angle_rad, num_points_circle)
        path_points.extend(zip(center_x + current_radius * np.cos(theta), center_y + current_radius * np.sin(theta)))
        next_radius = current_radius + radius_increment
        if next_radius <= max_radius: path_points.append((center_x + next_radius, center_y))
        else:
            final_theta_points = max(2, int((current_radius * transition_angle_rad) / path_point_spacing_m))
            final_theta = np.linspace(2 * np.pi - transition_angle_rad, 2 * np.pi, final_theta_points)
            path_points.extend(zip(center_x + current_radius * np.cos(final_theta), center_y + current_radius * np.sin(final_theta)))
        current_radius = next_radius
    full_path = LineString(path_points) if path_points else LineString()
    return split_path_for_drones(full_path, num_drones)

# Add **kwargs here too to handle arguments like 'transition_distance_m'
def generate_pizza_zigzag_path(center_x: float, center_y: float, max_radius: float, num_drones: int, fov_deg: float, altitude: float, overlap: float, path_point_spacing_m: float, border_gap_m: float, **kwargs) -> list[LineString]:
    paths, section_angle_rad = [], 2 * np.pi / num_drones
    pass_width = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    for i in range(num_drones):
        base_start_angle, base_end_angle = i * section_angle_rad, (i + 1) * section_angle_rad
        points, radius, direction = [(center_x, center_y)], pass_width, 1
        while radius <= max_radius:
            angular_offset_rad = border_gap_m / radius if radius > 0 else 0
            start_angle, end_angle = base_start_angle + angular_offset_rad, base_end_angle - angular_offset_rad
            if start_angle >= end_angle:
                radius += pass_width; continue
            arc_length = radius * (end_angle - start_angle)
            num_arc_points = max(2, int(arc_length / path_point_spacing_m))
            current_arc_angles = np.linspace(start_angle, end_angle, num_arc_points) if direction == 1 else np.linspace(end_angle, start_angle, num_arc_points)
            points.extend(zip(center_x + radius * np.cos(current_arc_angles), center_y + radius * np.sin(current_arc_angles)))
            radius += pass_width; direction *= -1
        if len(points) > 1: paths.append(LineString(points))
    return paths

def generate_greedy_path(center_x: float, center_y: float, num_drones: int, probability_map:np.ndarray, bounds:tuple, **kwargs) -> list[LineString]:
    """
    Generates paths for multiple drones using a greedy algorithm on a probability map.

    Each drone starts at a specified center coordinate and iteratively moves to the adjacent
    (including diagonal) grid cell with the highest probability value that has not been
    visited by any other drone. The simulation runs for a number of steps equal to the
    total number of cells in the map.
    """
    height, width = probability_map.shape
    num_steps = (height * width)*10
    minx, miny, maxx, maxy = bounds

    if maxx <= minx or maxy <= miny:
        # Return empty paths if bounds are invalid
        return [LineString() for _ in range(num_drones)]

    # Create arrays that map a grid index to the real-world coordinate of the cell's center
    x_map = np.linspace(minx + (maxx - minx) / (2 * width), maxx - (maxx - minx) / (2 * width), width)
    y_map = np.linspace(miny + (maxy - miny) / (2 * height), maxy - (maxy - miny) / (2 * height), height)

    # Convert the real-world starting coordinates to grid indices
    start_col = int(((center_x - minx) / (maxx - minx)) * width)
    start_row = int(((center_y - miny) / (maxy - miny)) * height)

    # Clamp indices to ensure they are within the valid grid dimensions
    start_pos = (np.clip(start_row, 0, height - 1), np.clip(start_col, 0, width - 1))

    # --- Grid-based simulation starts here ---
    visited = np.zeros_like(probability_map, dtype=bool)
    current_positions = [start_pos] * num_drones
    # Store paths as lists of (row, col) indices
    paths = [[pos] for pos in current_positions]
    visited[start_pos] = True

    # Loop for a fixed number of steps
    for _ in range(num_steps):
        for i in range(num_drones):
            current_r, current_c = current_positions[i]
            best_neighbor = None
            max_prob = -np.inf

            # Evaluate 8 neighbors in the grid
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = current_r + dr, current_c + dc

                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc]:
                        if probability_map[nr, nc] > max_prob:
                            max_prob = probability_map[nr, nc]
                            best_neighbor = (nr, nc)

            if best_neighbor is not None:
                current_positions[i] = best_neighbor
                paths[i].append(best_neighbor)
                visited[best_neighbor] = True

    # --- Convert grid paths back to real-world coordinate paths ---
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            # For each (row, col) in the path, look up the corresponding (x, y) coordinate
            world_coords = [(x_map[c], y_map[r]) for r, c in drone_path_indices]
            line_paths.append(LineString(world_coords))
        else:
            line_paths.append(LineString())

    return line_paths
