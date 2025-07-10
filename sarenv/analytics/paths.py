# sarenv/analytics/paths.py
"""
Collection of coverage path generation algorithms for drones.
"""
import numpy as np
from shapely.geometry import LineString, Point
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
    budget = kwargs.get('budget')
    loop_spacing = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    a = loop_spacing / (2 * np.pi)
    num_rotations = max_radius / loop_spacing
    theta_max = num_rotations * 2 * np.pi
    approx_path_length = 0.5 * a * (theta_max * np.sqrt(1 + theta_max**2) + np.log(theta_max + np.sqrt(1 + theta_max**2))) if theta_max > 0 else 0
    num_points = int(approx_path_length / path_point_spacing_m) if path_point_spacing_m > 0 else 2000
    theta = np.linspace(0, theta_max, max(2, num_points))
    radius = np.clip(a * theta, 0, max_radius)
    full_path = LineString(zip(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta), strict=True))
    paths = split_path_for_drones(full_path, num_drones)
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

# Add **kwargs here as well for consistency
def generate_concentric_circles_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, transition_distance_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    radius_increment = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    path_points, current_radius = [], radius_increment
    while current_radius <= max_radius:
        transition_angle_rad = transition_distance_m / current_radius if current_radius > 0 else np.radians(45)
        arc_length = current_radius * (2 * np.pi - transition_angle_rad)
        num_points_circle = max(2, int(arc_length / path_point_spacing_m))
        theta = np.linspace(0, 2 * np.pi - transition_angle_rad, num_points_circle)
        path_points.extend(zip(center_x + current_radius * np.cos(theta), center_y + current_radius * np.sin(theta), strict=True))
        next_radius = current_radius + radius_increment
        if next_radius <= max_radius:
            path_points.append((center_x + next_radius, center_y))
        else:
            final_theta_points = max(2, int((current_radius * transition_angle_rad) / path_point_spacing_m))
            final_theta = np.linspace(2 * np.pi - transition_angle_rad, 2 * np.pi, final_theta_points)
            path_points.extend(zip(center_x + current_radius * np.cos(final_theta), center_y + current_radius * np.sin(final_theta), strict=True))
        current_radius = next_radius
    full_path = LineString(path_points) if path_points else LineString()
    paths = split_path_for_drones(full_path, num_drones)
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

# Add **kwargs here too to handle arguments like 'transition_distance_m'
def generate_pizza_zigzag_path(center_x: float, center_y: float, max_radius: float, num_drones: int, fov_deg: float, altitude: float, overlap: float, path_point_spacing_m: float, border_gap_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    paths, section_angle_rad = [], 2 * np.pi / num_drones
    pass_width = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    for i in range(num_drones):
        base_start_angle, base_end_angle = i * section_angle_rad, (i + 1) * section_angle_rad
        points, radius, direction = [(center_x, center_y)], pass_width, 1
        while radius <= max_radius:
            angular_offset_rad = border_gap_m / radius if radius > 0 else 0
            start_angle, end_angle = base_start_angle + angular_offset_rad, base_end_angle - angular_offset_rad
            if start_angle >= end_angle:
                radius += pass_width
                continue
            arc_length = radius * (end_angle - start_angle)
            num_arc_points = max(2, int(arc_length / path_point_spacing_m))
            current_arc_angles = np.linspace(start_angle, end_angle, num_arc_points) if direction == 1 else np.linspace(end_angle, start_angle, num_arc_points)
            points.extend(zip(center_x + radius * np.cos(current_arc_angles), center_y + radius * np.sin(current_arc_angles), strict=True))
            radius += pass_width
            direction *= -1
        if len(points) > 1:
            paths.append(LineString(points))
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

def generate_greedy_path(center_x: float, center_y: float, num_drones: int, probability_map: np.ndarray, bounds: tuple, max_radius: float, **kwargs) -> list[LineString]:
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds

    if maxx <= minx or maxy <= miny:
        return [LineString() for _ in range(num_drones)]

    # Pre-compute coordinate mappings for efficiency
    dx = (maxx - minx) / width
    dy = (maxy - miny) / height
    x_offset = minx + dx / 2
    y_offset = miny + dy / 2

    # Convert the real-world starting coordinates to grid indices
    start_col = np.clip(int((center_x - minx) / dx), 0, width - 1)
    start_row = np.clip(int((center_y - miny) / dy), 0, height - 1)
    start_pos = (start_row, start_col)

    # Pre-compute squared max radius for faster distance checks
    max_radius_sq = max_radius * max_radius

    # Grid-based simulation
    visited = np.zeros((height, width), dtype=bool)

    # Initialize drone positions efficiently
    current_positions = [start_pos]
    for i in range(1, num_drones):
        angle = 2 * np.pi * i / num_drones
        offset_r = min(2, height // 10)
        offset_c = min(2, width // 10)
        new_r = np.clip(start_pos[0] + int(offset_r * np.sin(angle)), 0, height - 1)
        new_c = np.clip(start_pos[1] + int(offset_c * np.cos(angle)), 0, width - 1)
        current_positions.append((new_r, new_c))

    # Store paths as lists of (row, col) indices
    paths = [[] for _ in range(num_drones)]
    for i, pos in enumerate(current_positions):
        paths[i].append(pos)
        visited[pos] = True

    # Pre-define neighbor offsets
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    max_iterations = height * width // num_drones
    # Estimate max_iterations based on budget (in meters) and minimum move length (x_offset)
    budget = kwargs.get('budget')
    if budget is not None and budget > 0:
        # Each move is at least x_offset meters (cell width)
        max_iterations = (budget // dx ) // num_drones

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        for i in range(num_drones):
            current_r, current_c = current_positions[i]
            valid_neighbors = []

            # Check all 8 neighbors
            for dr, dc in neighbor_offsets:
                nr, nc = current_r + dr, current_c + dc
                # Bounds check
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                
                # Distance check using squared distance
                world_x = x_offset + nc * dx
                world_y = y_offset + nr * dy
                dist_sq = (world_x - center_x) ** 2 + (world_y - center_y) ** 2
                if dist_sq >= max_radius_sq:
                    continue

                # Allow revisiting cells - add them with 0 probability
                if visited[nr, nc]:
                    valid_neighbors.append(((nr, nc), 0))
                else:
                    # Store valid neighbor and its probability
                    prob = probability_map[nr, nc]
                    valid_neighbors.append(((nr, nc), prob))

            # Choose next position based on probabilities
            if valid_neighbors:
                # Select the first neighbor with the maximum probability
                best_neighbor, _ = max(valid_neighbors, key=lambda x: x[1])
                next_pos = best_neighbor
                current_positions[i] = best_neighbor
                paths[i].append(best_neighbor)
                visited[next_pos] = True

    # --- Convert grid paths back to real-world coordinate paths ---
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            # Convert (row, col) indices directly to world coordinates
            line_paths.append(LineString([(x_offset + c * dx, y_offset + r * dy) for r, c in drone_path_indices]))
        else:
            line_paths.append(LineString())
    budget = kwargs.get('budget')

    # # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(line_paths, budget / num_drones)
    else:
        paths = line_paths
    return paths

def generate_random_walk_path(
    center_x: float,
    center_y: float,
    num_drones: int,
    probability_map,
    **kwargs
) -> list[LineString]:
    return generate_greedy_path(
        center_x=center_x,
        center_y=center_y,
        num_drones=num_drones,
        probability_map=np.zeros_like(probability_map),
        **kwargs
    )

def restrict_path_length(line: LineString, max_length: float) -> LineString:
    if isinstance(line, list):
        return [restrict_path_length(l, max_length) for l in line]
    if line.is_empty or max_length is None or max_length <= 0 or line.length <= max_length:
        return line
    return substring(line, 0, max_length)
