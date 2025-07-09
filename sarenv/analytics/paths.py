# sarenv/analytics/paths.py
"""
Collection of coverage path generation algorithms for drones.
"""
import random
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

def calculate_max_distance_in_paths(paths: list[LineString]) -> float:
    """
    Calculate the maximum distance covered by any path in the list.
    Args:
        paths (list[LineString]): List of LineString paths.
    Returns:
        float: Maximum distance covered by any path.
    """
    if not paths:
        return 0.0
    max_distance = 0.0
    for path in paths:
        if not path.is_empty:
            distance = path.length
            if distance > max_distance:
                max_distance = distance
    return max_distance

def resample_path_to_length(path, target_length):
    if path.is_empty:
        return path
    original_length = path.length
    if original_length <= target_length:
        return path
    num_points = max(2, int(target_length / (original_length / len(path.coords))))
    distances = np.linspace(0, original_length, num_points)
    new_points = [path.interpolate(distance) for distance in distances]
    return LineString(new_points)

def extend_path_to_length(path, target_length):
    if path.is_empty:
        return path
    current_length = path.length
    coords = list(path.coords)
    while current_length < target_length:
        if len(coords) < 2:
            break
        last_point = Point(coords[-1])
        second_last_point = Point(coords[-2])
        dx = last_point.x - second_last_point.x
        dy = last_point.y - second_last_point.y
        segment_length = (dx**2 + dy**2)**0.5
        if segment_length == 0:
            break
        needed_length = target_length - current_length
        repeat_factor = int(np.ceil(needed_length / segment_length))
        for _ in range(repeat_factor):
            new_point = (coords[-1][0] + dx, coords[-1][1] + dy)
            coords.append(new_point)
            current_length += segment_length
            if current_length >= target_length:
                break
    return LineString(coords)

def split_path_for_drones(full_path, num_drones):
    if full_path.is_empty or num_drones <= 1:
        return [full_path]
    total_length = full_path.length
    segment_length = total_length / num_drones
    paths = []
    for i in range(num_drones):
        start_dist = i * segment_length
        end_dist = (i + 1) * segment_length
        segment_points = []
        distances = np.linspace(start_dist, end_dist, max(2, int(segment_length / (total_length / len(full_path.coords)))))
        for d in distances:
            segment_points.append(full_path.interpolate(d))
        paths.append(LineString(segment_points))
    return paths

def generate_spiral_path(center_x, center_y, max_radius, fov_deg, altitude, overlap, num_drones, path_point_spacing_m, max_length=None, **kwargs):
    loop_spacing = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    a = loop_spacing / (2 * np.pi)
    num_rotations = max_radius / loop_spacing
    theta_max = num_rotations * 2 * np.pi
    approx_path_length = 0.5 * a * (theta_max * np.sqrt(1 + theta_max**2) + np.log(theta_max + np.sqrt(1 + theta_max**2))) if theta_max > 0 else 0
    num_points = int(approx_path_length / path_point_spacing_m) if path_point_spacing_m > 0 else 2000
    theta = np.linspace(0, theta_max, max(2, num_points))
    radius = np.clip(a * theta, 0, max_radius)
    full_path = LineString(zip(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)))
    if max_length is not None and full_path.length > max_length:
        full_path = resample_path_to_length(full_path, max_length)
    return split_path_for_drones(full_path, num_drones)

def generate_concentric_circles_path(center_x, center_y, max_radius, fov_deg, altitude, overlap, num_drones, path_point_spacing_m, transition_distance_m, max_length=None, **kwargs):
    radius_increment = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    path_points, current_radius = [], radius_increment
    while current_radius <= max_radius:
        transition_angle_rad = transition_distance_m / current_radius if current_radius > 0 else np.radians(45)
        arc_length = current_radius * (2 * np.pi - transition_angle_rad)
        num_points_circle = max(2, int(arc_length / path_point_spacing_m))
        theta = np.linspace(0, 2 * np.pi - transition_angle_rad, num_points_circle)
        path_points.extend(zip(center_x + current_radius * np.cos(theta), center_y + current_radius * np.sin(theta)))
        next_radius = current_radius + radius_increment
        if next_radius <= max_radius:
            path_points.append((center_x + next_radius, center_y))
        else:
            final_theta_points = max(2, int((current_radius * transition_angle_rad) / path_point_spacing_m))
            final_theta = np.linspace(2 * np.pi - transition_angle_rad, 2 * np.pi, final_theta_points)
            path_points.extend(zip(center_x + current_radius * np.cos(final_theta), center_y + current_radius * np.sin(final_theta)))
        current_radius = next_radius
    full_path = LineString(path_points) if path_points else LineString()
    if max_length is not None and full_path.length > max_length:
        full_path = resample_path_to_length(full_path, max_length)
    return split_path_for_drones(full_path, num_drones)

def generate_pizza_zigzag_path(center_x, center_y, max_radius, num_drones, fov_deg, altitude, overlap, path_point_spacing_m, border_gap_m, max_length=None, **kwargs):
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
            points.extend(zip(center_x + radius * np.cos(current_arc_angles), center_y + radius * np.sin(current_arc_angles)))
            radius += pass_width
            direction *= -1
        if len(points) > 1:
            path = LineString(points)
            if max_length is not None and path.length > max_length:
                path = resample_path_to_length(path, max_length)
            paths.append(path)
    return paths

def generate_greedy_path(
    center_x, center_y, num_drones, probability_map, bounds, max_length=None, **kwargs
):
    import numpy as np
    from shapely.geometry import LineString

    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds

    if maxx <= minx or maxy <= miny:
        return [LineString() for _ in range(num_drones)]

    x_map = np.linspace(minx + (maxx - minx) / (2 * width), maxx - (maxx - minx) / (2 * width), width)
    y_map = np.linspace(miny + (maxy - miny) / (2 * height), maxy - (maxy - miny) / (2 * height), height)

    start_col = int(((center_x - minx) / (maxx - minx)) * width)
    start_row = int(((center_y - miny) / (maxy - miny)) * height)
    start_pos = (np.clip(start_row, 0, height - 1), np.clip(start_col, 0, width - 1))

    visited = np.zeros_like(probability_map, dtype=bool)
    current_positions = [start_pos] * num_drones
    paths = [[pos] for pos in current_positions]
    visited[start_pos] = True

    finished = [False] * num_drones

    while not all(finished):
        for i in range(num_drones):
            if finished[i]:
                continue
            current_r, current_c = current_positions[i]
            best_neighbor = None
            max_prob = -np.inf
            # Check 8 neighbors
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
            # Convert to real-world coords and check length
            world_coords = [(x_map[c], y_map[r]) for r, c in paths[i]]
            path = LineString(world_coords)
            if max_length is not None and path.length >= max_length:
                finished[i] = True
            # If no more moves, mark as finished
            elif best_neighbor is None:
                finished[i] = True

    # Convert all paths to LineStrings
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            world_coords = [(x_map[c], y_map[r]) for r, c in drone_path_indices]
            path = LineString(world_coords)
            if max_length is not None and path.length > max_length:
                # Truncate if slightly over
                num_points = max(2, int(len(path.coords) * max_length / path.length))
                distances = np.linspace(0, max_length, num_points)
                new_points = [path.interpolate(distance) for distance in distances]
                path = LineString(new_points)
            line_paths.append(path)
        else:
            line_paths.append(LineString())
    return line_paths

def generate_random_walk_path(
    center_x, center_y, num_drones, probability_map, bounds, max_length=None, N=10, **kwargs
):
    import numpy as np
    from shapely.geometry import LineString
    import random

    if probability_map is None:
        raise ValueError("A valid probability_map must be provided to define the grid.")
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds

    if maxx <= minx or maxy <= miny or width <= 1 or height <= 1:
        return [LineString() for _ in range(num_drones)]

    x_map = np.linspace(minx + (maxx - minx) / (2 * width), maxx - (maxx - minx) / (2 * width), width)
    y_map = np.linspace(miny + (maxy - miny) / (2 * height), maxy - (maxy - miny) / (2 * height), height)

    start_col = int(((center_x - minx) / (maxx - minx)) * width)
    start_row = int(((center_y - miny) / (maxy - miny)) * height)
    start_pos = (np.clip(start_row, 0, height - 1), np.clip(start_col, 0, width - 1))

    current_positions = [start_pos] * num_drones
    paths = [[pos] for pos in current_positions]

    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    directions = [random.choice(moves) for _ in range(num_drones)]
    steps_in_direction = [0] * num_drones

    finished = [False] * num_drones
    max_iterations = 100000
    iteration = 0

    while not all(finished) and iteration < max_iterations:
        for i in range(num_drones):
            if finished[i]:
                continue
            drone_path_indices = paths[i]
            if len(drone_path_indices) > 1:
                world_coords = [(x_map[c], y_map[r]) for r, c in drone_path_indices]
                path = LineString(world_coords)
                if max_length is not None and path.length >= max_length:
                    finished[i] = True
                    continue
            r, c = current_positions[i]
            # Change direction after N steps
            if steps_in_direction[i] >= N:
                directions[i] = random.choice(moves)
                steps_in_direction[i] = 0
            dr, dc = directions[i]
            nr, nc = r + dr, c + dc
            # If out of bounds, pick a new direction
            if not (0 <= nr < height and 0 <= nc < width):
                valid_move_found = False
                for _ in range(10):
                    directions[i] = random.choice(moves)
                    dr, dc = directions[i]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        valid_move_found = True
                        break
                if not valid_move_found:
                    finished[i] = True
                    continue
            current_positions[i] = (nr, nc)
            paths[i].append((nr, nc))
            steps_in_direction[i] += 1
        iteration += 1

    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            world_coords = [(x_map[c], y_map[r]) for r, c in drone_path_indices]
            path = LineString(world_coords)
            if max_length is not None and path.length > max_length:
                # Truncate if slightly over
                num_points = max(2, int(len(path.coords) * max_length / path.length))
                distances = np.linspace(0, max_length, num_points)
                new_points = [path.interpolate(distance) for distance in distances]
                path = LineString(new_points)
            line_paths.append(path)
        else:
            line_paths.append(LineString())
    return line_paths