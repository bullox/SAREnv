# sarenv/analytics/paths.py
"""
Collection of coverage path generation algorithms for drones.
"""
import random
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

def generate_greedy_path(center_x: float, center_y: float, num_drones: int, probability_map: np.ndarray, bounds: tuple, max_radius: float, **kwargs) -> list[LineString]:
    """
    Generates paths for multiple drones using a greedy algorithm on a probability map.

    Each drone starts at a specified center coordinate and iteratively moves to the adjacent
    (including diagonal) grid cell with the highest probability value that has not been
    visited by any drone. The simulation runs until no more moves are possible.
    """
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds

    if maxx <= minx or maxy <= miny:
        # Return empty paths if bounds are invalid
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

    # --- Grid-based simulation starts here ---
    visited = np.zeros((height, width), dtype=bool)

    # Initialize drone positions efficiently
    current_positions = [start_pos]
    for i in range(1, num_drones):
        # Place other drones in a small circle around the start position
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

    # Pre-define neighbor offsets to avoid array creation in loop
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    max_iterations = height * width // num_drones 
    # Main simulation loop - optimized for speed

    iteration = 0
    while  iteration < max_iterations:

        iteration += 1

        for i in range(num_drones):
            current_r, current_c = current_positions[i]
            best_neighbor = None
            best_prob = -1

            # Check all 8 neighbors directly
            for dr, dc in neighbor_offsets:
                nr, nc = current_r + dr, current_c + dc

                # Bounds check
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                # Distance check using squared distance
                world_x = x_offset + nc * dx
                world_y = y_offset + nr * dy
                dist_sq = (world_x - center_x) ** 2 + (world_y - center_y) ** 2
                if dist_sq > max_radius_sq:
                    continue

                # Skip if already visited
                if visited[nr, nc]:
                    continue

                # Check probability
                prob = probability_map[nr, nc]
                if prob > best_prob:
                    best_prob = prob
                    best_neighbor = (nr, nc)

            # Move drone if valid neighbor found
            if best_neighbor is not None:
                current_positions[i] = best_neighbor
                paths[i].append(best_neighbor)
                visited[best_neighbor] = True

    # --- Convert grid paths back to real-world coordinate paths ---
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            # Convert (row, col) indices directly to world coordinates
            world_coords = [(x_offset + c * dx, y_offset + r * dy) for r, c in drone_path_indices]
            line_paths.append(LineString(world_coords))
        else:
            line_paths.append(LineString())

    return line_paths

def generate_random_noncolliding_paths(center_x: float, center_y: float, max_radius: float, num_drones: int, path_point_spacing_m: float, max_points: int = 500, max_attempts: int = 1000, **kwargs) -> list[LineString]:
    # Constants inside the function
    SAFETY_DISTANCE = 5.0 # Minimum distance between paths
    MAX_TURN_ANGLE_DEG = 45.0
    MIN_RADIUS = 1.0
    SAFETY_DISTANCE_DISABLE_STEPS = 10  # Number of initial steps to disable safety distance

    sector_angle = 2 * np.pi / num_drones
    safety_distance = SAFETY_DISTANCE

    paths = []
    attempts = 0

    while len(paths) < num_drones and attempts < max_attempts:
        drone_idx = len(paths)
        base_angle = drone_idx * sector_angle
        points = [(center_x, center_y)]
        current_point = Point(center_x, center_y)
        prev_angle = base_angle

        for step in range(max_points):
            current_radius = current_point.distance(Point(center_x, center_y))
            min_next_radius = max(current_radius + 0.5 * path_point_spacing_m, MIN_RADIUS) if step > 0 else 0

            found_valid = False
            for _ in range(20):
                # Restrict angle for initial steps to stay in sector
                if step < SAFETY_DISTANCE_DISABLE_STEPS:
                    angle_offset = np.radians(np.random.uniform(-sector_angle/4, sector_angle/4))
                    new_angle = base_angle + angle_offset
                else:
                    angle_offset = np.radians(np.random.uniform(-MAX_TURN_ANGLE_DEG, MAX_TURN_ANGLE_DEG))
                    new_angle = prev_angle + angle_offset

                next_x = current_point.x + path_point_spacing_m * np.cos(new_angle)
                next_y = current_point.y + path_point_spacing_m * np.sin(new_angle)
                next_point = Point(next_x, next_y)
                next_radius = next_point.distance(Point(center_x, center_y))
                if next_radius < min_next_radius or next_radius > max_radius:
                    continue

                # Safety distance checks after initial steps
                if step >= SAFETY_DISTANCE_DISABLE_STEPS:
                    too_close = False
                    for path in paths:
                        if next_point.distance(path) < safety_distance:
                            too_close = True
                            break
                    if too_close:
                        continue

                temp_points = points + [(next_x, next_y)]
                candidate_line = LineString(temp_points)
                crosses_existing = False
                for existing_path in paths:
                    if candidate_line.crosses(existing_path):
                        crosses_existing = True
                        break
                if crosses_existing:
                    continue

                points.append((next_x, next_y))
                current_point = next_point
                prev_angle = new_angle
                found_valid = True
                break

            if not found_valid:
                direction = np.arctan2(current_point.y - center_y, current_point.x - center_x)
                border_x = center_x + max_radius * np.cos(direction)
                border_y = center_y + max_radius * np.sin(direction)
                border_point = Point(border_x, border_y)
                if border_point.distance(current_point) > 1e-3:
                    points.append((border_x, border_y))
                break

        if len(points) > 1 and Point(points[-1]).distance(Point(center_x, center_y)) >= max_radius - 1e-6:
            candidate = LineString(points)
            paths.append(candidate)

        attempts += 1

    return paths

def generate_random_walk_path(
    center_x: float,
    center_y: float,
    num_drones: int,
    probability_map: np.ndarray,
    bounds: tuple,
    num_jumps: int = 20,
    exploration_strength: float = 0.75,
    randomness: float = 2.0,
    memory_size: int = 10,
    **kwargs
) -> list[LineString]:
    """
    Generates random walk paths for multiple drones.

    Each drone performs a random walk where it moves in a direction for 10 steps,
    then changes direction. The walk continues until a specific distance is reached.

    Args:
        center_x, center_y: Starting coordinates for all drones
        num_drones: Number of drones to generate paths for
        probability_map: 2D array representing probability values (used for bounds reference)
        bounds: Tuple (minx, miny, maxx, maxy) defining the area bounds
        num_jumps: Number of direction changes before stopping (default: 20)
        exploration_strength: Controls how much drones prefer unexplored areas (0-1)
        randomness: Controls randomness of direction changes
        memory_size: Number of recent positions to remember for avoiding revisits
        **kwargs: Additional parameters (may include max_radius, path_point_spacing_m)

    Returns:
        List of LineString objects representing the path for each drone
    """
    # Extract additional parameters from kwargs
    max_radius = kwargs.get('max_radius', 100.0)  # Default radius if not provided
    path_point_spacing_m = kwargs.get('path_point_spacing_m', 5.0)  # Default step size

    # Constants
    STEPS_PER_DIRECTION = 10  # Number of steps before changing direction
    MIN_STEP_SIZE = path_point_spacing_m * 0.5
    MAX_STEP_SIZE = path_point_spacing_m * 1.5

    # Initialize random number generator
    rng = np.random.default_rng()

    paths = []

    for drone_idx in range(num_drones):
        # Initialize drone starting position with slight offset to avoid overlap
        angle_offset = (2 * np.pi * drone_idx) / num_drones if num_drones > 1 else 0
        start_offset = min(path_point_spacing_m, max_radius * 0.1)

        start_x = center_x + start_offset * np.cos(angle_offset)
        start_y = center_y + start_offset * np.sin(angle_offset)

        # Initialize path for this drone
        points = [(start_x, start_y)]
        current_x, current_y = start_x, start_y
        current_direction = rng.uniform(0, 2 * np.pi)  # Random initial direction

        # Memory of recent positions to avoid immediate revisits
        recent_positions = [(current_x, current_y)]

        # Perform random walk with direction changes
        for jump in range(num_jumps):
            # Change direction at the start of each jump (except first)
            if jump > 0:
                # Add some randomness to direction change
                direction_change = rng.normal(0, np.pi / randomness)
                current_direction += direction_change
                current_direction = current_direction % (2 * np.pi)  # Normalize to [0, 2π]

            # Walk in current direction for STEPS_PER_DIRECTION steps
            for _ in range(STEPS_PER_DIRECTION):
                # Add some randomness to step size and slight direction variation
                step_size = rng.uniform(MIN_STEP_SIZE, MAX_STEP_SIZE)
                direction_noise = rng.normal(0, np.pi / (randomness * 4))
                actual_direction = current_direction + direction_noise

                # Calculate next position
                next_x = current_x + step_size * np.cos(actual_direction)
                next_y = current_y + step_size * np.sin(actual_direction)

                # Check if next position is within radius
                distance_from_center = np.sqrt((next_x - center_x)**2 + (next_y - center_y)**2)

                if distance_from_center > max_radius:
                    # If we would go outside radius, reflect direction back toward center
                    to_center_direction = np.arctan2(center_y - current_y, center_x - current_x)
                    # Bias direction toward center with some randomness
                    current_direction = to_center_direction + rng.normal(0, np.pi / 4)

                    # Recalculate with new direction
                    next_x = current_x + step_size * np.cos(current_direction)
                    next_y = current_y + step_size * np.sin(current_direction)
                    distance_from_center = np.sqrt((next_x - center_x)**2 + (next_y - center_y)**2)

                    # If still outside, take a smaller step toward center
                    if distance_from_center > max_radius:
                        step_size *= 0.5
                        next_x = current_x + step_size * np.cos(current_direction)
                        next_y = current_y + step_size * np.sin(current_direction)

                # Avoid recent positions if memory is enabled
                if memory_size > 0 and len(recent_positions) > 0:
                    min_distance_to_recent = min(
                        np.sqrt((next_x - px)**2 + (next_y - py)**2)
                        for px, py in recent_positions[-memory_size:]
                    )

                    # If too close to recent position, add some randomness to avoid getting stuck
                    if min_distance_to_recent < path_point_spacing_m * 0.5:
                        avoidance_angle = rng.uniform(0, 2 * np.pi)
                        next_x += path_point_spacing_m * 0.3 * np.cos(avoidance_angle)
                        next_y += path_point_spacing_m * 0.3 * np.sin(avoidance_angle)

                # Add point to path
                points.append((next_x, next_y))
                current_x, current_y = next_x, next_y

                # Update recent positions memory
                recent_positions.append((current_x, current_y))
                if len(recent_positions) > memory_size:
                    recent_positions.pop(0)

                # Check if we've reached the target distance (edge of radius)
                if distance_from_center >= max_radius * 0.95:  # 95% of max radius
                    break

            # Break outer loop if we've reached the edge
            distance_from_center = np.sqrt((current_x - center_x)**2 + (current_y - center_y)**2)
            if distance_from_center >= max_radius * 0.95:
                break

        # Create LineString for this drone's path
        if len(points) > 1:
            paths.append(LineString(points))
        else:
            paths.append(LineString())

    return paths