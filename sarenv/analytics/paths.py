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
    num_jumps: int,
    exploration_strength: float = 0.75,
    randomness: float = 2.0,
    memory_size: int = 10,
    **kwargs
) -> list[LineString]:
    """
    Generates exploratory paths on the heatmap grid, balancing a push
    outwards with randomness to prevent straight-line movement.

    Args:
        center_x (float): The starting X coordinate.
        center_y (float): The starting Y coordinate.
        num_drones (int): The number of drones.
        num_jumps (int): The number of cell-to-cell steps for each drone's walk.
        bounds (tuple): Geographical boundaries (minx, miny, maxx, maxy).
        probability_map (np.ndarray): The heatmap defining the grid for movement.
        exploration_strength (float): The strength of the bias pushing drones
                                      outwards.
        randomness (float): Controls the 'temperature' of the random choice.
                            Higher values increase randomness.
        memory_size (int): The number of recent steps for a drone to remember
                           and avoid revisiting.
        **kwargs: Catches unused arguments for compatibility.

    Returns:
        list[LineString]: A list of paths for each drone.
    """
    if probability_map is None:
        raise ValueError("A valid probability_map must be provided to define the grid.")
    
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds

    if maxx <= minx or maxy <= miny or width <= 1 or height <= 1:
        return [LineString() for _ in range(num_drones)]

    # Create maps to convert grid indices back to real-world coordinates
    x_map = np.linspace(minx + (maxx - minx) / (2 * width), maxx - (maxx - minx) / (2 * width), width)
    y_map = np.linspace(miny + (maxy - miny) / (2 * height), maxy - (maxy - miny) / (2 * height), height)

    # Convert the starting real-world coordinates to grid indices
    start_col = int(((center_x - minx) / (maxx - minx)) * width)
    start_row = int(((center_y - miny) / (maxy - miny)) * height)
    start_pos = (np.clip(start_row, 0, height - 1), np.clip(start_col, 0, width - 1))

    # --- Grid-based simulation starts here ---
    visited = np.zeros((height, width), dtype=bool)
    current_positions = [start_pos] * num_drones
    paths = [[pos] for pos in current_positions]
    path_memory = [[start_pos] for _ in range(num_drones)]
    visited[start_pos] = True

    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    grid_center = (height / 2, width / 2)

    for _ in range(num_jumps):
        proposed_moves = {}
        for i in range(num_drones):
            r, c = current_positions[i]
            
            valid_neighbors = []
            scores = []
            current_dist_from_center = np.sqrt((r - grid_center[0])**2 + (c - grid_center[1])**2)

            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                neighbor_pos = (nr, nc)
                
                # Check if the move is valid (within grid, not visited, not in recent memory)
                if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and neighbor_pos not in path_memory[i]:
                    valid_neighbors.append(neighbor_pos)
                    neighbor_dist_from_center = np.sqrt((nr - grid_center[0])**2 + (nc - grid_center[1])**2)
                    distance_gain = neighbor_dist_from_center - current_dist_from_center
                    
                    # Calculate a score for the move
                    score = np.exp(exploration_strength * distance_gain / (randomness + 1e-6))
                    scores.append(score)
            
            if valid_neighbors:
                # Choose the next move based on a weighted random choice
                total_score = sum(scores)
                probabilities = [s / total_score for s in scores] if total_score > 0 else None
                chosen_neighbor = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
                proposed_moves[i] = chosen_neighbor

        # Resolve conflicts where multiple drones want the same cell
        claimed_cells = {}
        for drone_idx, pos in proposed_moves.items():
            if pos not in claimed_cells:
                claimed_cells[pos] = []
            claimed_cells[pos].append(drone_idx)

        final_moves = {}
        for pos, claimants in claimed_cells.items():
            winner = random.choice(claimants)
            final_moves[winner] = pos
        
        # Update drone positions, memory, and the master visited grid
        for drone_idx, new_pos in final_moves.items():
            current_positions[drone_idx] = new_pos
            paths[drone_idx].append(new_pos)
            visited[new_pos] = True
            
            path_memory[drone_idx].append(new_pos)
            if len(path_memory[drone_idx]) > memory_size:
                path_memory[drone_idx].pop(0)

    # Convert the grid-based paths to real-world coordinate paths
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            world_coords = [(x_map[c], y_map[r]) for r, c in drone_path_indices]
            line_paths.append(LineString(world_coords))
        else:
            line_paths.append(LineString())

    return line_paths