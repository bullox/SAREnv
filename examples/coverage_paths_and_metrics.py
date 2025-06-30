# examples/coverage_paths_and_metrics.py
import os

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from sarenv import (
    DatasetLoader,
    SARDatasetItem,
    SurvivorLocationGenerator,
    get_logger,
)
from sarenv.utils.plot_utils import DEFAULT_COLOR, FEATURE_COLOR_MAP
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

log = get_logger()

# --- Parameters ---
GRAPHS_DIR = "graphs"
FOV_DEGREES = 45.0
ALTITUDE_METERS = 80.0
OVERLAP_RATIO = 0.25
NUM_DRONES = 5
NUM_VICTIMS = 5
EVALUATION_SAMPLE_POINTS = 1000  # Control sampling density for evaluation
PLOT_MAP_FEATURES = False  # Set to True to render detailed map features, False for lightweight plots
PATH_POINT_SPACING_M = 10.0  # Distance between points on generated paths for higher resolution
TRANSITION_DISTANCE_M = 50.0 # Fixed distance for the transition segment between concentric circles
PIZZA_BORDER_GAP_M = 15.0   # New: Fixed distance gap from the slice borders for the pizza pattern

# --- Helper, Path Generation, and Evaluation functions ---
def get_utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{326 if lat >= 0 else 327}{zone}"

def split_path_for_drones(path: LineString, num_drones: int) -> list[LineString]:
    if num_drones <= 1 or path.is_empty or path.length == 0:
        return [path]
    segments = []
    segment_length = path.length / num_drones
    for i in range(num_drones):
        segments.append(substring(path, i * segment_length, (i + 1) * segment_length))
    return segments

def generate_spiral_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float) -> list[LineString]:
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

def generate_concentric_circles_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, transition_distance_m: float) -> list[LineString]:
    radius_increment = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    path_points = []
    current_radius = radius_increment

    while current_radius <= max_radius:
        # Calculate transition angle dynamically to maintain a fixed transition distance
        transition_angle_rad = transition_distance_m / current_radius if current_radius > 0 else np.radians(45)

        arc_length = current_radius * (2 * np.pi - transition_angle_rad)
        num_points_circle = max(2, int(arc_length / path_point_spacing_m))

        theta = np.linspace(0, 2 * np.pi - transition_angle_rad, num_points_circle)
        path_points.extend(zip(center_x + current_radius * np.cos(theta), center_y + current_radius * np.sin(theta)))

        next_radius = current_radius + radius_increment
        if next_radius <= max_radius:
            path_points.append((center_x + next_radius, center_y))
        else: # Complete the final circle
            final_theta_points = max(2, int((current_radius * transition_angle_rad) / path_point_spacing_m))
            final_theta = np.linspace(2 * np.pi - transition_angle_rad, 2 * np.pi, final_theta_points)
            path_points.extend(zip(center_x + current_radius * np.cos(final_theta), center_y + current_radius * np.sin(final_theta)))

        current_radius = next_radius

    full_path = LineString(path_points) if path_points else LineString()
    return split_path_for_drones(full_path, num_drones)

def generate_pizza_zigzag_path(center_x: float, center_y: float, max_radius: float, num_drones: int, fov_deg: float, altitude: float, overlap: float, path_point_spacing_m: float, border_gap_m: float) -> list[LineString]:
    paths, section_angle_rad = [], 2 * np.pi / num_drones
    pass_width = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)

    for i in range(num_drones):
        base_start_angle, base_end_angle = i * section_angle_rad, (i + 1) * section_angle_rad
        points, radius, direction = [(center_x, center_y)], pass_width, 1

        while radius <= max_radius:
            # Dynamically calculate angular gap to maintain a fixed distance border
            angular_offset_rad = border_gap_m / radius if radius > 0 else 0
            start_angle = base_start_angle + angular_offset_rad
            end_angle = base_end_angle - angular_offset_rad

            # If the gap is too large for the radius, skip this pass
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
            paths.append(LineString(points))
    return paths

class PathEvaluator:
    def __init__(self, heatmap: np.ndarray, extent: tuple, victims: gpd.GeoDataFrame, fov_deg: float, altitude: float):
        self.heatmap = heatmap; self.extent = extent; self.victims = victims
        self.detection_radius = altitude * np.tan(np.radians(fov_deg / 2))
        minx, miny, maxx, maxy = self.extent
        y_range = np.linspace(miny, maxy, heatmap.shape[0])
        x_range = np.linspace(minx, maxx, heatmap.shape[1])
        self.interpolator = RegularGridInterpolator((y_range, x_range), heatmap, bounds_error=False, fill_value=0)

    def calculate_likelihood_score(self, paths: list[LineString]) -> float:
        total_likelihood = 0
        for path in paths:
            if not path.is_empty and path.length > 0:
                points = [path.interpolate(d) for d in np.linspace(0, path.length, EVALUATION_SAMPLE_POINTS)]
                point_coords = [(p.y, p.x) for p in points]
                total_likelihood += np.sum(self.interpolator(point_coords))
        return total_likelihood

    def calculate_time_discounted_score(self, paths: list[LineString], discount_factor: float = 0.999) -> float:
        total_score = 0
        for path in paths:
            if not path.is_empty and path.length > 0:
                distances = np.linspace(0, path.length, EVALUATION_SAMPLE_POINTS)
                points = [path.interpolate(d) for d in distances]
                point_coords = [(p.y, p.x) for p in points]
                likelihoods = self.interpolator(point_coords)
                discounts = discount_factor ** distances
                total_score += np.sum(likelihoods * discounts)
        return total_score

    def calculate_victims_found_score(self, paths: list[LineString]) -> dict:
        valid_paths = [p for p in paths if not p.is_empty]
        if not valid_paths or self.victims.empty: return {'percentage_found': 0, 'detection_timeliness': 0}
        coverage_area = gpd.GeoSeries(valid_paths, crs=self.victims.crs).buffer(self.detection_radius).union_all()
        found_victims = self.victims[self.victims.within(coverage_area)]
        percentage_found = (len(found_victims) / len(self.victims)) * 100 if not self.victims.empty else 0
        timeliness = []
        for _, victim in found_victims.iterrows():
            min_dist = min(path.project(victim.geometry) for path in valid_paths)
            avg_path_length = np.mean([p.length for p in valid_paths])
            if avg_path_length > 0: timeliness.append(min_dist / avg_path_length)
        return {'percentage_found': percentage_found, 'detection_timeliness': np.mean(timeliness) if timeliness else 0}

# --- Visualization functions ---
def visualize_heatmap_plotly(item: SARDatasetItem, data_crs: str):
    log.info(f"Generating interactive heatmap visualization for size: {item.size}...")
    minx, miny, maxx, maxy = item.bounds
    fig = go.Figure(data=go.Heatmap(z=item.heatmap, x=np.linspace(minx, maxx, item.heatmap.shape[1]), y=np.linspace(miny, maxy, item.heatmap.shape[0]), colorscale='Inferno', colorbar=dict(title='Probability Density')))
    fig.update_layout(title=f"Interactive Heatmap: Size '{item.size}'", xaxis_title="Easting (meters)", yaxis_title="Northing (meters)", yaxis_scaleanchor="x", template="plotly_white")
    output_path = os.path.join(GRAPHS_DIR, "heatmap.html")
    fig.write_html(output_path, include_plotlyjs="cdn")
    log.info(f"Heatmap saved to {output_path}")

def visualize_single_path_plotly(item: SARDatasetItem, path_name: str, paths: list, colors: list, victims: gpd.GeoDataFrame, data_crs: str):
    log.info(f"Generating visualization for '{path_name}' path...")
    features_proj = item.features.to_crs(crs=data_crs)
    victims_proj = victims.to_crs(crs=data_crs) if not victims.empty else victims
    fig = go.Figure()

    if PLOT_MAP_FEATURES:
        for feature_type, data in features_proj.groupby('feature_type'):
            color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
            for i, geom in enumerate(data.geometry):
                if geom.is_empty: continue
                show_legend = (i == 0)
                geometries = geom.geoms if geom.geom_type.startswith('Multi') else [geom]
                for j, sub_geom in enumerate(geometries):
                    if sub_geom.geom_type == 'Polygon':
                        x, y = sub_geom.exterior.xy
                        fig.add_trace(go.Scatter(x=list(x), y=list(y), fill='toself', mode='lines', line=dict(color=color), opacity=0.6, name=feature_type.capitalize(), legendgroup=feature_type, showlegend=(show_legend and j == 0)))
                    elif sub_geom.geom_type == 'LineString':
                        x, y = sub_geom.xy
                        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color=color, width=2), name=feature_type.capitalize(), legendgroup=feature_type, showlegend=(show_legend and j == 0)))

    for i, path in enumerate(paths):
        if not path.is_empty:
            x, y = path.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color=colors[i % len(colors)], width=3), name=f'{path_name} Path', legendgroup=path_name, showlegend=(i == 0)))

    if not victims_proj.empty:
        fig.add_trace(go.Scatter(x=victims_proj.geometry.x, y=victims_proj.geometry.y, mode='markers', marker=dict(symbol='x', color='red', size=12), name='Victim Location'))

    clean_name = path_name.lower().replace(' ', '_')
    fig.update_layout(title=f"Coverage Path for '{path_name}' Pattern", xaxis_title="Easting (meters)", yaxis_title="Northing (meters)", legend_title_text="Legend", yaxis_scaleanchor="x", template="plotly_white")
    output_path = os.path.join(GRAPHS_DIR, f"coverage_{clean_name}.html")
    fig.write_html(output_path, include_plotlyjs="cdn")
    log.info(f"Path visualization saved to {output_path}")

# --- Main Execution Block ---
def run_evaluation_and_visualization():
    log.info("--- Starting Multi-Drone Path Evaluation and Visualization ---")
    dataset_dir = "sarenv_dataset"
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    try:
        loader = DatasetLoader(dataset_directory=dataset_dir)
        # Use the largest available size for the evaluation scenario
        evaluation_size = "xlarge"

        log.info(f"Loading data for evaluation size: '{evaluation_size}'")
        item = loader.load_size(evaluation_size)
        if not item:
            log.error(f"Could not load data for size '{evaluation_size}'. Please run the DataGenerator first.")
            return

        data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
        extent = item.bounds

        # Use the SurvivorLocationGenerator to place victims realistically
        log.info(f"Generating {NUM_VICTIMS} victim locations...")
        victim_generator = SurvivorLocationGenerator(item)
        victim_points = [victim_generator.generate_location() for _ in range(NUM_VICTIMS)]
        victim_points = [p for p in victim_points if p is not None] # Filter out None values

        if not victim_points:
            log.error("Failed to generate any victim locations.")
            victims = gpd.GeoDataFrame(columns=['geometry'], crs=data_crs)
        else:
            victims = gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)

        evaluator = PathEvaluator(item.heatmap, extent, victims, FOV_DEGREES, ALTITUDE_METERS)

        # Get center point from the loaded item for path generation
        center_lon, center_lat = item.center_point
        center_point_gdf = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs="EPSG:4326")
        center_proj = center_point_gdf.to_crs(data_crs).geometry.iloc[0]
        center_x, center_y = center_proj.x, center_proj.y
        max_radius_m = item.radius_km * 1000

        path_generators = {
            "Spiral": lambda: generate_spiral_path(center_x, center_y, max_radius_m, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M),
            "Concentric Circles": lambda: generate_concentric_circles_path(center_x, center_y, max_radius_m, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, NUM_DRONES, PATH_POINT_SPACING_M, TRANSITION_DISTANCE_M),
            "Pizza Zigzag": lambda: generate_pizza_zigzag_path(center_x, center_y, max_radius_m, NUM_DRONES, FOV_DEGREES, ALTITUDE_METERS, OVERLAP_RATIO, PATH_POINT_SPACING_M, PIZZA_BORDER_GAP_M)
        }

        colors = {
            "Spiral": ['#00FFFF', '#00E5EE', '#00C5CD', '#00868B', '#00688B', '#00468B', '#003366', '#002147', '#001A4D'],
            "Concentric Circles": ['#FF00FF', '#EE00EE', '#CD00CD', '#8B008B', '#6A006A', '#4B004B', '#330033', '#1A001A', '#0D000D'],
            "Pizza Zigzag": ['#33FF57', '#2E8B57', '#3CB371', '#66CDAA', '#20B2AA', '#008B8B', '#006666', '#004D4D', '#003333']
        }

        for name, generator in path_generators.items():
            log.info(f"Evaluating '{name}' path...")
            paths = generator()

            likelihood_score = evaluator.calculate_likelihood_score(paths)
            discounted_score = evaluator.calculate_time_discounted_score(paths)
            victim_score = evaluator.calculate_victims_found_score(paths)

            print(f"\n--- Metrics for {name} Path ---")
            print(f"  Likelihood of Detection Score: {likelihood_score:.2f}")
            print(f"  Time-Discounted Information Score: {discounted_score:.2f}")
            print(f"  Victims Found: {victim_score['percentage_found']:.2f}%")
            print(f"  Victim Detection Timeliness (0=early, 1=late): {victim_score['detection_timeliness']:.3f}")

            visualize_single_path_plotly(item, name, paths, colors[name], victims, data_crs)

        visualize_heatmap_plotly(item, data_crs)

    except FileNotFoundError:
        log.error(f"Dataset directory '{dataset_dir}' not found. Please run the DataGenerator first.")
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    run_evaluation_and_visualization()
