"""
Creates a comparative coverage video showing 4 algorithms (Concentric, Spiral, Greedy, Random) 
in a 2x2 grid layout with time-series graphs on the right showing metrics evolution.
Optimized with caching of cumulative unions and metrics for speed improvements.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import cv2


import sarenv
from sarenv.analytics import metrics
from sarenv.utils import plot
from sarenv.analytics.evaluator import ComparativeEvaluator


log = sarenv.get_logger()


class ComparativeCoverageVideoGenerator:
    """Generates a comparative video showing 4 algorithms side by side with time-series graphs."""
    
    def __init__(self, item, victims_gdf, path_evaluator, crs, output_dir, n_frames=200):
        self.item = item
        self.victims_gdf = victims_gdf
        self.path_evaluator = path_evaluator
        self.crs = crs
        self.output_dir = output_dir
        
        # Video settings
        self.fps = 10  # Frames per second
        self.dpi = 100
        self.figsize = (20, 12)  # Large figure for 2x2 + graphs
        
        # Animation frames fixed maximum (used for visualization pacing)
        self.n_frames = n_frames
        
        # Define colors for multiple drones
        self.drone_colors = ['blue', 'green', 'gray']
        
        # Algorithm names and colors for graphs
        self.algorithm_colors = {
            'Concentric': 'blue',
            'Pizza': 'red', 
            'Greedy': 'green',
            'RandomWalk': 'orange'
        }
        
    def create_comparative_video(self, algorithms_data):
        """Create a comparative video showing all algorithms side by side."""
        log.info("Creating comparative coverage video...")
        
        # Prepare animation data for all algorithms
        all_animation_data = {}
        
        for alg_name, paths in algorithms_data.items():
            if alg_name in self.algorithm_colors:  # Only process target algorithms
                log.info(f"Preparing animation data for {alg_name}...")
                animation_data = self._prepare_animation_data_fixed_frames(paths, alg_name)
                all_animation_data[alg_name] = animation_data
                log.info(f"Prepared animation data for {alg_name} with {len(animation_data['positions'])} frames.")
        
        if not all_animation_data:
            log.warning("No valid animation data found")
            return
            
        output_file = self.output_dir / "comparative_coverage_video.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.figsize[0] * self.dpi)
        frame_height = int(self.figsize[1] * self.dpi)
        video_writer = cv2.VideoWriter(str(output_file), fourcc, self.fps, (frame_width, frame_height))
        log.info(f"Video writer initialized: {output_file}")
        
        try:
            # Determine max frames from the prepared animation data to control video length
            max_frames = max(len(data['positions']) for data in all_animation_data.values())
            # Limit to self.n_frames or max_frames whichever is smaller
            total_frames = min(self.n_frames, max_frames)

            # Generate each frame
            for frame_idx in range(total_frames):
                if frame_idx % 20 == 0:
                    log.info(f"Generating frame {frame_idx + 1}/{total_frames}...")
                fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
                gs = fig.add_gridspec(6, 6, width_ratios=[1,1,1,1,0.8,0.8], hspace=0.25, wspace=0.25)
                
                ax_positions = {
                    'Concentric': (0, 0, 3, 2),
                    'Pizza': (0, 2, 3, 2),
                    'Greedy': (3, 0, 3, 2),
                    'RandomWalk': (3, 2, 3, 2)
                }
                algorithm_axes = {}
                for alg_name, (row, col, rowspan, colspan) in ax_positions.items():
                    algorithm_axes[alg_name] = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
                
                ax_area = fig.add_subplot(gs[0:2, 4:])
                ax_score = fig.add_subplot(gs[2:4, 4:])
                ax_victims = fig.add_subplot(gs[4:6, 4:])
                
                self._create_comparative_frame(frame_idx, all_animation_data, algorithm_axes, ax_area, ax_score, ax_victims)
                
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                buf = np.asarray(buf)
                buf = buf[:, :, :3] 
                frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                
                video_writer.write(frame)
                plt.close(fig)
                
        finally:
            video_writer.release()
            log.info("Video writer released.")
        
        log.info(f"Saved comparative coverage video: {output_file}")
        
    def _prepare_animation_data_fixed_frames(self, paths, algorithm_name):
        """Prepare animation data using original path points (no interpolation) with jumps to fit exactly n_frames."""
        log.info(f"Starting animation data for {algorithm_name} with {len(paths)} paths.")
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths:
            log.warning(f"No valid paths found for {algorithm_name}")
            return {'positions': [], 'drone_positions': [], 'metrics': [], 'num_drones': 0}

        num_drones = len(valid_paths)

        # Extract original path points
        drone_path_points = []
        max_path_len = 0
        for drone_idx, path in enumerate(valid_paths):
            pts = [(pt[0], pt) for pt in path.coords]
            drone_path_points.append(pts)
            if len(pts) > max_path_len:
                max_path_len = len(pts)

        total_frames = self.n_frames

        # For each drone, create an index list sampling its points to length total_frames (or repeating last)
        drone_sample_indices = []
        for pts in drone_path_points:
            L = len(pts)
            if L <= 1:
                # Path too short, use zeros repeated
                indices = [0] * total_frames

            else:
                # Generate indices: linearly spaced over [0, L-1] with total_frames steps, rounded to int
                indices = np.linspace(0, L-1, total_frames)
                indices = np.round(indices).astype(int).tolist()
            drone_sample_indices.append(indices)

        # Precompute cumulative unions per drone (cache) up to all points (to reuse)
        cumulative_paths_cache = []
        for drone_idx, pts in enumerate(drone_path_points):
            cumulative_per_segment = []
            cum_line = LineString()
            for i in range(len(pts)):
                if i == 0:
                    line = LineString([pts[0], pts])
                else:
                    line = LineString(pts[:i+1])
                cum_line = unary_union([cum_line, line])
                cumulative_per_segment.append(cum_line)
            cumulative_paths_cache.append(cumulative_per_segment)

        all_positions = []
        all_drone_positions = []
        all_metrics = []

        for frame_idx in range(total_frames):
            if frame_idx % 20 == 0:
                log.info(f"Caching metrics for frame {frame_idx + 1}/{total_frames}...")
            current_positions = []
            partial_paths = []
            for d_idx in range(num_drones):
                indices = drone_sample_indices[d_idx]
                # Get stepped index for current frame, clamp to max available index
                idx = min(indices[frame_idx], len(drone_path_points[d_idx]) - 1)
                current_positions.append(drone_path_points[d_idx][idx])
                # cumulative paths up to this idx
                cumulative_idx = min(idx, len(cumulative_paths_cache[d_idx]) - 1)
                partial_paths.append(cumulative_paths_cache[d_idx][cumulative_idx])

            all_positions.append(current_positions[0] if current_positions else (0, 0))
            all_drone_positions.append(current_positions)

            current_metrics = self.path_evaluator.calculate_all_metrics(partial_paths, 0.999)
            area_covered = current_metrics['area_covered']
            likelihood_score = current_metrics['total_likelihood_score']
            victims_found_pct = current_metrics['victim_detection_metrics']['percentage_found']

            all_metrics.append({
                'area_covered': area_covered,
                'likelihood_score': likelihood_score,
                'victims_found_pct': victims_found_pct,
            })

        log.info(f"Finished preparing animation data for {algorithm_name} with {total_frames} frames.")
        return {
            'positions': all_positions,
            'drone_positions': all_drone_positions,
            'metrics': all_metrics,
            'num_drones': num_drones
        }



    def _setup_algorithm_plot(self, ax, algorithm_name):
        x_min, y_min, x_max, y_max = self.item.bounds
        extent = [x_min, x_max, y_min, y_max]


        ax.imshow(
            self.item.heatmap, 
            extent=extent, 
            origin='lower',
            cmap='YlOrRd',
            alpha=0.8,
            aspect='equal'
        )


        if not self.victims_gdf.empty:
            victims_proj = self.victims_gdf.to_crs(self.crs)
            ax.scatter(
                victims_proj.geometry.x, 
                victims_proj.geometry.y,
                c='black', s=30, marker='X', linewidths=1,
                zorder=10, edgecolors='darkred'
            )


        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'{algorithm_name}', fontsize=12)
        ax.grid(True, alpha=0.3)


        ax.set_xticks([])
        ax.set_yticks([])


    def _create_comparative_frame(self, frame_idx, all_animation_data, algorithm_axes, 
                                  ax_area, ax_score, ax_victims):
        for alg_name, ax in algorithm_axes.items():
            self._setup_algorithm_plot(ax, alg_name)
            if alg_name in all_animation_data:
                animation_data = all_animation_data[alg_name]
                if frame_idx < len(animation_data['drone_positions']):
                    current_drone_positions = animation_data['drone_positions'][frame_idx]
                    num_drones = animation_data.get('num_drones', 1)
                    
                    # For each drone, plot entire path up to current frame, showing all original points traveled so far
                    for drone_idx in range(num_drones):
                        # Collect all the original points traveled by this drone up to current frame
                        drone_path_x = []
                        drone_path_y = []
                        
                        # Use the stored drone positions up to current frame (no interpolation)
                        for past_frame in range(frame_idx + 1):
                            if drone_idx < len(animation_data['drone_positions'][past_frame]):
                                pos = animation_data['drone_positions'][past_frame][drone_idx]
                                drone_path_x.append(pos[0])
                                drone_path_y.append(pos[1])  # Fix: pos is 2D tuple, use index 1 for y
                        
                        if len(drone_path_x) > 1:
                            drone_color = self.drone_colors[drone_idx % len(self.drone_colors)]
                            ax.plot(drone_path_x, drone_path_y, color=drone_color, linewidth=2, alpha=0.8)
                    
                    # Plot current drone positions
                    for drone_idx, drone_position in enumerate(current_drone_positions):
                        drone_color = self.drone_colors[drone_idx % len(self.drone_colors)]
                        detection_circle = plt.Circle(
                            drone_position, self.path_evaluator.detection_radius, 
                            fill=False, color=drone_color, alpha=0.3, linewidth=1, linestyle='--'
                        )
                        ax.add_patch(detection_circle)
                        if drone_position is not None and len(drone_position) == 2:
                            ax.scatter(
                                drone_position[0], drone_position[1], 
                                c=drone_color, s=100, marker='o', 
                                edgecolors='white', linewidths=2, zorder=15
                            )


        self._create_time_series_graphs(frame_idx, all_animation_data, ax_area, ax_score, ax_victims)
    
    def _create_time_series_graphs(self, frame_idx, all_animation_data, ax_area, ax_score, ax_victims):
        for ax in [ax_area, ax_score, ax_victims]:
            ax.clear()
        for alg_name, animation_data in all_animation_data.items():
            if frame_idx < len(animation_data['metrics']):
                color = self.algorithm_colors[alg_name]
                frames_so_far = min(frame_idx + 1, len(animation_data['metrics']))
                time_steps = list(range(frames_so_far))
                scores = [animation_data['metrics'][i]['likelihood_score'] for i in range(frames_so_far)]
                victims = [animation_data['metrics'][i]['victims_found_pct'] for i in range(frames_so_far)]
                areas = [animation_data['metrics'][i]['area_covered'] for i in range(frames_so_far)]
                ax_area.plot(time_steps, areas, color=color, linewidth=2, label=alg_name)
                ax_score.plot(time_steps, scores, color=color, linewidth=2, label=alg_name)
                ax_victims.plot(time_steps, victims, color=color, linewidth=2, label=alg_name)
        max_frames = self.n_frames
        for ax, title, ylabel in [
            (ax_area, 'Area Covered', 'Area (km²)'),
            (ax_score, 'Likelihood Score', 'Score'),
            (ax_victims, 'Victims Found', 'Percentage (%)')
        ]:
            ax.set_xlim(0, max_frames)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            if ax != ax_victims:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time Steps', fontsize=9)



if __name__ == "__main__":
    log.info("--- Initializing Comparative Coverage Video Generation ---")
    data_dir = "sarenv_dataset/19"
    
    evaluator = ComparativeEvaluator(
        dataset_directory=data_dir,
        evaluation_sizes=["medium"],
        num_drones=3,
        num_lost_persons=100,
        budget=100000,
    )
    
    output_dir = Path("coverage_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("--- Generating Comparative Coverage Video ---")
    
    evaluator.load_datasets()
    size = "medium"
    env_data = evaluator.environments[size]
    item = env_data["item"]
    victims_gdf = env_data["victims"]
    
    path_evaluator = metrics.PathEvaluator(
        item.heatmap,
        item.bounds,
        victims_gdf,
        evaluator.path_generator_config.fov_degrees,
        evaluator.path_generator_config.altitude_meters,
        evaluator.loader._meter_per_bin,
    )
    
    center_proj = (
        gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
        .to_crs(env_data["crs"])
        .geometry.iloc[0]
    )
    
    target_algorithms = ['Concentric', 'Pizza', 'Greedy', 'RandomWalk']
    algorithms_data = {}
    
    for name, generator in evaluator.path_generators.items():
        if name in target_algorithms:
            log.info(f"Generating paths for {name}...")
            generated_paths = generator(
                center_proj.x,
                center_proj.y,
                item.radius_km * 1000,
                item.heatmap,
                item.bounds,
            )
            algorithms_data[name] = generated_paths
            log.info(f"Generated {len(generated_paths)} paths for {name}")
    
    video_generator = ComparativeCoverageVideoGenerator(
        item, victims_gdf, path_evaluator, env_data["crs"], output_dir, n_frames=100
    )
    
    video_generator.create_comparative_video(algorithms_data)
    
    log.info("--- Comparative Coverage Video Generation Complete ---")
    log.info(f"Video saved in: {output_dir.absolute()}")
