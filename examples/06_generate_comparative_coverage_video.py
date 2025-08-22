# examples/06_generate_comparative_coverage_video.py
"""
Simplified Comparative Coverage Video Generator

Creates a comparative coverage video showing 4 algorithms (Concentric, Pizza, Greedy, RandomWalk) 
in a 2x2 grid layout with time-series graphs showing metrics evolution.

Features:
- Efficient precomputed metrics approach
- Clean and readable code structure
- Multiple drone support with distinct colors
- Real-time metrics visualization

Usage:
    python 06_generate_comparative_coverage_video.py
"""

import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import cv2

import sarenv
from sarenv.analytics import metrics
from sarenv.analytics.evaluator import ComparativeEvaluator
from sarenv.utils.plot import setup_algorithm_plot, plot_drone_paths, plot_current_drone_positions, create_time_series_graphs

log = sarenv.get_logger()


class ComparativeCoverageVideoGenerator:
    """Generates a comparative video showing 4 algorithms side by side with time-series graphs."""
    
    def __init__(self, item, victims_gdf, path_evaluator, crs, output_dir, interval_distance=2500.0):
        self.item = item
        self.victims_gdf = victims_gdf
        self.path_evaluator = path_evaluator
        self.crs = crs
        self.output_dir = Path(output_dir)
        self.interval_distance = interval_distance  # Distance in meters between calculations
        
        # Video settings
        self.fps = 2  # Frames per second
        self.dpi = 100
        self.figsize = (20, 12)  # Large figure for 2x2 + graphs
        self.n_frames = None  # Will be determined by the path lengths and interval distance
        
        # Define colors for multiple drones
        self.drone_colors = ['blue', 'green', 'gray', 'purple', 'orange']
        
        # Algorithm names and colors for graphs
        self.algorithm_colors = {
            'RandomWalk': 'orange',
            'Greedy': 'green', 
            'Concentric': 'blue',
            'Pizza': 'red'
        }
    
    def create_comparative_video(self, algorithms_data):
        """Create a comparative video showing all algorithms side by side."""
        log.info("Creating comparative coverage video...")
        
        # Prepare animation data for all algorithms using efficient method
        all_animation_data = {}
        
        for alg_name, paths in algorithms_data.items():
            if alg_name in self.algorithm_colors:  # Only process target algorithms
                log.info(f"Preparing animation data for {alg_name}...")
                animation_data = self._prepare_animation_data(paths, alg_name)
                    
                if animation_data['num_drones'] > 0:  # Only add if we have valid data
                    all_animation_data[alg_name] = animation_data
                    log.info(f"Prepared animation data for {alg_name} with {len(animation_data['positions'])} frames.")
                else:
                    log.warning(f"Skipping {alg_name} - no valid animation data")
        
        if not all_animation_data:
            log.error("No valid animation data found for any algorithm")
            return
            
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "comparative_coverage_video.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.figsize[0] * self.dpi)
        frame_height = int(self.figsize[1] * self.dpi)
        
        video_writer = cv2.VideoWriter(str(output_file), fourcc, self.fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            log.error(f"Failed to open video writer for {output_file}")
            return
            
        log.info(f"Video writer initialized: {output_file}")
        
        try:
            # Determine total frames
            max_frames = max(len(data['positions']) for data in all_animation_data.values())
            total_frames = min(self.n_frames, max_frames)

            # Generate each frame
            for frame_idx in range(total_frames):
                if frame_idx % 10 == 0:  # More frequent progress updates
                    log.info(f"Generating frame {frame_idx + 1}/{total_frames}... ({100*frame_idx/total_frames:.1f}%)")
                    
                try:
                    frame = self._create_video_frame(frame_idx, all_animation_data)
                    if frame is not None:
                        video_writer.write(frame)
                    else:
                        log.warning(f"Failed to create frame {frame_idx}")
                except Exception as e:
                    log.error(f"Error creating frame {frame_idx}: {e}")
                    continue
                    
        except Exception as e:
            log.error(f"Error during video generation: {e}")
        finally:
            video_writer.release()
            log.info("Video writer released.")
        
        if output_file.exists():
            log.info(f"Saved comparative coverage video: {output_file}")
        else:
            log.error("Failed to save video file")
    
    def _create_video_frame(self, frame_idx, all_animation_data):
        """Create a single video frame."""
        try:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            gs = fig.add_gridspec(6, 6, width_ratios=[1,1,1,1,0.8,0.8], hspace=0.25, wspace=0.25)
            
            # Algorithm subplot positions
            ax_positions = {
                'RandomWalk': (0, 0, 3, 2),
                'Greedy': (0, 2, 3, 2),
                'Concentric': (3, 0, 3, 2),
                'Pizza': (3, 2, 3, 2)
            }
            
            algorithm_axes = {}
            for alg_name, (row, col, rowspan, colspan) in ax_positions.items():
                algorithm_axes[alg_name] = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
            
            # Metrics subplots
            ax_area = fig.add_subplot(gs[0:2, 4:])
            ax_score = fig.add_subplot(gs[2:4, 4:])
            ax_victims = fig.add_subplot(gs[4:6, 4:])
            
            self._create_comparative_frame(frame_idx, all_animation_data, algorithm_axes, ax_area, ax_score, ax_victims)
            
            # Convert to video frame
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            buf = np.asarray(buf)
            buf = buf[:, :, :3]  # Remove alpha channel
            frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            return frame
            
        except Exception as e:
            log.error(f"Error creating video frame {frame_idx}: {e}")
            return None
        
    def _prepare_animation_data(self, paths, algorithm_name):
        """Prepare animation data using pre-computed metrics at intervals."""
        start_time = time.time()
        log.info(f"Preparing animation data for {algorithm_name} with {len(paths)} paths.")
        
        # Use the configured interval distance
        log.info(f"Pre-computing metrics every {self.interval_distance}m...")
        
        try:
            precomputed_data = self.path_evaluator.calculate_metrics_at_distance_intervals(
                paths, discount_factor=0.999, interval_distance=self.interval_distance
            )
            
            if not precomputed_data.get('interval_metrics'):
                log.warning(f"No precomputed metrics available for {algorithm_name}")
                return {'positions': [], 'drone_positions': [], 'path_coordinates': [], 'metrics': [], 'num_drones': 0}
                
            # Extract data from precomputed results
            total_intervals = precomputed_data['total_intervals']
            interval_positions = precomputed_data['interval_positions']
            interval_metrics = precomputed_data['interval_metrics']
            interval_path_coordinates = precomputed_data.get('interval_path_coordinates', [])
            
            self.n_frames = total_intervals
            log.info(f"Using {total_intervals} precomputed intervals for {self.n_frames} video frames")
            
            # Map intervals to video frames
            all_positions = []
            all_drone_positions = []
            all_metrics = []
            all_path_coordinates = []  # Store path coordinates for natural rendering
            all_interval_distances = []  # Store the actual interval distances (2.5km steps)
            
            for frame_idx in range(self.n_frames):
                # Map frame to interval (linear interpolation)
                interval_idx = int((frame_idx / max(1, self.n_frames - 1)) * max(1, total_intervals - 1))
                interval_idx = min(interval_idx, total_intervals - 1)
                
                # Store the actual interval distance (in km)
                interval_distance_km = interval_idx * (self.interval_distance / 1000.0)  # Convert meters to km
                all_interval_distances.append(interval_distance_km)
                
                # Get positions for this interval
                if interval_idx < len(interval_positions):
                    current_positions = interval_positions[interval_idx]
                    all_positions.append(current_positions[0] if current_positions else (0, 0))
                    all_drone_positions.append(current_positions)
                else:
                    # Use last available positions
                    last_positions = interval_positions[-1] if interval_positions else [(0, 0)]
                    all_positions.append(last_positions[0])
                    all_drone_positions.append(last_positions)
                
                # Get path coordinates for natural rendering
                if interval_idx < len(interval_path_coordinates[0]) if interval_path_coordinates else False:
                    frame_path_coords = []
                    for drone_idx in range(len(interval_path_coordinates)):
                        if interval_idx < len(interval_path_coordinates[drone_idx]):
                            frame_path_coords.append(interval_path_coordinates[drone_idx][interval_idx])
                        else:
                            frame_path_coords.append(interval_path_coordinates[drone_idx][-1])
                    all_path_coordinates.append(frame_path_coords)
                else:
                    # Use last available coordinates or empty
                    if all_path_coordinates:
                        all_path_coordinates.append(all_path_coordinates[-1])
                    else:
                        all_path_coordinates.append([])
                
                # Get metrics for this interval
                if interval_idx < len(interval_metrics):
                    all_metrics.append(interval_metrics[interval_idx])
                else:
                    # Use last available metrics
                    all_metrics.append(interval_metrics[-1] if interval_metrics else 
                                     {'area_covered': 0, 'likelihood_score': 0, 'victims_found_pct': 0})
            
            num_drones = len(paths) if paths else 0
            total_time = time.time() - start_time
            log.info(f"Finished animation data preparation for {algorithm_name} in {total_time:.2f} seconds.")
            
            return {
                'positions': all_positions,
                'drone_positions': all_drone_positions,
                'path_coordinates': all_path_coordinates,  # Natural path coordinates for rendering
                'metrics': all_metrics,
                'interval_distances': all_interval_distances,  # Store interval distances in km
                'num_drones': num_drones
            }
            
        except Exception as e:
            log.error(f"Error in animation data preparation for {algorithm_name}: {e}")
            return {'positions': [], 'drone_positions': [], 'path_coordinates': [], 'metrics': [], 'interval_distances': [], 'num_drones': 0}

    def _create_comparative_frame(self, frame_idx, all_animation_data, algorithm_axes, 
                                  ax_area, ax_score, ax_victims):
        """Create a single comparative frame with all algorithm visualizations."""
        try:
            # Plot each algorithm
            for alg_name, ax in algorithm_axes.items():
                setup_algorithm_plot(ax, self.item, self.victims_gdf, self.crs, alg_name, self.algorithm_colors)
                
                if alg_name in all_animation_data:
                    animation_data = all_animation_data[alg_name]
                    if frame_idx < len(animation_data['drone_positions']):
                        current_drone_positions = animation_data['drone_positions'][frame_idx]
                        
                        # Plot drone paths and current positions using efficient method
                        plot_drone_paths(ax, animation_data, frame_idx, self.drone_colors)
                        plot_current_drone_positions(ax, current_drone_positions, self.drone_colors, self.path_evaluator.detection_radius)

            # Create time-series graphs
            create_time_series_graphs(frame_idx, all_animation_data, ax_area, ax_score, ax_victims, 
                                    self.algorithm_colors, self.interval_distance / 1000.0)
            
        except Exception as e:
            log.warning(f"Error creating comparative frame {frame_idx}: {e}")



if __name__ == "__main__":
    log.info("=== Comparative Coverage Video Generation ===")
    
    # Configuration
    data_dir = "sarenv_dataset/19"
    output_dir = Path("coverage_videos")
    interval_distance = 1000.0  # Distance in meters between metric calculations (determines video granularity)
    
    try:
        # Initialize evaluator
        evaluator = ComparativeEvaluator(
            dataset_directory=data_dir,
            evaluation_sizes=["medium"],
            num_drones=3,
            num_lost_persons=100,
            budget=400000,
        )
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {output_dir.absolute()}")
        
        # Load datasets
        log.info("Loading datasets...")
        evaluator.load_datasets()
        
        size = "medium"
        env_data = evaluator.environments[size]
        item = env_data["item"]
        victims_gdf = env_data["victims"]
        
        # Initialize path evaluator
        path_evaluator = metrics.PathEvaluator(
            item.heatmap,
            item.bounds,
            victims_gdf,
            evaluator.path_generator_config.fov_degrees,
            evaluator.path_generator_config.altitude_meters,
            evaluator.loader._meter_per_bin,
        )
        
        # Get center point in projected coordinates
        center_proj = (
            gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
            .to_crs(env_data["crs"])
            .geometry.iloc[0]
        )
        
        # Generate paths for target algorithms
        target_algorithms = ['Concentric', 'Pizza', 'Greedy', 'RandomWalk']
        algorithms_data = {}
        
        log.info("Generating paths for algorithms...")
        for name, generator in evaluator.path_generators.items():
            if name in target_algorithms:
                log.info(f"Generating paths for {name}...")
                try:
                    generated_paths = generator(
                        center_proj.x,
                        center_proj.y,
                        item.radius_km * 1000,
                        item.heatmap,
                        item.bounds,
                    )
                    if generated_paths:
                        algorithms_data[name] = generated_paths
                        log.info(f"Generated {len(generated_paths)} paths for {name}")
                    else:
                        log.warning(f"No paths generated for {name}")
                except Exception as e:
                    log.error(f"Error generating paths for {name}: {e}")
        
        if not algorithms_data:
            log.error("No algorithm data was generated successfully")
            exit(1)
        
        # Create video generator and generate video
        log.info("Creating video generator...")
        video_generator = ComparativeCoverageVideoGenerator(
            item, victims_gdf, path_evaluator, env_data["crs"], output_dir, 
            interval_distance=interval_distance
        )
        
        log.info("Generating comparative coverage video...")
        video_generator.create_comparative_video(algorithms_data)
        
        log.info("=== Comparative Coverage Video Generation Complete ===")
        log.info(f"Video saved in: {output_dir.absolute()}")
        
    except Exception as e:
        log.error(f"Fatal error during video generation: {e}")
        raise
