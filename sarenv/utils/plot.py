# sarenv/utils/plot.py
"""
Collection of visualization functions for SARenv data.
"""
import os

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import stats
from shapely.geometry import Point

from sarenv.utils.logging_setup import get_logger
from sarenv.core.loading import SARDatasetItem
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.utils.geo import get_utm_epsg
from sarenv.utils.lost_person_behavior import get_environment_radius

log = get_logger()


FEATURE_COLOR_MAP = {
    # --- Infrastructure / Man-made ---
    # Greys and browns for concrete, metal, and wood structures.
    "structure": '#636363',  # Dark Grey (e.g., buildings)
    "road": '#bdbdbd',       # Light Grey (e.g., roads, paths)
    "linear": '#8B4513',     # Saddle Brown (e.g., fences, railways, pipelines)

    # --- Water Features ---
    # Blues for all water-related elements.
    "water": '#3182bd',      # Strong Blue (e.g., lakes, rivers)
    "drainage": '#9ecae1',   # Light Blue (e.g., ditches, canals)

    # --- Vegetation ---
    # Greens and yellows for different types of plant life.
    "woodland": '#31a354',   # Forest Green (e.g., forests)
    "scrub": '#78c679',      # Muted Green (e.g., scrubland)
    "brush": '#c2e699',      # Very Light Green (e.g., grass)
    "field": '#fee08b',      # Golden Yellow (e.g., farmland, meadows)
    
    # --- Natural Terrain ---
    # Earth tones for rock and soil.
    "rock": '#969696',       # Stony Grey (e.g., cliffs, bare rock)
}
DEFAULT_COLOR = '#f0f0f0' # A very light, neutral default color.

# Color palette for plotting multiple paths
COLORS_BLUE = [
    '#08519c',  # Dark blue
    '#3182bd',  # Medium blue
    '#6baed6',  # Light blue
    '#9ecae1',  # Very light blue
    '#c6dbef',  # Pale blue
]

# === PATH PLOTTING FUNCTIONS ===

def plot_heatmap(item, victims_gdf, generated_paths, name, x_min, x_max, y_min, y_max, output_file):
    """
    Plot paths on a probability heatmap with victims marked.
    
    Args:
        item (SARDatasetItem): The dataset item containing the heatmap
        victims_gdf (GeoDataFrame): GeoDataFrame containing victim locations
        generated_paths (list): List of path geometries to plot
        name (str): Name/title for the plot
        x_min, x_max, y_min, y_max (float): Coordinate bounds for the plot
        output_file (str): Path to save the output PDF file
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Plot heatmap if available
    if item.heatmap is not None:
        ax.imshow(
            item.heatmap,
            extent=[x_min, x_max, y_min, y_max],
            cmap='YlOrRd',
            alpha=0.7,
            origin='lower'
        )
    
    # Plot victims if available
    if not victims_gdf.empty:
        victims_gdf.plot(ax=ax, color='red', marker='x', markersize=50, label='Victims', zorder=5)
    
    # Plot paths
    colors = COLORS_BLUE
    line_width = 3.0
    for idx, path in enumerate(generated_paths):
        color = colors[idx % len(colors)]
        if path.geom_type == 'MultiLineString':
            for line in path.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
        else:
            x, y = path.xy
            ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
    
    # Format axes
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks])
    ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    if not victims_gdf.empty:
        ax.legend()
    
    # Save the plot
    fig.savefig(output_file, format='pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved heatmap plot to {output_file}")

# === EVALUATION PLOTTING FUNCTIONS ===

def plot_aggregate_bars(summary_df, evaluation_size, output_dir="graphs/aggregate"):
    """
    Plot aggregate bar charts for each metric across all datasets.
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with columns for Algorithm, mean values, and CI values
        evaluation_size (str): Size identifier for the evaluation (e.g., "medium", "large")
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = [
        ("Mean_Likelihood_Score", "CI_Likelihood_Score", "Likelihood Score"),
        ("Mean_Time_Discounted", "CI_Time_Discounted", "Time-Discounted Score"),
        ("Mean_Victims_Found", "CI_Victims_Found", "Victims Found (%)"),
        ("Mean_Area_Covered", "CI_Area_Covered", "Area Covered (km²)"),
        ("Mean_Path_Length", "CI_Path_Length", "Total Path Length (km)"),
    ]
    for mean_col, ci_col, label in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df["Algorithm"], summary_df[mean_col], yerr=summary_df[ci_col], capsize=5, alpha=0.7)
        plt.ylabel(label)
        plt.title(f"Algorithm Comparison: {label} (All Datasets)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"aggregate_{label.replace(' ','_').replace('(%)','').replace('(km²)','').replace('(km)','').lower()}_{evaluation_size}.png"))
        plt.close()


def plot_combined_normalized_bars(summary_df, evaluation_size, output_dir="graphs/aggregate"):
    """
    Creates a grouped bar chart comparing algorithms across normalized metrics (0 to 1 scale).
    Excludes Average Detection Distance.
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with columns for Algorithm, mean values, and CI values
        evaluation_size (str): Size identifier for the evaluation (e.g., "medium", "large")
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = [
        ("Mean_Likelihood_Score", "CI_Likelihood_Score", "Likelihood Score"),
        ("Mean_Time_Discounted", "CI_Time_Discounted", "Time-Discounted Score"),
        ("Mean_Victims_Found", "CI_Victims_Found", "Victims Found Score"),
    ]
    algorithms = summary_df["Algorithm"].tolist()
    n_algorithms = len(algorithms)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_algorithms

    # Normalize the mean values between 0 and 1 for each metric
    normalized_means = {}
    for metric in metrics:
        values = summary_df[metric[0]].values
        min_val = values.min()
        max_val = values.max()
        if max_val - min_val == 0:
            normalized_means[metric[0]] = np.ones_like(values)
        else:
            normalized_means[metric[0]] = (values - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10', n_algorithms)

    for i, alg in enumerate(algorithms):
        means = [normalized_means[metric[0]][summary_df["Algorithm"] == alg][0] for metric in metrics]
        cis = [summary_df.loc[summary_df["Algorithm"] == alg, metric[1]].values[0] for metric in metrics]
        positions = x - 0.4 + i * width + width / 2
        ax.bar(positions, means, width, yerr=cis, capsize=5, label=alg, color=colors(i), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in metrics])
    ax.set_ylabel('Normalized Score (0 to 1)')
    ax.set_title('Algorithm Comparison Across Normalized Metrics (All Datasets)')
    ax.legend(title='Algorithm')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'aggregate_normalized_metrics_{evaluation_size}.png'))
    plt.close()


def plot_time_series_with_ci(time_series_results, evaluation_size, output_dir="graphs/plots"):
    """
    For each algorithm, plot mean and 95% CI for time-series metrics.
    
    Args:
        time_series_results (dict): Dictionary with algorithm names as keys and lists of result dictionaries as values
        evaluation_size (str): Size identifier for the evaluation (e.g., "medium", "large")
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    def mean_ci(arrays):
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a)), mode='edge') if len(a) < max_len else a for a in arrays]
        data = np.vstack(padded)
        mean = np.mean(data, axis=0)
        sem = stats.sem(data, axis=0)
        h = sem * stats.t.ppf((1 + 0.95) / 2., data.shape[0] - 1)
        return mean, mean - h, mean + h

    for name, results_list in time_series_results.items():
        if not results_list:
            continue
        combined_likelihoods = [r['combined_cumulative_likelihood'] for r in results_list]
        combined_victims = [r['combined_cumulative_victims'] for r in results_list]

        mean_likelihood, ci_low_likelihood, ci_high_likelihood = mean_ci(combined_likelihoods)
        mean_victims, ci_low_victims, ci_high_victims = mean_ci(combined_victims)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_likelihood = 'tab:blue'
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Average Accumulated Likelihood (%)', color=color_likelihood)
        ax1.plot(100 * mean_likelihood, color=color_likelihood, label='Avg Accumulated Likelihood')
        ax1.fill_between(
            range(len(mean_likelihood)),
            100 * ci_low_likelihood, 100 * ci_high_likelihood,
            color=color_likelihood, alpha=0.3
        )
        ax1.tick_params(axis='y', labelcolor=color_likelihood)

        ax2 = ax1.twinx()
        color_victims = 'tab:red'
        ax2.set_ylabel('Average Victims Found', color=color_victims)
        ax2.plot(mean_victims, color=color_victims, label='Avg Victims Found')
        ax2.fill_between(range(len(mean_victims)), ci_low_victims, ci_high_victims, color=color_victims, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color_victims)

        plt.title(f'Average Combined Metrics with 95% CI for {name}')
        fig.tight_layout()
        filename = os.path.join(output_dir, f'{name}_{evaluation_size}_average_combined_metrics.pdf')
        plt.savefig(filename)
        plt.close()


def plot_combined_time_series_with_ci(time_series_results, evaluation_size, output_dir="graphs/plots"):
    """
    Plot all algorithms in a single figure with four subplots arranged vertically.
    Each subplot shows time-series metrics with 95% CI for one algorithm.
    
    Args:
        time_series_results (dict): Dictionary with algorithm names as keys and lists of result dictionaries as values
        evaluation_size (str): Size identifier for the evaluation (e.g., "medium", "large")
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    def mean_ci(arrays):
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a)), mode='edge') if len(a) < max_len else a for a in arrays]
        data = np.vstack(padded)
        mean = np.mean(data, axis=0)
        sem = stats.sem(data, axis=0)
        h = sem * stats.t.ppf((1 + 0.95) / 2., data.shape[0] - 1)
        return mean, mean - h, mean + h

    # Filter out empty results and get algorithm names
    valid_algorithms = {name: results for name, results in time_series_results.items() if results}
    
    if not valid_algorithms:
        log.warning("No valid time series results to plot")
        return
    
    # Determine the maximum time length across all algorithms for consistent x-axis
    max_time_length = 0
    processed_data = {}
    
    for name, results_list in valid_algorithms.items():
        combined_likelihoods = [r['combined_cumulative_likelihood'] for r in results_list]
        combined_victims = [r['combined_cumulative_victims'] for r in results_list]
        
        mean_likelihood, ci_low_likelihood, ci_high_likelihood = mean_ci(combined_likelihoods)
        mean_victims, ci_low_victims, ci_high_victims = mean_ci(combined_victims)
        
        processed_data[name] = {
            'mean_likelihood': mean_likelihood,
            'ci_low_likelihood': ci_low_likelihood,
            'ci_high_likelihood': ci_high_likelihood,
            'mean_victims': mean_victims,
            'ci_low_victims': ci_low_victims,
            'ci_high_victims': ci_high_victims
        }
        
        max_time_length = max(max_time_length, len(mean_likelihood))
    
    # Create figure with subplots arranged vertically
    n_algorithms = len(valid_algorithms)
    fig, axes = plt.subplots(n_algorithms, 1, figsize=(12, 4 * n_algorithms), sharex=True)
    
    # If only one algorithm, axes is not a list
    if n_algorithms == 1:
        axes = [axes]
    
    # Create common x-axis
    x_axis = np.arange(max_time_length)
    
    # Plot each algorithm in its own subplot
    for i, (name, data) in enumerate(processed_data.items()):
        ax1 = axes[i]
        
        # Pad data to match max_time_length if needed
        current_length = len(data['mean_likelihood'])
        if current_length < max_time_length:
            # Pad with the last value
            pad_width = max_time_length - current_length
            for key in data:
                data[key] = np.pad(data[key], (0, pad_width), mode='edge')
        
        # Plot likelihood on left y-axis
        color_likelihood = 'tab:blue'
        ax1.set_ylabel('Accumulated Likelihood (%)', color=color_likelihood)
        ax1.plot(x_axis, 100 * data['mean_likelihood'], color=color_likelihood, 
                label='Avg Accumulated Likelihood', linewidth=2)
        ax1.fill_between(
            x_axis,
            100 * data['ci_low_likelihood'], 
            100 * data['ci_high_likelihood'],
            color=color_likelihood, alpha=0.3
        )
        ax1.tick_params(axis='y', labelcolor=color_likelihood)
        ax1.grid(True, alpha=0.3)
        
        # Plot victims on right y-axis
        ax2 = ax1.twinx()
        color_victims = 'tab:red'
        ax2.set_ylabel('Victims Found', color=color_victims)
        ax2.plot(x_axis, data['mean_victims'], color=color_victims, 
                label='Avg Victims Found', linewidth=2)
        ax2.fill_between(x_axis, data['ci_low_victims'], data['ci_high_victims'], 
                        color=color_victims, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color_victims)
        
        # Set title for each subplot
        ax1.set_title(f'{name} - Time Series with 95% CI', fontsize=14, fontweight='bold')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel('Time Step', fontsize=12)
    
    # Add overall title
    fig.suptitle(f'Algorithm Comparison: Time Series Analysis ({evaluation_size})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle
    
    # Save the plot
    filename = os.path.join(output_dir, f'combined_time_series_{evaluation_size}.pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    log.info(f"Saved combined time series plot to {filename}")
    plt.close()


def plot_single_evaluation_results(results_df, evaluation_sizes, output_dir="graphs"):
    """
    Generates and saves plots for single evaluation results.
    
    Args:
        results_df (pd.DataFrame): The dataframe of results to plot
        evaluation_sizes (list): List of evaluation sizes for ordering
        output_dir (str): The directory to save the plots in
    """
    if results_df is None or results_df.empty:
        log.error("No results to plot. Please run an evaluation first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = [
        "Likelihood Score",
        "Time-Discounted Score",
        "Victims Found (%)",
        "Area Covered (km²)",
        "Total Path Length (km)",
    ]

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=results_df,
            x="Dataset",
            y=metric,
            hue="Algorithm",
            order=evaluation_sizes,
        )
        plt.title(f"Comparison of Algorithms: {metric}", fontsize=16)
        plt.ylabel(metric)
        plt.xlabel("Dataset Size")
        plt.legend(title="Algorithm")
        plt.tight_layout()

        plot_filename = os.path.join(
            output_dir, f"plot_{metric.replace(' ', '_').replace('(%)','').replace('(m)','').lower()}.png"
        )
        plt.savefig(plot_filename)
        log.info(f"Saved plot to {plot_filename}")
        plt.close()


# === EXISTING PLOTTING FUNCTIONS ===

def visualize_heatmap_matplotlib(item: SARDatasetItem, data_crs: str, plot_basemap: bool = True):
    fig, ax = plt.subplots(figsize=(12, 10))
    minx, miny, maxx, maxy = item.bounds
    im = ax.imshow(item.heatmap, extent=(minx, maxx, miny, maxy), origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Probability Density")
    if plot_basemap:
        cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7)
    ax.set_title(f"Heatmap Visualization: Size '{item.size}'")
    ax.set_xlabel("Easting (meters)"); ax.set_ylabel("Northing (meters)")
    plt.tight_layout()

def visualize_heatmap(item: SARDatasetItem, plot_basemap: bool = True, plot_inset: bool = True, plot_show = True):
    """
    Creates a plot to visualize the heatmap with a circular, magnified inset.
    The colorbar is positioned to the right of the inset.

    Args:
        item (SARDatasetItem): The loaded dataset item to visualize.
        plot_basemap (bool): Whether to plot the basemap.
        plot_inset (bool): Whether to plot the inset.
    """
    log.info(f"Generating heatmap visualization for size: {item.size}...")

    fig, ax = plt.subplots(figsize=(15, 13))

    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    minx, miny, maxx, maxy = item.bounds
    # Plot main heatmap
    im = ax.imshow(
        item.heatmap, extent=(minx, maxx, miny, maxy), origin="lower", cmap="YlOrRd"
    )

    if plot_basemap:
        cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7, zorder=0)

    radii = get_environment_radius(item.environment_type, item.environment_climate)
    center_point_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    )
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    legend_handles = []
    colors = ["blue", "orange", "red", "green"]
    labels = ["25th", "50th", "75th", "95th"]
    # Plot main radius circles with zorder=2 so they are on top of connector lines
    for idx, r in enumerate(radii):
        circle = center_point_proj.buffer(r * 1000).iloc[0]
        color = colors[idx % len(colors)]
        gpd.GeoSeries([circle], crs=data_crs).boundary.plot(
            ax=ax, edgecolor=color, linestyle="--", linewidth=2, alpha=1, zorder=2)
        label = f"Radius: {labels[idx]} ({r} km)"
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5, linestyle="--", label=label)
        )
    ax.legend(handles=legend_handles, title="Legend", loc="upper left", fontsize=16, title_fontsize=18)

    if plot_inset:
        medium_radius_idx = 1
        medium_radius_m = radii[medium_radius_idx] * 1000

        inset_width = 0.60
        inset_height = 0.60
        ax_inset = ax.inset_axes([1.05, 0.5 - (inset_height / 2), inset_width, inset_height])

        # Plot full heatmap and circles on the inset axes
        ax_inset.imshow(item.heatmap, extent=(minx, maxx, miny, maxy), origin="lower", cmap="YlOrRd")
        if plot_basemap:
            cx.add_basemap(ax_inset, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7)

        for idx, r in enumerate(radii):
            circle_geom = center_point_proj.buffer(r * 1000).iloc[0]
            gpd.GeoSeries([circle_geom], crs=data_crs).boundary.plot(
                ax=ax_inset, edgecolor=colors[idx % len(colors)], linestyle="--", linewidth=2, alpha=1)

        # Set the zoom level for the inset
        center_x = center_point_proj.geometry.x.iloc[0]
        center_y = center_point_proj.geometry.y.iloc[0]
        ax_inset.set_xlim(center_x - medium_radius_m, center_x + medium_radius_m)
        ax_inset.set_ylim(center_y - medium_radius_m, center_y + medium_radius_m)

        # Make the inset circular
        circle_patch = Circle((0.5, 0.5), 0.5, transform=ax_inset.transAxes, facecolor='none', edgecolor='black', linewidth=1)
        ax_inset.set_clip_path(circle_patch)
        ax_inset.patch.set_alpha(0.0)

        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        # Set zorder for connector lines to 1 to be under the main rings
        pp1, c1, c2 = mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp1.set_zorder(0)
        c1.set_zorder(0)
        c2.set_zorder(0)

        pp2, c3, c4 = mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp2.set_zorder(0)
        c3.set_zorder(0)
        c4.set_zorder(0)

        cbar_ax = fig.add_axes([0.88, 0.25, 0.03, 0.5])
        fig.colorbar(im, cax=cbar_ax, label="Probability Density")
    else:
        fig.colorbar(im, ax=ax, shrink=0.8, label="Probability Density")
    
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks])
    ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"heatmap_{item.size}_magnified.pdf", bbox_inches='tight')
    if plot_show:
        plt.show()


def visualize_features(item: SARDatasetItem, plot_basemap: bool = False, plot_inset: bool = False, num_lost_persons: int = 0, plot_show=True):
    """
    Creates a plot with a circular, magnified callout of the "medium" radius.

    Args:
        item (SARDatasetItem): The loaded dataset item to visualize.
        plot_basemap (bool): Whether to plot the basemap.
        plot_inset (bool): Whether to plot the inset.
        sample_lost_persons (bool): Whether to sample and plot lost person locations.
        num_lost_persons (int): Number of lost person locations to generate.
    """
    if not item:
        log.warning("No dataset items provided to visualize.")
        return

    radii = get_environment_radius(item.environment_type, item.environment_climate)

    log.info(f"Generating nested visualization for '{item.size}' with circular magnification...")
    center_point_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    )
    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    fig, ax = plt.subplots(figsize=(18, 15))

    legend_handles = []
    # Set zorder for features to 1
    for feature_type, data in item.features.groupby("feature_type"):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
        data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.7, zorder=1)
        legend_handles.append(Patch(color=color, label=feature_type.capitalize()))

    if plot_basemap:
        # The basemap will have a zorder of 0 by default
        cx.add_basemap(ax, crs=item.features.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
    
    # Sample and plot lost person locations if requested
    lost_person_gdf = None
    if num_lost_persons > 0:
        log.info(f"Generating {num_lost_persons} lost person locations...")
        lost_person_generator = LostPersonLocationGenerator(item)
        locations = lost_person_generator.generate_locations(num_lost_persons, 0) # 0% random samples
        
        if locations:
            lost_person_gdf = gpd.GeoDataFrame(geometry=locations, crs=item.features.crs)
            lost_person_gdf.plot(ax=ax, marker='*', color='red', markersize=200, zorder=1, label="Lost Person")
            legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Lost Person'))

    colors = ["blue", "orange", "red", "green"]
    labels = ["25th", "50th", "75th", "95th"]
    # Set zorder for rings to 2 to be on top of features
    for idx, r in enumerate(radii):
        circle = center_point_proj.buffer(r * 1000).iloc[0]
        color = colors[idx % len(colors)]
        gpd.GeoSeries([circle], crs=data_crs).boundary.plot(
            ax=ax, edgecolor=color, linestyle="--", linewidth=2, alpha=1, zorder=2)
        label = f"Radius: {labels[idx]} ({r} km)"
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5, linestyle="--", label=label)
        )

    ax.legend(handles=legend_handles, title="Legend", loc="upper left", fontsize=10, title_fontsize=12)

    if plot_inset:
        medium_radius_idx = 1
        medium_radius_m = radii[medium_radius_idx] * 1000

        inset_width = 0.60
        inset_height = 0.60
        ax_inset = ax.inset_axes([1.05, 0.5 - (inset_height / 2), inset_width, inset_height])

        # Create clipping circle for the inset
        center_x = center_point_proj.geometry.x.iloc[0]
        center_y = center_point_proj.geometry.y.iloc[0]
        clipping_circle = Point(center_x, center_y).buffer(medium_radius_m)
        clipping_gdf = gpd.GeoDataFrame([1], geometry=[clipping_circle], crs=data_crs)

        # Ensure features are in the same CRS as the clipping circle for proper clipping
        features_proj = item.features.to_crs(data_crs)
        clipped_features = gpd.clip(features_proj, clipping_gdf)

        # Plot the CLIPPED features on the inset axes
        if plot_basemap:
            cx.add_basemap(ax_inset, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik)
        
        if not clipped_features.empty:
            for feature_type, data in clipped_features.groupby("feature_type"):
                data.plot(ax=ax_inset, color=FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR), alpha=0.7)
        
        for idx, r in enumerate(radii):
            circle_geom = center_point_proj.buffer(r * 1000).iloc[0]
            gpd.GeoSeries([circle_geom], crs=data_crs).boundary.plot(
                ax=ax_inset, edgecolor=colors[idx % len(colors)], linestyle="--", linewidth=2, alpha=1
            )

        # Add lost person locations to inset if they were generated
        if num_lost_persons>0 and lost_person_gdf is not None:
            # Ensure lost person data is in the same CRS as the clipping circle
            lost_person_proj = lost_person_gdf.to_crs(data_crs)
            clipped_lost_persons = gpd.clip(lost_person_proj, clipping_gdf)
            
            if not clipped_lost_persons.empty:
                clipped_lost_persons.plot(ax=ax_inset, marker='*', color='red', markersize=250, zorder=3)

        ax_inset.set_xlim(center_x - medium_radius_m, center_x + medium_radius_m)
        ax_inset.set_ylim(center_y - medium_radius_m, center_y + medium_radius_m)

        circle = Circle((0.5, 0.5), 0.5, transform=ax_inset.transAxes, facecolor='none', edgecolor='black', linewidth=1)
        ax_inset.set_clip_path(circle)
        ax_inset.patch.set_alpha(0.0)

        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        # Set zorder for connector lines to 1 to be under the main rings
        pp1, c1, c2 = mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp1.set_zorder(0)
        c1.set_zorder(0)
        c2.set_zorder(0)

        pp2, c3, c4 = mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp2.set_zorder(0)
        c3.set_zorder(0)
        c4.set_zorder(0)
        
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks])
    ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"features_{item.size}_circular_magnified_final.pdf", bbox_inches='tight')
    if plot_show:
        plt.show()