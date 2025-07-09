# sarenv/utils/plot.py
"""
Collection of visualization functions for SARenv data.
"""
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from shapely.geometry import Point
from sarenv.core.loading import SARDatasetItem
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.utils.geo import get_utm_epsg
from sarenv.utils.lost_person_behavior import get_environment_radius
from sarenv import get_logger

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

def visualize_features_matplotlib(item: SARDatasetItem, data_crs: str, plot_basemap: bool = True):
    fig, ax = plt.subplots(figsize=(13, 13))
    legend_handles = []
    for feature_type, data in item.features.groupby("feature_type"):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
        data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.7)
        legend_handles.append(Patch(color=color, label=feature_type.capitalize()))
    if plot_basemap:
        cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik)
    ax.legend(handles=legend_handles, title="Legend", loc="upper left")
    ax.set_title(f"Features for Dataset Size: {item.size}")
    ax.set_xlabel("Easting (meters)"); ax.set_ylabel("Northing (meters)")
    plt.tight_layout()

def visualize_heatmap_plotly(item: SARDatasetItem, output_path: str):
    minx, miny, maxx, maxy = item.bounds
    fig = go.Figure(data=go.Heatmap(z=item.heatmap, x=np.linspace(minx, maxx, item.heatmap.shape[1]), y=np.linspace(miny, maxy, item.heatmap.shape[0]), colorscale='Inferno', colorbar=dict(title='Probability Density')))
    fig.update_layout(title=f"Interactive Heatmap: Size '{item.size}'", xaxis_title="Easting (meters)", yaxis_title="Northing (meters)", yaxis_scaleanchor="x", template="plotly_white")
    fig.write_html(output_path, include_plotlyjs="cdn")

def visualize_path_plotly(item: SARDatasetItem, path_name: str, paths: list, colors: list, victims: gpd.GeoDataFrame, data_crs: str, output_path: str, plot_map_features: bool = True):
    fig = go.Figure()
    if plot_map_features:
        features_proj = item.features.to_crs(crs=data_crs)
        for feature_type, data in features_proj.groupby('feature_type'):
            color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
            for i, geom in enumerate(data.geometry):
                if geom.is_empty: continue
                geometries = geom.geoms if geom.geom_type.startswith('Multi') else [geom]
                for j, sub_geom in enumerate(geometries):
                    x, y = (sub_geom.exterior.xy if sub_geom.geom_type == 'Polygon' else sub_geom.xy)
                    fill = 'toself' if sub_geom.geom_type == 'Polygon' else None
                    mode = 'lines'
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), fill=fill, mode=mode, line=dict(color=color), opacity=0.6, name=feature_type.capitalize(), legendgroup=feature_type, showlegend=(i == 0 and j == 0)))

    for i, path in enumerate(paths):
        if not path.is_empty:
            x, y = path.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color=colors[i % len(colors)], width=3), name=f'{path_name} Path', legendgroup=path_name, showlegend=(i == 0)))

    if not victims.empty:
        victims_proj = victims.to_crs(crs=data_crs)
        fig.add_trace(go.Scatter(x=victims_proj.geometry.x, y=victims_proj.geometry.y, mode='markers', marker=dict(symbol='x', color='red', size=12), name='Victim Location'))

    fig.update_layout(title=f"Coverage Path for '{path_name}' Pattern", xaxis_title="Easting (meters)", yaxis_title="Northing (meters)", legend_title_text="Legend", yaxis_scaleanchor="x", template="plotly_white")
    fig.write_html(output_path, include_plotlyjs="cdn")


def visualize_heatmap(item: SARDatasetItem, plot_basemap: bool = True, plot_inset: bool = True):
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
    plt.show()


def visualize_features(item: SARDatasetItem, plot_basemap: bool = False, plot_inset: bool = False, num_lost_persons: int = 0):
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
    plt.show()