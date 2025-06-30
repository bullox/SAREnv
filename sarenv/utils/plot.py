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
from matplotlib.patches import Patch
from sarenv.core.loading import SARDatasetItem


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