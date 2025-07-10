# examples/edouard_polygon_heatmap.py
"""
Script for Edouard: Generate SAR data using a polygon and visualize heatmaps
in both 2D and 3D perspective views.
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import shapely

from sarenv import (
    CLIMATE_TEMPERATE,
    DataGenerator,
    DatasetLoader,
    ENVIRONMENT_TYPE_FLAT,
    get_logger,
)

log = get_logger()


def create_3d_heatmap_matplotlib(heatmap, bounds):
    """
    Create a clean 3D perspective heatmap using matplotlib with no axes or labels.

    Args:
        heatmap (np.ndarray): 2D heatmap array
        bounds (tuple): (minx, miny, maxx, maxy) bounds of the heatmap
    """
    minx, miny, maxx, maxy = bounds

    # Create coordinate grids
    x = np.linspace(minx, maxx, heatmap.shape[1])
    y = np.linspace(miny, maxy, heatmap.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create 3D plot with no axes
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    ax.plot_surface(X, Y, heatmap, cmap='YlOrRd', alpha=0.9,
                   linewidth=0, antialiased=True)

    # Remove all axes, labels, and decorations
    ax.set_axis_off()

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    # Remove margins and make plot fill the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, ax


def create_3d_heatmap_plotly(heatmap, bounds):
    """
    Create a clean interactive 3D perspective heatmap using plotly with minimal decorations.

    Args:
        heatmap (np.ndarray): 2D heatmap array
        bounds (tuple): (minx, miny, maxx, maxy) bounds of the heatmap
    """
    minx, miny, maxx, maxy = bounds

    # Create coordinate grids
    x = np.linspace(minx, maxx, heatmap.shape[1])
    y = np.linspace(miny, maxy, heatmap.shape[0])

    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=x,
        y=y,
        z=heatmap,
        colorscale='YlOrRd',
        showscale=False,  # Remove colorbar
        hoverinfo='skip'  # Remove hover information
    )])

    # Update layout to remove all axes and decorations
    fig.update_layout(
        showlegend=False,
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {
                "eye": {"x": 1.5, "y": 1.5, "z": 1.5}
            }
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=900,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_wireframe_heatmap(heatmap, bounds):
    """
    Create a clean wireframe perspective view of the heatmap with no axes or labels.

    Args:
        heatmap (np.ndarray): 2D heatmap array
        bounds (tuple): (minx, miny, maxx, maxy) bounds of the heatmap
    """
    minx, miny, maxx, maxy = bounds

    # Create coordinate grids
    x = np.linspace(minx, maxx, heatmap.shape[1])
    y = np.linspace(miny, maxy, heatmap.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create 3D wireframe plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create wireframe plot
    ax.plot_wireframe(X, Y, heatmap, color='darkred', alpha=0.7, linewidth=0.8)

    # Remove all axes, labels, and decorations
    ax.set_axis_off()

    # Set viewing angle
    ax.view_init(elev=25, azim=60)

    # Remove margins and make plot fill the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, ax


def create_clean_2d_heatmap(heatmap):
    """
    Create a clean 2D heatmap visualization without any axes, labels, or decorations.

    Args:
        heatmap (np.ndarray): 2D heatmap array
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the heatmap
    ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', origin='lower')
    
    # Remove all axes, labels, and decorations
    ax.set_axis_off()
    
    # Remove margins and make plot fill the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig, ax


def run_edouard_polygon_example():
    """
    Generate SAR data using a polygon and create multiple heatmap visualizations
    including 2D standard view, 3D perspective, and wireframe views.
    """
    log.info("--- Starting Edouard's Polygon Heatmap Generation ---")

    # 1. Initialize the generator
    data_gen = DataGenerator()

    # 2. Define a custom polygon (you can modify these coordinates)        # Example: A pentagon around a fictional search area
    polygon_coords = [
        [10.280, 55.140],  # Point 1
        [10.300, 55.140],  # Point 2
        [10.310, 55.148],  # Point 3
        [10.295, 55.155],  # Point 4
        [10.275, 55.150],  # Point 5
        [10.280, 55.140],  # Close the polygon
    ]

    # Create a shapely Polygon object
    custom_polygon = shapely.geometry.Polygon(polygon_coords)

    output_dir = "sarenv_dataset_edouard"

    # 3. Generate the dataset from the polygon
    log.info("--- Generating dataset from polygon ---")
    data_gen.export_dataset_from_polygon(
        polygon=custom_polygon,
        output_directory=output_dir,
        environment_climate=CLIMATE_TEMPERATE,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        meter_per_bin=10,  # Higher resolution for better 3D visualization
    )

    # 4. Load the generated dataset
    log.info("--- Loading generated dataset ---")
    try:
        loader = DatasetLoader(dataset_directory=output_dir)
        size_to_load = "xlarge"  # Use the largest size for best visualization
        item = loader.load_environment(size_to_load)

        if not item:
            log.error(f"Could not load the specified size: '{size_to_load}'")
            return

        log.info(f"Successfully loaded dataset with heatmap shape: {item.heatmap.shape}")
        log.info(f"Heatmap bounds: {item.bounds}")
        log.info(f"Heatmap statistics - min: {np.min(item.heatmap):.6f}, max: {np.max(item.heatmap):.6f}")

        # 5. Create multiple visualizations

        # 5a. Clean 2D heatmap without axes or decorations
        log.info("--- Creating clean 2D heatmap visualization ---")
        fig_2d, ax_2d = create_clean_2d_heatmap(item.heatmap)
        plt.savefig(f"heatmap_2d_clean_{item.size}.pdf", bbox_inches='tight')
        plt.show()

        # 5b. 3D Surface plot with matplotlib
        log.info("--- Creating 3D surface heatmap (Matplotlib) ---")
        fig_3d_mpl, ax_3d_mpl = create_3d_heatmap_matplotlib(
            item.heatmap,
            item.bounds
        )
        plt.savefig(f"heatmap_3d_surface_{item.size}.pdf", bbox_inches='tight')
        plt.show()

        # 5c. Wireframe plot
        log.info("--- Creating wireframe heatmap ---")
        fig_wire, ax_wire = create_wireframe_heatmap(
            item.heatmap,
            item.bounds
        )
        plt.savefig(f"heatmap_wireframe_{item.size}.pdf", bbox_inches='tight')
        plt.show()

        # 5d. Interactive 3D plot with plotly
        log.info("--- Creating interactive 3D heatmap (Plotly) ---")
        fig_plotly = create_3d_heatmap_plotly(
            item.heatmap,
            item.bounds
        )

        # Save interactive plot as HTML
        fig_plotly.write_html(f"heatmap_3d_interactive_{item.size}.html")
        log.info(f"Interactive 3D plot saved as: heatmap_3d_interactive_{item.size}.html")

        # Show interactive plot
        fig_plotly.show()

        # 5e. Multiple perspective views
        log.info("--- Creating multiple perspective views ---")
        create_multiple_perspective_views(item)

        log.info("--- All visualizations completed successfully ---")

    except FileNotFoundError:
        log.error(
            f"Error: The dataset directory '{output_dir}' or its master files were not found."
        )
        log.error(
            "Please check if the polygon data generation was successful."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)


def create_multiple_perspective_views(item):
    """
    Create multiple clean 3D perspective views of the same heatmap from different angles.

    Args:
        item (SARDatasetItem): The loaded dataset item
    """
    minx, miny, maxx, maxy = item.bounds
    x = np.linspace(minx, maxx, item.heatmap.shape[1])
    y = np.linspace(miny, maxy, item.heatmap.shape[0])
    X, Y = np.meshgrid(x, y)

    # Define different viewing angles
    views = [
        {"elev": 20, "azim": 45},
        {"elev": 45, "azim": 90},
        {"elev": 60, "azim": 135},
        {"elev": 30, "azim": 180}
    ]

    # Create subplot figure with multiple views
    fig = plt.figure(figsize=(16, 12))

    for i, view in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')

        # Create surface plot
        ax.plot_surface(X, Y, item.heatmap, cmap='YlOrRd',
                       alpha=0.8, linewidth=0, antialiased=True)

        # Set viewing angle
        ax.view_init(elev=view["elev"], azim=view["azim"])

        # Remove all axes, labels, and decorations
        ax.set_axis_off()

    # Remove all margins and spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"heatmap_multiple_perspectives_{item.size}.pdf", bbox_inches='tight')
    plt.show()


def create_custom_polygon_demo():
    """
    Demonstrate how to create different polygon shapes for SAR data generation.
    """
    log.info("--- Polygon Shape Examples ---")

    # Example polygons
    polygons = {
        "Rectangle": [
            [10.280, 55.140], [10.300, 55.140], [10.300, 55.150],
            [10.280, 55.150], [10.280, 55.140]
        ],
    }

    log.info("Available polygon shapes:")
    for name, coords in polygons.items():
        area = shapely.geometry.Polygon(coords).area
        log.info(f"  {name}: {len(coords)-1} vertices, area ≈ {area:.6f} square degrees")

    log.info("To use a different polygon, modify the 'polygon_coords' variable in run_edouard_polygon_example()")


if __name__ == "__main__":
    # Run the main example
    run_edouard_polygon_example()
