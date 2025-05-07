import json
import geopandas as gpd
import matplotlib.pyplot as plt

import osmnx as ox
from shapely.geometry import Polygon
import geopandas as gpd
from matplotlib.patches import Patch
from shapely.geometry import Point
from geopy.distance import geodesic


class GeospatialDataLoader:
    def __init__(self, file_path="test_case.geojson"):
        self.file_path = file_path
        self.boundary = None
        self.load_geojson()

    def download_geojson(
        self, boundary_points, output_file="downloaded_features.geojson"
    ):
        """
        Download features from OpenStreetMap within the boundary defined by boundary_points using OSMnx.

        Parameters:
        -----------
        boundary_points: list of (lat, lon) tuples defining a closed polygon.
        output_file: str, path to save the output GeoJSON file.

        The function downloads features corresponding to:
        structure, road, linear, drainage, water, brush, scrub, woodland, field and rock.
        The downloaded data is converted to a GeoJSON FeatureCollection and saved to output_file.
        """
        # Ensure the polygon is closed
        if boundary_points[0] != boundary_points[-1]:
            boundary_points.append(boundary_points[0])

        # Create a shapely polygon from the boundary points
        # Note: OSMnx expects (lat, lon) format
        polygon = Polygon([(lat, lon) for lat, lon in boundary_points])

        # Define OSM tags mapping for each feature type
        tags_mapping = {
            "structure": {
                "building": True,
                "man_made": True,
                "bridge": True,
                "tunnel": True,
            },
            "road": {"highway": True, "tracktype": True},
            "linear": {
                "railway": True,
                "barrier": True,
                "fence": True,
                "wall": True,
                "pipeline": True,
            },
            "drainage": {"waterway": ["drain", "ditch", "culvert", "canal"]},
            "water": {
                "natural": "water",
                "water": True,
                "wetland": True,
                "reservoir": True,
            },
            "brush": {"landuse": ["grass", "meadow"]},
            "scrub": {"natural": "scrub"},
            "woodland": {"landuse": ["forest", "wood"], "natural": "wood"},
            "field": {"landuse": ["farmland", "farm", "meadow"]},
            "rock": {"natural": ["rock", "bare_rock", "scree", "cliff"]},
        }

        # Dictionary to store features by type
        features_by_type = {}

        # Query features for each type
        for feature_type, tags in tags_mapping.items():
            try:
                # Query OSM for this feature type
                gdf = ox.features_from_polygon(polygon, tags=tags)
                if not gdf.empty:
                    # Convert to specified CRS if needed
                    gdf = gdf.to_crs("EPSG:4326")  # Ensure WGS84
                    features_by_type[feature_type] = gdf
                    print(f"Downloaded {len(gdf)} {feature_type} features")
                else:
                    features_by_type[feature_type] = None
                    print(f"No {feature_type} features found")
            except Exception as e:
                print(f"Error downloading {feature_type}: {str(e)}")
                features_by_type[feature_type] = None

        # Create boundary feature
        boundary_coords = [[lat, lon] for lat, lon in boundary_points]
        boundary_feature = {
            "type": "Feature",
            "id": "boundary",
            "geometry": {"type": "Polygon", "coordinates": [boundary_coords]},
            "properties": {"crs": "EPSG:4326", "name": "boundary"},  # WGS84
        }

        # Create features for the nested collection
        nested_features = []
        for feature_type, gdf in features_by_type.items():
            if gdf is not None and not gdf.empty:
                # Group polygons by type
                polygon_geoms = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

                if not polygon_geoms.empty:
                    # Dissolve all polygons of this type into a MultiPolygon
                    multi_poly = gpd.GeoDataFrame(
                        geometry=[polygon_geoms.unary_union], crs=gdf.crs
                    )

                    # Create a Feature with MultiPolygon geometry
                    feature_json = json.loads(multi_poly.to_json())
                    if feature_json["features"]:
                        multi_polygon_feature = {
                            "type": "Feature",
                            "id": feature_type,
                            "geometry": feature_json["features"][0]["geometry"],
                            "properties": {
                                "crs": "EPSG:4326",
                                "name": feature_type.capitalize(),
                            },
                        }
                        nested_features.append(multi_polygon_feature)

        # Create final GeoJSON
        feature_collection = {"type": "FeatureCollection", "features": nested_features}

        geojson_data = {
            "type": "FeatureCollection",
            "features": [boundary_feature, feature_collection],
        }

        # Save the GeoJSON data to a file
        with open(output_file, "w") as f:
            json.dump(geojson_data, f, indent=4)
        print(f"Downloaded GeoJSON saved to {output_file}")

    def load_geojson(self):
        """Load and parse the GeoJSON file with coordinates assumed to be in (lon, lat) order"""
        with open(self.file_path, "r") as f:
            self.data = json.load(f)

        # Extract Boundary feature
        boundary_feature = [
            f for f in self.data["features"] if f.get("id") == "boundary"
        ][0]
        self.boundary = gpd.GeoDataFrame.from_features(
            [boundary_feature], crs="EPSG:4326"
        )
        nested_collection = [
            f for f in self.data["features"] if f.get("type") == "FeatureCollection"
        ][0]

        # Extract features from nested collection; the GeoJSON coordinates are (lon, lat)
        feature_types = [
            "structure",
            "road",
            "linear",
            "drainage",
            "water",
            "brush",
            "scrub",
            "woodland",
            "field",
            "rock",
        ]

        for feature_type in feature_types:
            features = [
                f for f in nested_collection["features"] if f.get("id") == feature_type
            ]
            if features:
                # Specify CRS to correctly interpret coordinates in (lon, lat) order
                gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                # Explode any MultiPolygon into individual polygon rows
                gdf = gdf.explode(index_parts=False)
                setattr(self, feature_type, gdf)
            else:
                setattr(self, feature_type, None)

        # Print summary of loaded features
        loaded_features = {
            ft: len(getattr(self, ft)) if getattr(self, ft) is not None else 0
            for ft in feature_types
        }
        print(
            f"Loaded: Boundary ({len(self.boundary)}), "
            + ", ".join([f"{k.capitalize()} ({v})" for k, v in loaded_features.items()])
        )

    def visualise_geo_data(self):
        """Visualize the loaded geospatial data using matplotlib with different color coding for each feature type"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot boundary
        self.boundary.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)

        # Define color mapping for each feature type
        color_mapping = {
            "structure": "red",
            "road": "blue",
            "linear": "green",
            "drainage": "purple",
            "water": "cyan",
            "brush": "orange",
            "scrub": "magenta",
            "woodland": "darkgreen",
            "field": "yellow",
            "rock": "brown",
        }

        feature_types = [
            "structure",
            "road",
            "linear",
            "drainage",
            "water",
            "brush",
            "scrub",
            "woodland",
            "field",
            "rock",
        ]

        for feature_type in feature_types:
            gdf = getattr(self, feature_type)
            if gdf is not None and not gdf.empty:
                color = color_mapping.get(feature_type, "black")
                gdf.plot(ax=ax, label=feature_type.capitalize(), color=color, alpha=0.5)

        ax.set_title("Geospatial Data Visualization")

        legend_handles = []
        for feature_type in feature_types:
            gdf = getattr(self, feature_type)
            if gdf is not None and not gdf.empty:
                legend_handles.append(
                    Patch(
                        facecolor=color_mapping[feature_type],
                        edgecolor="black",
                        label=feature_type.capitalize(),
                    )
                )
        if legend_handles:
            ax.legend(handles=legend_handles)
        plt.show()


def visualise_geo_data(boundary, feature_data):
    """Visualize the geospatial data using matplotlib with different color coding for each feature type"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot boundary
    boundary.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)

    # Define color mapping for each feature type
    color_mapping = {
        "structure": "red",
        "road": "blue",
        "linear": "green",
        "drainage": "purple",
        "water": "cyan",
        "brush": "orange",
        "scrub": "magenta",
        "woodland": "darkgreen",
        "field": "yellow",
        "rock": "brown",
    }

    for feature_type, gdf in feature_data.items():
        if gdf is not None and not gdf.empty:
            color = color_mapping.get(feature_type, "black")
            gdf.plot(ax=ax, label=feature_type.capitalize(), color=color, alpha=0.5)

    ax.set_title("Geospatial Data Visualization")

    legend_handles = []
    for feature_type, gdf in feature_data.items():
        if gdf is not None and not gdf.empty:
            legend_handles.append(
                Patch(
                    facecolor=color_mapping[feature_type],
                    edgecolor="black",
                    label=feature_type.capitalize(),
                )
            )
    if legend_handles:
        ax.legend(handles=legend_handles)
    plt.show()


if __name__ == "__main__":
    # Define a center point (latitude, longitude)
    center_point = (37.7749, -122.4194)  # Example: San Francisco

    # Create a circular boundary with a radius of 1 km
    num_points = 100  # Number of points to approximate the circle
    radius_km = 1  # Radius in kilometers
    circle_points = [
        geodesic(kilometers=radius_km).destination(center_point, angle)
        for angle in range(0, 360, int(360 / num_points))
    ]
    circle_boundary = [(point.latitude, point.longitude) for point in circle_points]

    # Initialize the GeospatialDataLoader
    loader = GeospatialDataLoader()

    # Download features within the circular boundary
    loader.download_geojson(
        boundary_points=circle_boundary, output_file="circle_features.geojson"
    )

    # Load the downloaded GeoJSON
    loader.file_path = "circle_features.geojson"
    loader.load_geojson()

    # Visualize the features
    loader.visualise_geo_data()
