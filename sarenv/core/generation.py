# sarenv/core/environment.py
import concurrent.futures
import json
import os

import elevation
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Point
from skimage.draw import (
    polygon as ski_polygon,
)

from ..io.osm_query import query_features
from ..utils import (
    logging_setup,
)
from ..utils.geo import image_to_world, world_to_image
from .geometries import GeoPolygon

log = logging_setup.get_logger()
EPS = 1e-9

FEATURE_PROBABILITIES ={
            "linear": 0.25,
            "field": 0.14,
            "structure": 0.13,
            "road": 0.13,
            "drainage": 0.12,
            "water": 0.08,
            "brush": 0.02,
            "scrub": 0.03,
            "woodland": 0.07,
            "rock": 0.04,
        }

def process_feature_osm(key_val_pair, query_polygon_wgs84, projected_crs):
    """
    Queries OSM for a given feature type, processes the geometries, and reprojects them.
    This function is defined at the module level to be pickleable for multiprocessing.
    """
    key, tag_dict = key_val_pair
    osm_geometries_dict = query_features(query_polygon_wgs84, tag_dict)

    if osm_geometries_dict is None:  # query_features now returns a dict or None
        log.warning(
            f"No geometries returned from OSM query for features: {key}"
        )
        return key, None  # Return key and None for the GeoDataFrame

    all_geoms_for_key = []
    for geom in osm_geometries_dict.values():
        if geom is not None and not geom.is_empty:
            if hasattr(geom, "geoms"):  # MultiGeometry
                all_geoms_for_key.extend(
                    g for g in geom.geoms if g is not None and not g.is_empty
                )
            else:
                all_geoms_for_key.append(geom)

    if not all_geoms_for_key:
        log.info(
            f"No valid geometries found for feature type '{key}' after filtering empty ones."
        )
        return key, None

    gdf_wgs84 = gpd.GeoDataFrame(geometry=all_geoms_for_key, crs="EPSG:4326")
    gdf_projected = gdf_wgs84.to_crs(projected_crs)
    log.info(
        f"Processed {len(gdf_projected)} geometries for feature type '{key}'"
    )
    return key, gdf_projected


def interpolate_line(line, distance):
    """Interpolates points along a LineString at a given distance."""
    if distance <= 0:
        return [shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])]

    points = []
    for i in range(len(line.coords) - 1):
        segment = LineString([line.coords[i], line.coords[i + 1]])
        segment_length = segment.length
        num_points = max(1, int(segment_length / distance))
        points.extend(
            segment.interpolate(float(j) / num_points * segment_length)
            for j in range(num_points)
        )
    points.append(
        shapely.Point(line.coords[-1])
    )  # Ensure the last point is included
    return points


def generate_heatmap_task(
    feature_key,
    geometry_series,
    sample_distance,
    xedges,
    yedges,
    meter_per_bin,
    minx,
    miny,
    buffer_val,
    infill_geometries=True,
):
    """
    Generates a heatmap for a given set of geometries.
    This function is defined at the module level to be pickleable for multiprocessing.
    """
    heatmap = np.zeros((len(yedges) - 1, len(xedges) - 1), dtype=float)
    skipped_points = 0

    for geometry in geometry_series:
        if geometry is None or geometry.is_empty:
            continue

        current_geom_img_coords_x = []
        current_geom_img_coords_y = []

        if isinstance(geometry, LineString):
            points_on_line = interpolate_line(geometry, sample_distance)
            if points_on_line:
                world_x = [p.x for p in points_on_line]
                world_y = [p.y for p in points_on_line]
                img_x, img_y = world_to_image(
                    np.array(world_x), np.array(world_y), meter_per_bin, minx, miny, buffer_val
                )
                current_geom_img_coords_x.extend(img_x)
                current_geom_img_coords_y.extend(img_y)
        elif isinstance(geometry, shapely.geometry.Polygon):
            if infill_geometries:
                ext_coords_world = np.array(list(geometry.exterior.coords))
                ext_coords_img_x_arr, ext_coords_img_y_arr = world_to_image(
                    ext_coords_world[:, 0], ext_coords_world[:, 1], meter_per_bin, minx, miny, buffer_val
                )
                rr, cc = ski_polygon(ext_coords_img_y_arr, ext_coords_img_x_arr, shape=heatmap.shape)
                current_geom_img_coords_y.extend(rr)
                current_geom_img_coords_x.extend(cc)
            else:
                points_on_exterior = interpolate_line(geometry.exterior, sample_distance)
                if points_on_exterior:
                    world_x = [p.x for p in points_on_exterior]
                    world_y = [p.y for p in points_on_exterior]
                    img_x, img_y = world_to_image(
                        np.array(world_x), np.array(world_y), meter_per_bin, minx, miny, buffer_val
                    )
                    current_geom_img_coords_x.extend(img_x)
                    current_geom_img_coords_y.extend(img_y)

            for interior in geometry.interiors:
                interior_coords_world = np.array(list(interior.coords))
                interior_coords_img_x, interior_coords_img_y = world_to_image(
                    interior_coords_world[:, 0], interior_coords_world[:, 1], meter_per_bin, minx, miny, buffer_val
                )
                for ix, iy in zip(interior_coords_img_x, interior_coords_img_y):
                    if ix in current_geom_img_coords_x and iy in current_geom_img_coords_y:
                        idx = current_geom_img_coords_x.index(ix)
                        if current_geom_img_coords_y[idx] == iy:
                            current_geom_img_coords_x.pop(idx)
                            current_geom_img_coords_y.pop(idx)
        elif isinstance(geometry, shapely.geometry.Point):
            skipped_points += 1
        else:
            log.warning(f"Unsupported geometry type for heatmap: {type(geometry)} for feature {feature_key}")
            continue

        if current_geom_img_coords_x:
            valid_indices = [
                i for i, (x, y) in enumerate(zip(current_geom_img_coords_x, current_geom_img_coords_y))
                if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]
            ]
            if valid_indices:
                valid_x = np.array(current_geom_img_coords_x)[valid_indices]
                valid_y = np.array(current_geom_img_coords_y)[valid_indices]
                heatmap[valid_y, valid_x] = 1

    if skipped_points > 0:
        log.warning(f"Skipped {skipped_points} Point geometries for feature {feature_key} during heatmap generation.")

    return heatmap

class EnvironmentBuilder:
    def __init__(self):
        self.polygon = None
        self.meter_per_bin = 3
        self.sample_distance = 1
        self.buffer = 0
        self.tags = {}
        self.projected_crs = None  # Add projected_crs attribute

    def set_polygon(self, polygon):
        self.polygon = polygon
        return self

    def set_sample_distance(self, sample_distance):
        self.sample_distance = sample_distance
        return self

    def set_meter_per_bin(self, meter_per_bin):
        self.meter_per_bin = meter_per_bin
        return self

    def set_projected_crs(self, crs: str):
        self.projected_crs = crs
        return self

    def set_buffer(self, buffer_val):
        self.buffer = buffer_val
        return self

    def set_features(self, features):
        if not isinstance(features, dict):
            raise ValueError("Features must be a dictionary")
        self.tags.update(features)
        return self

    def set_feature(self, name, tags):
        self.tags[name] = tags
        return self

    def build(self):
        if self.polygon is None:
            raise ValueError("Polygon must be set before building the environment.")
        if self.projected_crs is None:
            raise ValueError("Projected CRS must be set before building.")

        return Environment(
            self.polygon,
            self.sample_distance,
            self.meter_per_bin,
            self.buffer,
            self.tags,
            self.projected_crs,  # Pass the CRS
        )

class Environment:
    def __init__(
        self,
        bounding_polygon,
        sample_distance,
        meter_per_bin,
        buffer_val,
        tags,
        projected_crs,
    ):
        self.tags = tags
        self.sample_distance = sample_distance
        self.meter_per_bin = meter_per_bin
        self.buffer_val = buffer_val
        self.projected_crs = projected_crs  # Store the projected CRS

        self.polygon: GeoPolygon | None = None
        self.xedges: np.ndarray | None = None
        self.yedges: np.ndarray | None = None
        self.heatmaps: dict[str, np.ndarray | None] = {}
        self.features: dict[str, gpd.GeoDataFrame | None] = {}
        self.heightmap: np.ndarray | None = None # New attribute for heightmap

        self.polygon = GeoPolygon(
            bounding_polygon, crs="EPSG:4326"
        )  # make sure it is WGS84
        self.polygon.set_crs(
            self.projected_crs
        )  # Use the dynamically provided projected CRS
        log.info(f"Environment polygon CRS set to: {self.polygon.crs}")

        self.area = self.polygon.geometry.area
        log.info(
            "Area of the polygon: %s m² (approx. %.2f km²)", self.area, self.area / 1e6
        )
        self.minx, self.miny, self.maxx, self.maxy = self.polygon.geometry.bounds

        num_bins_x = int(
            abs(self.maxx - self.minx + 2 * self.buffer_val) / self.meter_per_bin
        )
        num_bins_y = int(
            abs(self.maxy - self.miny + 2 * self.buffer_val) / self.meter_per_bin
        )

        if num_bins_x <= 0:
            num_bins_x = 1
        if num_bins_y <= 0:
            num_bins_y = 1

        log.info("Number of bins x: %i y: %i", num_bins_x, num_bins_y)

        self.xedges = np.linspace(
            self.minx - self.buffer_val, self.maxx + self.buffer_val, num_bins_x + 1
        )
        self.yedges = np.linspace(
            self.miny - self.buffer_val, self.maxy + self.buffer_val, num_bins_y + 1
        )

        self._load_features()


    def _load_features(self):
        query_polygon_wgs84 = GeoPolygon(self.polygon.geometry, crs=self.polygon.crs)
        query_polygon_wgs84.set_crs("EPSG:4326")  # Ensure the query polygon is in WGS84

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Prepare arguments for each process
            tasks = [
                (item, query_polygon_wgs84, self.projected_crs)
                for item in self.tags.items()
            ]
            # Map tasks to futures
            future_to_key = {
                executor.submit(process_feature_osm, *task): task[0][0] for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    _, feature_gdf = future.result()
                    self.features[key] = feature_gdf
                    if feature_gdf is not None:
                        log.info(
                            f"Stored {len(feature_gdf)} features for '{key}' in CRS {feature_gdf.crs}"
                        )
                    else:
                        log.info(f"No features stored for '{key}'")
                except Exception as exc:
                    log.error(f"Error processing feature {key}: {exc}", exc_info=True)
                    self.features[key] = None

    def interpolate_line(self, line, distance):
        if distance <= 0:
            return [shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])]

        points = []
        for i in range(len(line.coords) - 1):
            segment = LineString([line.coords[i], line.coords[i + 1]])
            segment_length = segment.length
            num_points = max(1, int(segment_length / distance))
            points.extend(
                segment.interpolate(float(j) / num_points * segment_length)
                for j in range(num_points)
            )
        points.append(
            shapely.Point(line.coords[-1])
        )  # Ensure the last point is included
        return points

    def generate_heatmaps(self):
        log.info("Generating heatmaps for all features...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            tasks = [
                (
                    key,
                    feature_gdf.geometry,
                    self.sample_distance,
                    self.xedges,
                    self.yedges,
                    self.meter_per_bin,
                    self.minx,
                    self.miny,
                    self.buffer_val,
                )
                for key, feature_gdf in self.features.items()
                if feature_gdf is not None and not feature_gdf.empty
            ]
            future_to_key = {
                executor.submit(generate_heatmap_task, *task): task[0] for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    heatmap_result = future.result()
                    self.heatmaps[key] = heatmap_result
                    log.info(f"Generated heatmap for {key}")
                except Exception as exc:
                    log.error(
                        f"Error generating heatmap for {key}: {exc}", exc_info=True
                    )
                    self.heatmaps[key] = None
        log.info("Heatmap generation complete.")

    def get_combined_heatmap(self):
        if not self.heatmaps:  # If heatmaps dict is empty
            log.info("Individual heatmaps not generated yet. Generating them now.")
            self.generate_heatmaps()
        # Check if any heatmaps were actually generated
        if not any(h is not None for h in self.heatmaps.values()):
            log.warning(
                "No individual heatmaps available to combine. Returning zero map."
            )
            if self.xedges is None or self.yedges is None:
                return None  # Cannot determine shape
            return np.zeros((len(self.yedges) - 1, len(self.xedges) - 1), dtype=float)

        if self.xedges is None or self.yedges is None:
            log.error("Cannot combine heatmaps, xedges or yedges not initialized.")
            return None

        combined_heatmap = np.zeros(
            (len(self.yedges) - 1, len(self.xedges) - 1), dtype=float
        )

        for key, individual_heatmap in self.heatmaps.items():
            if individual_heatmap is None:
                log.warning(
                    f"Skipping feature '{key}' in combined heatmap as its individual heatmap is None."
                )
                continue
            if individual_heatmap.shape != combined_heatmap.shape:
                log.error(
                    f"Shape mismatch for '{key}': {individual_heatmap.shape} vs {combined_heatmap.shape}"
                )
                continue

            alpha = FEATURE_PROBABILITIES.get(key,0)
            filtered_heatmap_part = individual_heatmap.astype(float) * alpha

            combined_heatmap = np.maximum(combined_heatmap, filtered_heatmap_part)
        return combined_heatmap

    def generate_heightmap(self, output_dir=".") -> np.ndarray | None:
        """
        Generates a heightmap for the environment's extent by downloading
        DEM data and sampling it locally.
        """
        log.info("Generating heightmap from local DEM files...")
        if self.polygon is None:
            log.error("Cannot generate heightmap, environment polygon not set.")
            return None

        # Get the bounding box in WGS84
        
        minx, miny, maxx, maxy = self.polygon.set_crs("EPSG:4326").geometry.bounds
        output_path = os.path.join(output_dir, "temp_dem.tif")
        log.info(
            f"Downloading DEM for bounds: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}"
        )
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Download the DEM for the bounds and clip it
            elevation.clip(bounds=(minx, maxx, miny, maxy), output=output_path)

            log.info(f"DEM data saved to {output_path}")

            with rasterio.open(output_path) as dem_dataset:
                # Create a grid of points for sampling
                x_centers = (self.xedges[:-1] + self.xedges[1:]) / 2
                y_centers = (self.yedges[:-1] + self.yedges[1:]) / 2
                xv, yv = np.meshgrid(x_centers, y_centers)
                
                points_proj = [Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]
                grid_gdf_proj = gpd.GeoDataFrame(geometry=points_proj, crs=self.projected_crs)

                # Transform sample points to the DEM's CRS
                grid_gdf_dem_crs = grid_gdf_proj.to_crs(dem_dataset.crs)
                coords = [(p.x, p.y) for p in grid_gdf_dem_crs.geometry]

                # Sample the raster
                elevations = [val[0] for val in dem_dataset.sample(coords)]
                
                # Reshape the flat list of elevations into a 2D grid
                heightmap_grid = np.array(elevations).reshape(len(y_centers), len(x_centers))
                self.heightmap = heightmap_grid
                log.info(f"Successfully generated heightmap with shape {self.heightmap.shape}")
                return self.heightmap
        
        except Exception as e:
            log.error(f"Failed to generate heightmap from local DEM: {e}", exc_info=True)
            return None
        finally:
            # Clean up the temporary DEM file
            if os.path.exists(output_path):
                os.remove(output_path)
                log.info(f"Removed temporary DEM file: {output_path}")


class DataGenerator:
    """
    Generates and exports a complete, multi-size SAR environment dataset.
    """

    def __init__(self):
        self.tags_mapping = {
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
                "natural": ["water", "wetland"],
                "water": True,
                "wetland": True,
                "reservoir": True,
            },
            "brush": {"landuse": ["grass"]},
            "scrub": {"natural": "scrub"},
            "woodland": {"landuse": ["forest", "wood"], "natural": "wood"},
            "field": {"landuse": ["farmland", "farm", "meadow"]},
            "rock": {"natural": ["rock", "bare_rock", "scree", "cliff"]},
        }

        self._builder = EnvironmentBuilder()
        for feature_category, osm_tags in self.tags_mapping.items():
            self._builder.set_feature(feature_category, osm_tags)

        self.size_radii_km = {
            "small": 0.6,
            "medium": 1.8,
            "large": 3.2,
            "xlarge": 9.9,
        }

    def _get_utm_epsg(self, lon: float, lat: float) -> str:
        """Calculates the appropriate UTM zone EPSG code for a given point."""
        zone = int((lon + 180) / 6) + 1
        epsg_code = f"326{zone}" if lat >= 0 else f"327{zone}"
        log.info(f"Determined UTM zone for point ({lon}, {lat}) as EPSG:{epsg_code}")
        return f"EPSG:{epsg_code}"

    def _create_circular_polygon(
        self, center_lon: float, center_lat: float, radius_km: float
    ) -> dict:
        """Creates a GeoJSON dictionary for a circular polygon centered at a point."""
        point = shapely.Point(center_lon, center_lat)
        gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")

        projected_crs = self._get_utm_epsg(center_lon, center_lat)
        gdf_proj = gdf.to_crs(projected_crs)

        radius_meters = radius_km * 1000
        circle_proj = gdf_proj.buffer(radius_meters).iloc[0]

        circle_gdf_proj = gpd.GeoDataFrame(
            [1], geometry=[circle_proj], crs=projected_crs
        )
        circle_gdf_wgs84 = circle_gdf_proj.to_crs("EPSG:4326")

        return circle_gdf_wgs84.geometry.iloc[0]

    def generate_environment(
        self, center_point: tuple[float, float], size: str, meter_per_bin: int = 30
    ) -> "Environment | None":
        """Generates a single Environment object for a given location and size."""
        center_lon, center_lat = center_point
        radius_km = self.size_radii_km[size]
        projected_crs = self._get_utm_epsg(center_lon, center_lat)

        log.info(
            f"Generating data for center ({center_lon:.4f}, {center_lat:.4f}) with size '{size}' (radius: {radius_km} km)."
        )
        geojson_poly = self._create_circular_polygon(center_lon, center_lat, radius_km)

        try:
            env = (
                self._builder.set_polygon(geojson_poly)
                .set_projected_crs(projected_crs)
                .set_meter_per_bin(meter_per_bin)
                .build()
            )
            return env
        except Exception as e:
            log.error(
                f"An error occurred during environment generation: {e}", exc_info=True
            )
            return None

    def export_dataset(
        self,
        center_point: tuple[float, float],
        output_directory: str,
        meter_per_bin: int = 30,
    ):
        """
        Generates and exports a single, large 'master' dataset for the 'xlarge' radius.
        This master dataset contains all features and the full heatmap, which can be
        dynamically clipped later by the DynamicDatasetLoader.

        Args:
            center_point (tuple[float, float]): The (longitude, latitude) center for the dataset.
            output_directory (str): The directory where master files will be saved.
            meter_per_bin (int): The resolution of the heatmap in meters per pixel.
        """
        log.info(f"--- Starting master dataset export to '{output_directory}' ---")
        os.makedirs(output_directory, exist_ok=True)

        # 1. Generate the single, largest environment ('xlarge')
        log.info(
            "--- Generating master environment for the largest radius ('xlarge') ---"
        )

        master_env = self.generate_environment(center_point, "xlarge", meter_per_bin)
        if not master_env:
            log.error("Failed to generate the master environment. Aborting export.")
            return

        # 2. Combine all features from the master environment into one GeoDataFrame
        master_features_list = []
        for key, gdf in master_env.features.items():
            if gdf is not None and not gdf.empty:
                temp_gdf = gdf.copy()
                temp_gdf["feature_type"] = key
                master_features_list.append(temp_gdf)

        if not master_features_list:
            log.error(
                "No features found in the master environment. No features file will be exported."
            )
            master_features_gdf = gpd.GeoDataFrame(
                columns=["geometry", "feature_type"],
                geometry="geometry",
                crs=master_env.projected_crs,
            )
        else:
            master_features_gdf = pd.concat(master_features_list, ignore_index=True)

        log.info("Calculating area-weighted probabilities for master features...")

        def get_influence_area(geom):
            if isinstance(geom, shapely.Polygon):
                return geom.area
            elif isinstance(geom, LineString):
                return geom.buffer(15).area #TODO this should be defined somewhere
            return 0

        master_features_gdf["influence_area"] = master_features_gdf.geometry.apply(
            get_influence_area
        )

        # Normalize by the sum of all influence areas
        total_influence_area = master_features_gdf["influence_area"].sum()

        if total_influence_area > 0:
            master_features_gdf["area_probability"] = (
                master_features_gdf["influence_area"] / total_influence_area
            )
        else:
            master_features_gdf["area_probability"] = 0

        log.info("Finished calculating area-weighted probabilities.")

        # 3. Export the combined features GeoDataFrame with metadata
        geojson_path = os.path.join(output_directory, "features.geojson")

        # Manually create the GeoJSON dictionary to add custom metadata
        # Ensure final export is in WGS84 for broad compatibility
        master_features_gdf.to_crs("EPSG:4326", inplace=True)
        geojson_dict = master_features_gdf.__geo_interface__
        
        geojson_dict["center_point"] = center_point
        geojson_dict["meter_per_bin"] = meter_per_bin
        geojson_dict["radius_km"] = self.size_radii_km["xlarge"]
        geojson_dict["bounds"] = [
            master_env.minx,
            master_env.miny,
            master_env.maxx,
            master_env.maxy,
        ]

        with open(geojson_path, "w") as f:
            json.dump(geojson_dict, f)
        log.info(f"Exported {len(master_features_gdf)} features to {geojson_path}")

        # 4. Generate and export the single, master heatmap
        log.info("--- Generating and exporting master heatmap ---")
        master_heatmap = master_env.get_combined_heatmap()

        if master_heatmap is not None:
            heatmap_path = os.path.join(output_directory, "heatmap.npy")
            np.save(heatmap_path, master_heatmap)
            log.info(
                f"Exported master heatmap (shape: {master_heatmap.shape}) to {heatmap_path}"
            )
        else:
            log.warning(
                "Master heatmap was not generated. No heatmap file will be exported."
            )


        # # 4. Generate the master heightmap
        # log.info("--- Generating master heightmap ---")
        # master_heightmap = master_env.generate_heightmap(output_dir=output_directory)
        # if master_heightmap is None:
        #     log.error("Failed to generate master heightmap. Aborting heightmap export.")
        # else:
        #     heightmap_path = os.path.join(output_directory, "heightmap.npy")
        #     np.save(heightmap_path, master_heightmap)
        #     log.info(
        #         f"Exported master heightmap (shape: {master_heightmap.shape}) to {heightmap_path}"
        #     )
        log.info("--- Master dataset export completed. ---")

