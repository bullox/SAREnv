from pathlib import Path

import osmnx as ox
import shapely
from geojson import Feature, FeatureCollection, dump

from sarenv import Logging
from sarenv.core.geometries import GeoPolygon

log = Logging.get_logger()


def query_features(area: GeoPolygon, tags: dict):
    # Check that the geometry has the right crs
    if not area.crs and area.crs != "WGS84":
        msg = "The geometry must use WGS84 CRS!"
        raise ValueError(msg)

    try:
        geometries = ox.features_from_polygon(area.get_geometry(), tags=tags)
    except Exception as e:
        log.warning("An error occurred while querying features: %s", str(e))
        return None
    
    results = {tag: [] for tag in tags}
    for tag in tags:
        if tag in geometries.columns:
            filtered_features = geometries[geometries[tag].notna()]
            results[tag] = shapely.intersection(
                area.get_geometry(), filtered_features.geometry.unary_union
            )
        else:
            log.warning("Tag '%s' not found in the geometries. Skipping.", tag)
        # TODO Also include the metadata of the features for future analysis

    log.info("Extracted %d features with the tags: %s", len(geometries), str(tags))
    return results


def export_as_geojson(boundary: shapely.Polygon, obstacles, features, crs):
    boundary_feature = Feature(geometry=boundary, id="boundary")
    obstacles_feature = Feature(
        geometry=shapely.MultiPolygon(obstacles), id="obstacles"
    )
    task_feature = Feature(geometry=shapely.MultiLineString(features), id="tasks")
    feature_collection = FeatureCollection(
        [boundary_feature, task_feature, obstacles_feature], crs=crs
    )

    # Write the FeatureCollection to a GeoJSON file
    with Path("output.json").open("w") as geojson_file:
        dump(feature_collection, geojson_file)
