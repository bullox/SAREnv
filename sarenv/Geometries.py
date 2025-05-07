import math

import geojson
import pyproj
import shapely
import shapely.plotting as shplt
from shapely.geometry.polygon import orient
import random

from sarenv import Logging

log = Logging.get_logger()


class GeoData:
    def __init__(self, geometry, crs="WGS84"):
        self.geometry = geometry
        self.crs = crs

    def set_crs(self, crs):
        if not isinstance(crs, str):
            msg = "New CRS must be a string."
            raise ValueError(msg)

        if crs != self.crs:
            # Apply the transformer to the geometry
            self._convert_to_crs(crs)
        self.crs = crs
        return self

    def _convert_to_crs(self, crs):  # noqa: ARG002
        msg = "_convert_to_crs(crs) sould be implemented in the data classes!"
        raise NotImplementedError(msg)

    def is_geometry_of_type(self, geometry, expected_class):
        if expected_class and not isinstance(geometry, expected_class):
            msg = f"Geometry must be a {expected_class.__name__}."
            raise ValueError(msg)

    def get_geometry(self):
        return self.geometry

    def buffer(self, distance, quad_segs=1, cap_style="square", join_style="bevel"):
        self.geometry = self.geometry.buffer(distance, quad_segs, cap_style, join_style)
        return self

    def __str__(self):
        return f"Geometry in CRS: {self.crs}\nGeometry: {self.geometry}"

    def __geo_interface__(self):
        return self.geometry.__geo_interface__

    def to_geojson(self, id=None, name=None, properties=None):
        if id is None:
            id = random.randint(0, 1000000000)
        if name is None:
            name = str(id)
        if properties is None:
            properties = {}

        properties["crs"] = self.crs
        properties["name"] = name
        return geojson.Feature(id, self.geometry, properties=properties)


class GeoTrajectory(GeoData):
    def __init__(self, geometry, crs="WGS84"):
        self.is_geometry_of_type(geometry, shapely.LineString)
        super().__init__(geometry, crs)

    def plot(self, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        shplt.plot_line(self.geometry, ax, add_points, color, linewidth, **kwargs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)

        converted_coords = [
            transformer.transform(x, y) for x, y in list(self.geometry.coords)
        ]
        self.geometry = shapely.LineString(converted_coords)


class GeoMultiTrajectory(GeoData):
    def __init__(
        self,
        geometry: (
            shapely.MultiLineString
            | list[shapely.LineString]
            | list[GeoTrajectory]
            | shapely.LineString
        ),
        crs="WGS84",
    ):
        super().__init__(geometry, crs)
        if isinstance(geometry, list):
            for line in geometry:
                self.is_geometry_of_type(line, shapely.LineString)

            super().__init__(shapely.MultiLineString(geometry), crs)
        elif isinstance(geometry, shapely.LineString):
            self.is_geometry_of_type(geometry, shapely.LineString)
            super().__init__(shapely.MultiLineString([geometry]), crs)
        elif isinstance(geometry, GeoTrajectory):
            self.is_geometry_of_type(geometry, GeoTrajectory)
            super().__init__(shapely.MultiLineString([geometry.geometry]), crs)
        else:
            self.is_geometry_of_type(geometry, shapely.MultiLineString)
            super().__init__(geometry, crs)

    def plot(self, ax=None, add_points=False, color=None, linewidth=2, **kwargs):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        for line in self.geometry.geoms:
            shplt.plot_line(line, ax, add_points, color, linewidth, **kwargs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        # Convert the coordinates of each line in the MultiLineString
        converted_geoms = [
            [
                transformer.transform(x, y) for x, y in list(line.coords)
            ]  # Convert the coordinates of each line in the MultiLineString
            for line in self.geometry.geoms
        ]
        self.geometry = shapely.MultiLineString(converted_geoms)


class GeoPoint(GeoData):
    def __init__(self, geometry: shapely.Point, crs="WGS84"):
        self.is_geometry_of_type(geometry, shapely.Point)
        super().__init__(geometry, crs)

    def plot(self, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        shplt.plot_points(self.geometry, ax, add_points, color, linewidth, **kwargs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        x, y = transformer.transform(self.geometry.x, self.geometry.y)
        self.geometry = shapely.Point(x, y)


class GeoPolygon(GeoData):
    def __init__(self, geometry: shapely.Polygon | shapely.LineString, crs="WGS84"):
        self.is_geometry_of_type(geometry, shapely.Polygon | shapely.LineString)
        geometry = shapely.Polygon(geometry)
        super().__init__(geometry, crs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        # Convert each point in the polygon
        exterior = [
            transformer.transform(x, y) for x, y in self.geometry.exterior.coords
        ]
        interiors = [
            [transformer.transform(x, y) for x, y in interior.coords]
            for interior in self.geometry.interiors
        ]
        self.geometry = shapely.Polygon(exterior, interiors)

    def plot(
        self,
        ax=None,
        add_points=False,
        color=None,
        facecolor=None,
        edgecolor=None,
        linewidth=2,
        **kwargs,
    ):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        shplt.plot_polygon(
            polygon=self.geometry,
            ax=ax,
            add_points=add_points,
            color=color,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs,
        )


class GeoMultiPolygon(GeoData):
    def __init__(self, geometry, crs="WGS84"):
        if isinstance(geometry, list):
            for geom in geometry:
                self.is_geometry_of_type(geom, shapely.Polygon)
            geometry = shapely.MultiPolygon(geometry)
        else:
            self.is_geometry_of_type(geometry, shapely.MultiPolygon)
        super().__init__(geometry, crs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        polygon_list = []
        for polygon in list(self.geometry.geoms):
            # Convert each point in the polygon
            exterior = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
            interiors = [
                [transformer.transform(x, y) for x, y in interior.coords]
                for interior in polygon.interiors
            ]
            polygon_list.append(shapely.Polygon(exterior, interiors))

        self.geometry = shapely.MultiPolygon(polygon_list)

    def plot(
        self,
        ax=None,
        add_points=False,
        color=None,
        facecolor=None,
        edgecolor=None,
        linewidth=2,
        **kwargs,
    ):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        shplt.plot_polygon(
            polygon=self.geometry,
            ax=ax,
            add_points=add_points,
            color=color,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs,
        )
