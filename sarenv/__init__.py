# Import key modules and classes for public use
from . import Geometries, Query, Utils
from .Environment import Environment, EnvironmentBuilder

# Define the public API of the package
__all__ = [
    "Environment",
    "EnvironmentBuilder",
    "Geometries",
    "Query",
    "Utils",
]
