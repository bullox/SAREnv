# sarenv/__init__.py
"""
SARenv: A Python toolkit for generating, loading, and evaluating
Search and Rescue environment data.
"""
from .analytics.evaluator import ComparativeEvaluator
from .core.generation import DataGenerator
from .core.loading import DatasetLoader, SARDatasetItem
from .core.lost_person import LostPersonLocationGenerator
from .utils.logging_setup import get_logger
from .utils.lost_person_behavior import (
    FEATURE_PROBABILITIES,
    CLIMATE_TEMPERATE,
    CLIMATE_DRY,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
)
from .utils.plot import visualize_heatmap, visualize_features

__all__ = [
    "ComparativeEvaluator",
    "DataGenerator",
    "DatasetLoader",
    "SARDatasetItem",
    "LostPersonLocationGenerator",
    "get_logger",
    "ENVIRONMENT_TYPE_FLAT",
    "ENVIRONMENT_TYPE_MOUNTAINOUS",
    "CLIMATE_DRY",
    "CLIMATE_TEMPERATE",
    "FEATURE_PROBABILITIES",
    "visualize_heatmap",
    "visualize_features",
]
