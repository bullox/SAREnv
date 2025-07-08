"""Unit tests for the DatasetLoader functionality."""

import json

import geopandas as gpd
import numpy as np
import pytest
from sarenv import (
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    DatasetLoader,
)

from tests.conftest import TestHelpers


class TestDatasetLoader:
    """Test cases for the DatasetLoader class."""

    def test_dataset_loader_initialization(self, temp_output_dir):
        """Test that DatasetLoader can be initialized with a directory."""
        # Create a simple dataset first
        temp_output_dir.mkdir(exist_ok=True)
        loader = DatasetLoader(dataset_directory=str(temp_output_dir))
        assert loader is not None

    def test_load_environment_default(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test loading environment with default parameters."""
        center_point = sample_coordinates["glastonbury_uk"]

        # First create a dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=CLIMATE_TEMPERATE,
            environment_type=ENVIRONMENT_TYPE_FLAT,
            meter_per_bin=30,
        )

        # Load the dataset
        loader = DatasetLoader(dataset_directory=str(temp_output_dir))
        item = loader.load_environment()

        # Verify the loaded item
        assert item is not None
        assert hasattr(item, "center_point")
        assert hasattr(item, "heatmap")
        assert hasattr(item, "features")
        assert hasattr(item, "size")
        assert hasattr(item, "radius_km")
        assert hasattr(item, "environment_type")
        assert hasattr(item, "environment_climate")

    def test_loaded_item_properties(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test that loaded item has correct properties and data types."""
        center_point = sample_coordinates["glastonbury_uk"]

        # Create dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=CLIMATE_TEMPERATE,
            environment_type=ENVIRONMENT_TYPE_FLAT,
            meter_per_bin=30,
        )

        # Load dataset
        loader = DatasetLoader(dataset_directory=str(temp_output_dir))
        item = loader.load_environment()

        # Test center_point
        assert isinstance(item.center_point, (list, tuple))
        assert len(item.center_point) == 2
        assert isinstance(item.center_point[0], (int, float))
        assert isinstance(item.center_point[1], (int, float))

        # Test heatmap
        assert isinstance(item.heatmap, np.ndarray)
        assert item.heatmap.ndim == 2
        assert item.heatmap.shape[0] > 0
        assert item.heatmap.shape[1] > 0
        assert np.all(item.heatmap >= 0)

        # Test features
        assert isinstance(item.features, gpd.GeoDataFrame)
        assert len(item.features) > 0
        assert "feature_type" in item.features.columns
        assert item.features.crs is not None

        # Test other properties
        assert isinstance(item.size, str)
        assert isinstance(item.radius_km, (int, float))
        assert item.radius_km > 0
        assert isinstance(item.environment_type, str)
        assert isinstance(item.environment_climate, str)

    def test_load_environment_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            loader = DatasetLoader(dataset_directory="/nonexistent/path")
            loader.load_environment()

    def test_load_environment_empty_directory(self, temp_output_dir):
        """Test error handling for empty directory."""
        # Create empty directory
        temp_output_dir.mkdir(exist_ok=True)

        loader = DatasetLoader(dataset_directory=str(temp_output_dir))
        with pytest.raises(FileNotFoundError):
            loader.load_environment()

    def test_load_environment_missing_files(self, temp_output_dir):
        """Test error handling when dataset files are missing."""
        temp_output_dir.mkdir(exist_ok=True)

        # Create only one file (missing others)
        (temp_output_dir / "metadata.json").write_text("{}")

        loader = DatasetLoader(dataset_directory=str(temp_output_dir))
        with pytest.raises(FileNotFoundError):
            loader.load_environment()

    def test_multiple_dataset_loads(
        self, data_generator, temp_output_dir, test_location_params
    ):
        """Test loading multiple datasets from different locations."""
        for params in test_location_params[:2]:  # Test first 2 locations
            location_dir = temp_output_dir / str(params["id"])
            location_dir.mkdir(exist_ok=True)

            # Create dataset
            data_generator.export_dataset(
                center_point=params["center_point"],
                output_directory=str(location_dir),
                environment_climate=params["climate"],
                environment_type=params["environment"],
                meter_per_bin=30,
            )

            # Load and verify dataset
            loader = DatasetLoader(dataset_directory=str(location_dir))
            item = loader.load_environment()

            assert item is not None
            assert item.environment_climate == params["climate"]
            assert item.environment_type == params["environment"]
            # Center point might be slightly different due to processing
            assert abs(item.center_point[0] - params["center_point"][0]) < 0.1
            assert abs(item.center_point[1] - params["center_point"][1]) < 0.1

    def test_dataset_bounds_property(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test that loaded dataset has valid bounds."""
        center_point = sample_coordinates["glastonbury_uk"]

        # Create dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=CLIMATE_TEMPERATE,
            environment_type=ENVIRONMENT_TYPE_FLAT,
            meter_per_bin=30,
        )

        # Load dataset
        loader = DatasetLoader(dataset_directory=str(temp_output_dir))
        item = loader.load_environment()

        # Test bounds
        assert hasattr(item, "bounds")
        bounds = item.bounds
        assert len(bounds) == 4  # minx, miny, maxx, maxy
        minx, miny, maxx, maxy = bounds
        assert minx < maxx
        assert miny < maxy
        assert all(isinstance(coord, (int, float)) for coord in bounds)

    def test_dataset_metadata_consistency(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test that loaded dataset metadata matches the original export parameters."""
        center_point = sample_coordinates["glastonbury_uk"]
        expected_climate = CLIMATE_TEMPERATE
        expected_environment = ENVIRONMENT_TYPE_FLAT

        # Create dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=expected_climate,
            environment_type=expected_environment,
            meter_per_bin=30,
        )

        # Load dataset
        loader = DatasetLoader(dataset_directory=str(temp_output_dir))
        item = loader.load_environment()

        # Verify metadata consistency
        assert item.environment_climate == expected_climate
        assert item.environment_type == expected_environment

        # Also check the raw metadata file
        metadata_path = temp_output_dir / "metadata.json"
        with metadata_path.open() as f:
            metadata = json.load(f)

        assert metadata["environment_climate"] == expected_climate
        assert metadata["environment_type"] == expected_environment
        assert len(metadata["center_point"]) == 2
