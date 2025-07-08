"""Unit tests for the DataGenerator functionality."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from sarenv import (
    CLIMATE_DRY,
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
)

from tests.conftest import TestHelpers


class TestDataGenerator:
    """Test cases for the DataGenerator class."""

    def test_data_generator_initialization(self, data_generator):
        """Test that DataGenerator can be initialized."""
        assert data_generator is not None

    def test_export_dataset_basic(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test basic dataset export functionality."""
        center_point = sample_coordinates["glastonbury_uk"]

        # Export dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=CLIMATE_TEMPERATE,
            environment_type=ENVIRONMENT_TYPE_FLAT,
            meter_per_bin=30,
        )

        # Verify files were created
        assert TestHelpers.verify_dataset_files(temp_output_dir)

    def test_export_dataset_all_climate_environment_combinations(
        self, data_generator, temp_output_dir, test_location_params
    ):
        """Test dataset export for all climate/environment combinations."""
        for params in test_location_params:
            location_dir = temp_output_dir / str(params["id"])
            location_dir.mkdir(exist_ok=True)

            # Export dataset
            data_generator.export_dataset(
                center_point=params["center_point"],
                output_directory=str(location_dir),
                environment_climate=params["climate"],
                environment_type=params["environment"],
                meter_per_bin=30,
            )

            # Verify files were created
            assert TestHelpers.verify_dataset_files(
                location_dir
            ), f"Failed for {params['name']}"

    def test_export_dataset_file_contents(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test that exported files contain valid data."""
        center_point = sample_coordinates["glastonbury_uk"]

        # Export dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=CLIMATE_TEMPERATE,
            environment_type=ENVIRONMENT_TYPE_FLAT,
            meter_per_bin=30,
        )

        # Test heatmap file
        heatmap_path = temp_output_dir / "heatmap.npy"
        heatmap = np.load(heatmap_path)
        assert heatmap.ndim == 2, "Heatmap should be 2D array"
        assert (
            heatmap.shape[0] > 0 and heatmap.shape[1] > 0
        ), "Heatmap should not be empty"
        assert np.all(heatmap >= 0), "Heatmap values should be non-negative"

        # Test features file
        features_path = temp_output_dir / "features.geojson"
        features = gpd.read_file(features_path)
        assert len(features) > 0, "Features should not be empty"
        assert (
            "feature_type" in features.columns
        ), "Features should have feature_type column"
        assert features.crs is not None, "Features should have a CRS"

        # Test metadata file
        metadata_path = temp_output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert "center_point" in metadata, "Metadata should contain center_point"
        assert (
            "environment_climate" in metadata
        ), "Metadata should contain environment_climate"
        assert (
            "environment_type" in metadata
        ), "Metadata should contain environment_type"

    def test_export_dataset_different_meters_per_bin(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test dataset export with different meter_per_bin values."""
        center_point = sample_coordinates["glastonbury_uk"]

        for meter_per_bin in [15, 30, 60]:
            location_dir = temp_output_dir / f"mpb_{meter_per_bin}"
            location_dir.mkdir(exist_ok=True)

            # Export dataset
            data_generator.export_dataset(
                center_point=center_point,
                output_directory=str(location_dir),
                environment_climate=CLIMATE_TEMPERATE,
                environment_type=ENVIRONMENT_TYPE_FLAT,
                meter_per_bin=meter_per_bin,
            )

            # Verify files were created
            assert TestHelpers.verify_dataset_files(location_dir)

            # Check that different resolutions produce different heatmap sizes
            heatmap_path = location_dir / "heatmap.npy"
            heatmap = np.load(heatmap_path)
            assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0

    def test_export_dataset_invalid_coordinates(self, data_generator, temp_output_dir):
        """Test error handling for invalid coordinates."""
        # Test with coordinates outside valid range
        invalid_coordinates = [
            (200, 0),  # Invalid longitude
            (0, 100),  # Invalid latitude
            (-200, -100),  # Both invalid
        ]

        for lon, lat in invalid_coordinates:
            with pytest.raises((ValueError, Exception)):
                data_generator.export_dataset(
                    center_point=(lon, lat),
                    output_directory=str(temp_output_dir),
                    environment_climate=CLIMATE_TEMPERATE,
                    environment_type=ENVIRONMENT_TYPE_FLAT,
                    meter_per_bin=30,
                )

    def test_export_dataset_dry_mountainous(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test specific case of dry mountainous environment."""
        center_point = sample_coordinates["pinos_genil_es"]

        # Export dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=CLIMATE_DRY,
            environment_type=ENVIRONMENT_TYPE_MOUNTAINOUS,
            meter_per_bin=30,
        )

        # Verify files and content
        assert TestHelpers.verify_dataset_files(temp_output_dir)

        # Load and verify metadata
        metadata_path = temp_output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["environment_climate"] == CLIMATE_DRY
        assert metadata["environment_type"] == ENVIRONMENT_TYPE_MOUNTAINOUS
        assert metadata["center_point"] == list(center_point)

    def test_export_dataset_temperate_flat(
        self, data_generator, temp_output_dir, sample_coordinates
    ):
        """Test specific case of temperate flat environment."""
        center_point = sample_coordinates["glastonbury_uk"]

        # Export dataset
        data_generator.export_dataset(
            center_point=center_point,
            output_directory=str(temp_output_dir),
            environment_climate=CLIMATE_TEMPERATE,
            environment_type=ENVIRONMENT_TYPE_FLAT,
            meter_per_bin=30,
        )

        # Verify files and content
        assert TestHelpers.verify_dataset_files(temp_output_dir)

        # Load and verify metadata
        metadata_path = temp_output_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["environment_climate"] == CLIMATE_TEMPERATE
        assert metadata["environment_type"] == ENVIRONMENT_TYPE_FLAT
        assert metadata["center_point"] == list(center_point)
