"""Integration tests for the complete SAR dataset workflow."""

import json
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from sarenv import (
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    DataGenerator,
    DatasetLoader,
)


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""

    def test_end_to_end_workflow(self, sample_coordinates):
        """Test the complete end-to-end workflow for one location."""
        center_point = sample_coordinates["glastonbury_uk"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Generate dataset
            data_gen = DataGenerator()
            data_gen.export_dataset(
                center_point=center_point,
                output_directory=str(temp_path),
                environment_climate=CLIMATE_TEMPERATE,
                environment_type=ENVIRONMENT_TYPE_FLAT,
                meter_per_bin=30,
            )
            
            # Step 2: Verify exported files
            assert (temp_path / "heatmap.npy").exists()
            assert (temp_path / "features.geojson").exists()
            assert (temp_path / "metadata.json").exists()
            
            # Step 3: Load and verify dataset
            loader = DatasetLoader(dataset_directory=str(temp_path))
            item = loader.load_environment()
            
            assert item is not None
            assert item.center_point is not None
            assert item.heatmap is not None
            assert item.features is not None
            
            # Step 4: Verify data integrity
            # Heatmap should be a 2D numpy array with positive values
            assert isinstance(item.heatmap, np.ndarray)
            assert item.heatmap.ndim == 2
            assert np.all(item.heatmap >= 0)
            
            # Features should be a valid GeoDataFrame
            assert isinstance(item.features, gpd.GeoDataFrame)
            assert len(item.features) > 0
            assert "feature_type" in item.features.columns
            
            # Metadata should match input parameters
            assert item.environment_climate == CLIMATE_TEMPERATE
            assert item.environment_type == ENVIRONMENT_TYPE_FLAT

    def test_workflow_with_different_parameters(self, test_location_params):
        """Test workflow with different climate and environment combinations."""
        for params in test_location_params[:2]:  # Test first 2 to keep it fast
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate dataset
                data_gen = DataGenerator()
                data_gen.export_dataset(
                    center_point=params["center_point"],
                    output_directory=str(temp_path),
                    environment_climate=params["climate"],
                    environment_type=params["environment"],
                    meter_per_bin=30,
                )
                
                # Load and verify
                loader = DatasetLoader(dataset_directory=str(temp_path))
                item = loader.load_environment()
                
                assert item.environment_climate == params["climate"]
                assert item.environment_type == params["environment"]

    def test_data_consistency_across_reload(self, sample_coordinates):
        """Test that data remains consistent when reloaded."""
        center_point = sample_coordinates["glastonbury_uk"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate dataset
            data_gen = DataGenerator()
            data_gen.export_dataset(
                center_point=center_point,
                output_directory=str(temp_path),
                environment_climate=CLIMATE_TEMPERATE,
                environment_type=ENVIRONMENT_TYPE_FLAT,
                meter_per_bin=30,
            )
            
            # Load dataset twice
            loader1 = DatasetLoader(dataset_directory=str(temp_path))
            item1 = loader1.load_environment()
            
            loader2 = DatasetLoader(dataset_directory=str(temp_path))
            item2 = loader2.load_environment()
            
            # Verify consistency
            assert item1.center_point == item2.center_point
            assert item1.environment_climate == item2.environment_climate
            assert item1.environment_type == item2.environment_type
            assert np.array_equal(item1.heatmap, item2.heatmap)
            assert len(item1.features) == len(item2.features)

    def test_export_and_load_multiple_locations(self, small_test_points):
        """Test exporting and loading multiple locations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            data_gen = DataGenerator()
            loaded_items = []
            
            # Export datasets for multiple locations
            for point_id, lon, lat, climate, env_type in small_test_points:
                location_dir = base_path / str(point_id)
                location_dir.mkdir()
                
                data_gen.export_dataset(
                    center_point=(lon, lat),
                    output_directory=str(location_dir),
                    environment_climate=climate,
                    environment_type=env_type,
                    meter_per_bin=30,
                )
                
                # Load the dataset
                loader = DatasetLoader(dataset_directory=str(location_dir))
                item = loader.load_environment()
                loaded_items.append((point_id, item))
            
            # Verify all items were loaded successfully
            assert len(loaded_items) == len(small_test_points)
            
            # Verify each item has valid data
            for point_id, item in loaded_items:
                assert item is not None
                assert isinstance(item.heatmap, np.ndarray)
                assert isinstance(item.features, gpd.GeoDataFrame)
                assert item.environment_climate is not None
                assert item.environment_type is not None

    @pytest.mark.slow
    def test_file_size_constraints(self, sample_coordinates):
        """Test that generated files are within reasonable size limits."""
        center_point = sample_coordinates["glastonbury_uk"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate dataset
            data_gen = DataGenerator()
            data_gen.export_dataset(
                center_point=center_point,
                output_directory=str(temp_path),
                environment_climate=CLIMATE_TEMPERATE,
                environment_type=ENVIRONMENT_TYPE_FLAT,
                meter_per_bin=30,
            )
            
            # Check file sizes (these are rough estimates)
            heatmap_size = (temp_path / "heatmap.npy").stat().st_size
            features_size = (temp_path / "features.geojson").stat().st_size
            metadata_size = (temp_path / "metadata.json").stat().st_size
            
            # Heatmap should be reasonable size (not empty, not too large)
            assert 1000 < heatmap_size < 50_000_000  # Between 1KB and 50MB
            
            # Features file should contain data
            assert features_size > 100  # At least 100 bytes
            
            # Metadata should be small JSON file
            assert 50 < metadata_size < 10000  # Between 50 bytes and 10KB

    def test_coordinate_bounds_validation(self):
        """Test that the system handles coordinate bounds correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_gen = DataGenerator()
            
            # Test with coordinates at various locations in Europe
            test_coordinates = [
                (10.0, 55.0),  # Denmark
                (-2.0, 51.0),  # UK
                (2.0, 46.0),   # France
                (12.0, 48.0),  # Germany/Austria border
            ]
            
            for lon, lat in test_coordinates:
                location_dir = temp_path / f"test_{lon}_{lat}"
                location_dir.mkdir()
                
                # This should work without errors
                data_gen.export_dataset(
                    center_point=(lon, lat),
                    output_directory=str(location_dir),
                    environment_climate=CLIMATE_TEMPERATE,
                    environment_type=ENVIRONMENT_TYPE_FLAT,
                    meter_per_bin=30,
                )
                
                # Verify files were created
                assert (location_dir / "heatmap.npy").exists()
                assert (location_dir / "features.geojson").exists()
                assert (location_dir / "metadata.json").exists()

    def test_metadata_completeness(self, sample_coordinates):
        """Test that metadata contains all expected fields."""
        center_point = sample_coordinates["glastonbury_uk"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate dataset
            data_gen = DataGenerator()
            data_gen.export_dataset(
                center_point=center_point,
                output_directory=str(temp_path),
                environment_climate=CLIMATE_TEMPERATE,
                environment_type=ENVIRONMENT_TYPE_FLAT,
                meter_per_bin=30,
            )
            
            # Load and check metadata
            metadata_path = temp_path / "metadata.json"
            with metadata_path.open() as f:
                metadata = json.load(f)
            
            # Check required fields
            required_fields = [
                "center_point",
                "environment_climate", 
                "environment_type",
            ]
            
            for field in required_fields:
                assert field in metadata, f"Metadata missing required field: {field}"
            
            # Check field types and values
            assert isinstance(metadata["center_point"], list)
            assert len(metadata["center_point"]) == 2
            assert isinstance(metadata["environment_climate"], str)
            assert isinstance(metadata["environment_type"], str)
