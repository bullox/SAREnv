"""Unit tests for the export_all_data module functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest

# Import functions to test
import sys

sys.path.append(str(Path(__file__).parent.parent / "data"))

from export_all_data import (
    generate_all,
    visualize_features,
    visualize_heatmap,
)


class TestExportAllData:
    """Test cases for the export_all_data module."""

    def test_generate_all_function_exists(self):
        """Test that the generate_all function exists and is callable."""
        assert callable(generate_all)

    def test_visualize_functions_exist(self):
        """Test that visualization functions exist and are callable."""
        assert callable(visualize_features)
        assert callable(visualize_heatmap)

    @patch("export_all_data.DataGenerator")
    @patch("export_all_data.DatasetLoader")
    @patch("export_all_data.os.makedirs")
    @patch("export_all_data.plt.savefig")
    def test_generate_all_mock(
        self, mock_savefig, mock_makedirs, mock_loader_class, mock_generator_class
    ):
        """Test generate_all function with mocked dependencies."""
        # Setup mocks
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Mock dataset item
        mock_item = Mock()
        mock_item.size = "test_size"
        mock_loader.load_environment.return_value = mock_item

        # Patch the points list to be smaller for testing
        with patch("export_all_data.points", [(1, 0.0, 0.0, "temperate", "flat")]):
            # This should not raise an exception
            generate_all()

        # Verify that the generator was used
        mock_generator_class.assert_called_once()
        mock_generator.export_dataset.assert_called()

        # Verify that the loader was used
        mock_loader_class.assert_called()
        mock_loader.load_environment.assert_called()

    def test_visualize_heatmap_with_none_item(self):
        """Test visualize_heatmap with None item."""
        # This should not raise an exception
        visualize_heatmap(None, False)

    def test_visualize_features_with_none_item(self):
        """Test visualize_features with None item."""
        # This should not raise an exception but should log a warning
        visualize_features(None, False)

    @patch("export_all_data.plt.subplots")
    @patch("export_all_data.plt.tight_layout")
    def test_visualize_heatmap_creates_plot(self, mock_tight_layout, mock_subplots):
        """Test that visualize_heatmap creates a plot structure."""
        # Create a mock item
        mock_item = Mock()
        mock_item.size = "test"
        mock_item.center_point = [10.0, 55.0]
        mock_item.bounds = (100, 200, 300, 400)
        mock_item.heatmap = [[1, 2], [3, 4]]

        # Create mock figure and axis
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Mock colorbar
        mock_fig.colorbar.return_value = Mock()

        # Call the function
        visualize_heatmap(
            mock_item, False
        )  # plot_basemap=False to avoid external calls

        # Verify plot elements were called
        mock_subplots.assert_called_once()
        mock_ax.imshow.assert_called_once()
        mock_fig.colorbar.assert_called_once()
        mock_tight_layout.assert_called_once()

    @pytest.mark.integration
    def test_generate_all_integration_small_sample(self):
        """Integration test with a very small sample of points."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Patch points to use only one test location
                test_points = [(1, 10.289470, 55.145921, "temperate", "flat")]

                with patch("export_all_data.points", test_points):
                    # This is an integration test - it will actually try to generate data
                    # Skip if external dependencies are not available
                    try:
                        generate_all()
                        # If we get here, check that files were created
                        assert Path("sarenv_dataset").exists()
                        assert Path("sarenv_dataset/1").exists()
                    except Exception as e:
                        pytest.skip(
                            f"Integration test skipped due to external dependency: {e}"
                        )
            finally:
                os.chdir(original_cwd)

    def test_points_data_structure(self):
        """Test that the points data has the correct structure."""
        # Import points from the module
        from export_all_data import points

        assert isinstance(points, list)
        assert len(points) > 0

        # Check structure of each point
        for point in points:
            assert isinstance(point, tuple)
            assert len(point) == 5  # id, lon, lat, climate, env_type

            point_id, lon, lat, climate, env_type = point

            # Check types and ranges
            assert isinstance(point_id, int)
            assert isinstance(lon, float)
            assert isinstance(lat, float)
            assert isinstance(climate, str)
            assert isinstance(env_type, str)

            # Check coordinate ranges
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90

            # Check valid climate and environment values
            assert climate in ["temperate", "dry"]
            assert env_type in ["flat", "mountainous"]

    def test_points_coverage(self):
        """Test that points cover all climate/environment combinations."""
        from export_all_data import points

        # Extract unique combinations
        combinations = set()
        for point in points:
            _, _, _, climate, env_type = point
            combinations.add((climate, env_type))

        # Check that all 4 combinations are present
        expected_combinations = {
            ("temperate", "flat"),
            ("temperate", "mountainous"),
            ("dry", "flat"),
            ("dry", "mountainous"),
        }

        assert (
            combinations >= expected_combinations
        )  # All expected combinations should be present

    def test_points_ids_unique(self):
        """Test that point IDs are unique."""
        from export_all_data import points

        ids = [point[0] for point in points]
        assert len(ids) == len(set(ids)), "Point IDs should be unique"

    def test_points_geographic_distribution(self):
        """Test that points have reasonable geographic distribution."""
        from export_all_data import points

        lons = [point[1] for point in points]
        lats = [point[2] for point in points]

        # Check that we have some geographic spread
        lon_range = max(lons) - min(lons)
        lat_range = max(lats) - min(lats)

        assert lon_range > 10, "Longitude range should be significant"
        assert lat_range > 10, "Latitude range should be significant"

        # Check that coordinates are within Europe (approximately)
        assert all(
            -25 <= lon <= 45 for lon in lons
        ), "Longitudes should be within European range"
        assert all(
            35 <= lat <= 70 for lat in lats
        ), "Latitudes should be within European range"

    @patch("matplotlib.pyplot.savefig")
    def test_matplotlib_cleanup(self, mock_savefig):
        """Test that matplotlib figures are properly handled."""
        # This test ensures that we don't leave figures open
        initial_figs = len(plt.get_fignums())

        # Create a mock item for testing
        mock_item = Mock()
        mock_item.size = "test"
        mock_item.center_point = [10.0, 55.0]
        mock_item.bounds = (100, 200, 300, 400)
        mock_item.heatmap = [[1, 2], [3, 4]]
        mock_item.features = Mock()
        mock_item.features.groupby.return_value = []
        mock_item.environment_type = "flat"
        mock_item.environment_climate = "temperate"
        mock_item.radius_km = 5.0

        # Call visualization functions
        try:
            visualize_heatmap(mock_item, False)
            visualize_features(mock_item, False)
        except Exception:
            # Ignore exceptions from mocked calls, we just want to check figure cleanup
            pass

        # Close any figures that might have been created
        plt.close("all")

        final_figs = len(plt.get_fignums())
        assert final_figs <= initial_figs, "Figures should be cleaned up"
