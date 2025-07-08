"""Unit tests for visualization functionality."""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest

from tests.conftest import TestHelpers


class TestVisualization:
    """Test cases for visualization functions."""

    def test_test_helpers_verify_dataset_files(self, temp_output_dir):
        """Test the TestHelpers.verify_dataset_files method."""
        # Initially should return False (no files)
        assert not TestHelpers.verify_dataset_files(temp_output_dir)

        # Create required files
        (temp_output_dir / "heatmap.npy").touch()
        (temp_output_dir / "features.geojson").touch()
        (temp_output_dir / "metadata.json").touch()

        # Now should return True
        assert TestHelpers.verify_dataset_files(temp_output_dir)

    def test_test_helpers_verify_visualization_files(self, temp_output_dir):
        """Test the TestHelpers.verify_visualization_files method."""
        item_size = "test_size"

        # Initially should return False
        assert not TestHelpers.verify_visualization_files(temp_output_dir, item_size)

        # Create visualization files
        (temp_output_dir / f"features_{item_size}.png").touch()
        (temp_output_dir / f"heatmap_{item_size}.png").touch()

        # Now should return True
        assert TestHelpers.verify_visualization_files(temp_output_dir, item_size)

    def test_matplotlib_figure_management(self):
        """Test that matplotlib figures are properly managed."""
        initial_fig_count = len(plt.get_fignums())

        # Create a figure
        fig, ax = plt.subplots()
        assert len(plt.get_fignums()) == initial_fig_count + 1

        # Close the figure
        plt.close(fig)
        assert len(plt.get_fignums()) == initial_fig_count

    def test_close_all_figures(self):
        """Test plt.close('all') functionality."""
        # Create multiple figures
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()
        fig3, _ = plt.subplots()

        assert len(plt.get_fignums()) >= 3

        # Close all figures
        plt.close("all")
        assert len(plt.get_fignums()) == 0

    @patch("matplotlib.pyplot.subplots")
    def test_mock_matplotlib_usage(self, mock_subplots):
        """Test using mocked matplotlib components."""
        # Setup mock
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Use matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([1, 2, 3], [1, 4, 2])

        # Verify mock was called
        mock_subplots.assert_called_once_with(figsize=(10, 8))
        mock_ax.plot.assert_called_once_with([1, 2, 3], [1, 4, 2])

    def test_figure_size_configuration(self):
        """Test that figures can be created with specific sizes."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Check figure size (allowing for small floating point differences)
        size = fig.get_size_inches()
        assert abs(size[0] - 12) < 0.1
        assert abs(size[1] - 10) < 0.1

        plt.close(fig)

    def test_colormap_availability(self):
        """Test that required colormaps are available."""
        import matplotlib.cm as cm

        # Test colormaps used in the visualization functions
        required_colormaps = ["inferno", "YlOrRd"]

        for cmap_name in required_colormaps:
            cmap = cm.get_cmap(cmap_name)
            assert cmap is not None, f"Colormap {cmap_name} should be available"

    def test_legend_components(self):
        """Test that matplotlib legend components work."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        # Test creating legend elements
        patch = Patch(color="red", label="Test Patch")
        line = Line2D([0], [0], color="blue", lw=2, label="Test Line")

        assert patch.get_label() == "Test Patch"
        assert line.get_label() == "Test Line"
        assert patch.get_facecolor() is not None
        assert line.get_color() == "blue"

    @pytest.mark.parametrize("file_format", ["png", "jpg", "pdf", "svg"])
    def test_supported_output_formats(self, file_format, temp_output_dir):
        """Test that matplotlib supports various output formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        output_path = temp_output_dir / f"test_plot.{file_format}"

        try:
            fig.savefig(output_path, format=file_format)
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except Exception as e:
            pytest.skip(f"Format {file_format} not supported: {e}")
        finally:
            plt.close(fig)

    def test_axis_labeling(self):
        """Test that axis labeling works correctly."""
        fig, ax = plt.subplots()

        ax.set_xlabel("Test X Label")
        ax.set_ylabel("Test Y Label")
        ax.set_title("Test Title")

        assert ax.get_xlabel() == "Test X Label"
        assert ax.get_ylabel() == "Test Y Label"
        assert ax.get_title() == "Test Title"

        plt.close(fig)

    def test_tight_layout(self):
        """Test that tight_layout works without errors."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("Test plot")

        # This should not raise an exception
        plt.tight_layout()

        plt.close(fig)

    def test_colorbar_functionality(self):
        """Test that colorbar creation works."""
        import numpy as np

        fig, ax = plt.subplots()

        # Create a simple image
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap="viridis")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Test Colorbar")

        assert cbar is not None

        plt.close(fig)

    def test_multiple_subplots(self):
        """Test creating multiple subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        assert axes.shape == (2, 2)

        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                ax.plot([1, 2, 3], [i + 1, j + 2, i + j + 3])
                ax.set_title(f"Subplot {i},{j}")

        plt.tight_layout()
        plt.close(fig)
