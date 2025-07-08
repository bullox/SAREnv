"""Test configuration and fixtures for the SAR dataset tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sarenv import (
    CLIMATE_DRY,
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
    DataGenerator,
)


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def data_generator() -> DataGenerator:
    """Create a DataGenerator instance for testing."""
    return DataGenerator()


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing."""
    return {
        "glastonbury_uk": (-2.704825, 51.117314),  # Temperate, Flat
        "svanninge_bakker_dk": (10.289470, 55.145921),  # Temperate, Flat
        "davos_ch": (9.838304, 46.826512),  # Temperate, Mountainous
        "valfarta_es": (-0.128459, 41.522016),  # Dry, Flat
        "pinos_genil_es": (-3.453313, 37.140462),  # Dry, Mountainous
    }


@pytest.fixture
def test_location_params():
    """Parameters for test locations covering all climate/environment combinations."""
    return [
        {
            "id": 1,
            "center_point": (-2.704825, 51.117314),
            "climate": CLIMATE_TEMPERATE,
            "environment": ENVIRONMENT_TYPE_FLAT,
            "name": "Glastonbury, UK"
        },
        {
            "id": 2,
            "center_point": (9.838304, 46.826512),
            "climate": CLIMATE_TEMPERATE,
            "environment": ENVIRONMENT_TYPE_MOUNTAINOUS,
            "name": "Davos, CH"
        },
        {
            "id": 3,
            "center_point": (-0.128459, 41.522016),
            "climate": CLIMATE_DRY,
            "environment": ENVIRONMENT_TYPE_FLAT,
            "name": "Valfarta, ES"
        },
        {
            "id": 4,
            "center_point": (-3.453313, 37.140462),
            "climate": CLIMATE_DRY,
            "environment": ENVIRONMENT_TYPE_MOUNTAINOUS,
            "name": "Pinos Genil, ES"
        }
    ]


@pytest.fixture
def small_test_points():
    """A small subset of points for faster testing."""
    return [
        (1, -2.704825, 51.117314, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
        (2, 10.289470, 55.145921, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT),
        (3, 9.838304, 46.826512, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS),
    ]


class TestHelpers:
    """Helper methods for tests."""

    @staticmethod
    def verify_dataset_files(output_dir: Path) -> bool:
        """Verify that the expected dataset files exist."""
        required_files = [
            "heatmap.npy",
            "features.geojson",
            "metadata.json"
        ]

        for file_name in required_files:
            file_path = output_dir / file_name
            if not file_path.exists():
                return False
        return True

    @staticmethod
    def verify_visualization_files(output_dir: Path, item_size: str) -> bool:
        """Verify that visualization files exist."""
        viz_files = [
            f"features_{item_size}.png",
            f"heatmap_{item_size}.png"
        ]

        for file_name in viz_files:
            file_path = output_dir / file_name
            if not file_path.exists():
                return False
        return True
