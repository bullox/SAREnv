import os
import glob
import json
import pytest

# Directory where the exported datasets are stored
data_dir = os.path.join(os.path.dirname(__file__), '..', 'sarenv_dataset')

@pytest.mark.parametrize("geojson_path", glob.glob(os.path.join(data_dir, '*', '*.geojson')))
def test_geojson_features_have_area_probability(geojson_path):
    """
    Ensure every feature in every exported geojson has the 'area_probability' property.
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    assert 'features' in data, f"No 'features' key in {geojson_path}"
    for feature in data['features']:
        assert 'properties' in feature, f"No 'properties' in feature in {geojson_path}"
        assert 'area_probability' in feature['properties'], (
            f"Missing 'area_probability' in feature properties in {geojson_path}"
        )
        assert isinstance(feature['properties']['area_probability'], (float, int)), (
            f"'area_probability' is not a number in {geojson_path}"
        )

@pytest.mark.parametrize("json_path", glob.glob(os.path.join(data_dir, '*', '*.json')))
def test_json_metadata_has_expected_keys(json_path):
    """
    Check that each metadata JSON file has required top-level keys.
    """
    required_keys = {"center_point", "environment_climate", "environment_type", "radius_km"}
    with open(json_path, 'r') as f:
        data = json.load(f)
    missing = required_keys - set(data.keys())
    assert not missing, f"Missing keys {missing} in {json_path}"
