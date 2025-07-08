# Tests for SAR Dataset Project

This directory contains comprehensive unit and integration tests for the SAR dataset export functionality.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Test configuration and fixtures
├── test_data_generator.py      # Tests for DataGenerator class
├── test_dataset_loader.py      # Tests for DatasetLoader class
├── test_export_all_data.py     # Tests for export_all_data module
├── test_integration.py         # Integration tests for complete workflow
└── test_visualization.py       # Tests for visualization functions
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov
```

### Quick Start

```bash
# Run all unit tests (fastest)
python -m pytest tests/ -m "not integration and not slow"

# Run all tests including integration tests
python -m pytest tests/

# Run with coverage reporting
python -m pytest tests/ --cov=sarenv --cov-report=html
```

### Using the Test Runner

Use the provided test runner script:

```bash
# Run unit tests only (fast)
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run all tests
python run_tests.py all

# Run with coverage
python run_tests.py unit --cov
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocking for external dependencies
- Fast execution (< 30 seconds)
- Run on every commit

**Files:** `test_data_generator.py`, `test_dataset_loader.py`, `test_visualization.py`

### Integration Tests
- Test complete workflows end-to-end
- Use real external APIs and data
- Slower execution (may take several minutes)
- Marked with `@pytest.mark.integration`

**Files:** `test_integration.py`, parts of `test_export_all_data.py`

### Slow Tests
- Tests that take a long time to run
- Large dataset processing
- Marked with `@pytest.mark.slow`

## Test Features

### Fixtures (conftest.py)
- `temp_output_dir`: Temporary directory for test outputs
- `data_generator`: DataGenerator instance
- `sample_coordinates`: Sample test coordinates
- `test_location_params`: Test parameters for different locations
- `small_test_points`: Subset of points for faster testing

### Test Helpers
- `TestHelpers.verify_dataset_files()`: Verify required dataset files exist
- `TestHelpers.verify_visualization_files()`: Verify visualization files exist

### Mocking
Tests use `unittest.mock` to:
- Mock external API calls
- Mock file system operations
- Mock matplotlib plotting functions
- Test error conditions

## Coverage

Run coverage analysis:
```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=sarenv --cov-report=html

# View coverage in terminal
python -m pytest tests/ --cov=sarenv --cov-report=term

# Generate XML coverage for CI
python -m pytest tests/ --cov=sarenv --cov-report=xml
```

Coverage reports are generated in:
- `htmlcov/index.html` (HTML report)
- `coverage.xml` (XML report for CI)

## Test Data

### Sample Coordinates
Tests use real European coordinates covering different climate and environment combinations:

- **Temperate, Flat**: Glastonbury UK, Suserup DK
- **Temperate, Mountainous**: Davos CH
- **Dry, Flat**: Valfarta ES
- **Dry, Mountainous**: Pinos Genil ES

### Test Points Structure
Each test point includes:
- ID (integer)
- Longitude (float)
- Latitude (float) 
- Climate ("temperate" or "dry")
- Environment type ("flat" or "mountainous")

## Continuous Integration

The GitHub workflow (`.github/workflows/export-dataset.yml`) includes test execution:

```yaml
- name: Run tests
  run: |
    pytest tests/ -v --tb=short -m "not integration and not slow"
    pytest tests/ --cov=sarenv --cov-report=xml -m "not integration and not slow"
```

Only fast unit tests run in CI to keep build times reasonable.

## Adding New Tests

### Test File Naming
- `test_*.py` for test files
- `Test*` for test classes
- `test_*` for test methods

### Test Categories
Mark tests appropriately:
```python
@pytest.mark.integration
def test_complete_workflow():
    """Integration test."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Slow test."""
    pass
```

### Using Fixtures
```python
def test_with_fixture(temp_output_dir, data_generator):
    """Test using fixtures."""
    # temp_output_dir and data_generator are automatically provided
    pass
```

### Mocking External Dependencies
```python
@patch("module.external_function")
def test_with_mock(mock_external):
    """Test with mocked dependency."""
    mock_external.return_value = "test_value"
    # Your test code here
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project is installed in development mode:
   ```bash
   pip install -e .
   ```

2. **Missing Dependencies**: Install test dependencies:
   ```bash
   pip install pytest pytest-cov
   ```

3. **Integration Test Failures**: Integration tests require internet access and external APIs. They may fail due to:
   - Network connectivity issues
   - API rate limiting
   - External service downtime

4. **Temporary Directory Issues**: Tests use temporary directories that are automatically cleaned up. If you see permission errors, check that the test process has write access.

### Debugging Tests

Run with verbose output:
```bash
python -m pytest tests/ -v --tb=long
```

Run a single test:
```bash
python -m pytest tests/test_data_generator.py::TestDataGenerator::test_export_dataset_basic -v
```

Debug with pdb:
```bash
python -m pytest tests/ --pdb
```

## Performance

### Test Execution Times
- Unit tests: ~10-30 seconds
- Integration tests: ~2-5 minutes per location
- Full test suite: ~5-10 minutes

### Optimization
- Use `small_test_points` fixture for faster testing
- Mock external dependencies in unit tests
- Use `@pytest.mark.slow` for long-running tests
- Parallel execution: `pytest -n auto` (requires pytest-xdist)

## Best Practices

1. **Keep unit tests fast** - Mock external dependencies
2. **Test edge cases** - Invalid inputs, error conditions
3. **Use descriptive test names** - Clearly describe what is being tested
4. **Test one thing at a time** - Single responsibility per test
5. **Clean up resources** - Use fixtures and context managers
6. **Document complex tests** - Add docstrings explaining test purpose
