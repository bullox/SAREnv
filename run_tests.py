#!/usr/bin/env python3
"""Test runner script for the SAR dataset project."""

import subprocess
import sys
from pathlib import Path


def run_tests(test_type="unit"):
    """Run tests with different configurations."""
    project_root = Path(__file__).parent
    
    print(f"Running {test_type} tests...")
    print(f"Project root: {project_root}")
    
    # Base pytest command
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    
    if test_type == "unit":
        # Run only unit tests (exclude integration and slow tests)
        cmd.extend(["-m", "not integration and not slow"])
        print("Running unit tests only")
    elif test_type == "integration":
        # Run integration tests
        cmd.extend(["-m", "integration"])
        print("Running integration tests")
    elif test_type == "all":
        # Run all tests
        print("Running all tests")
    elif test_type == "fast":
        # Run fast tests only
        cmd.extend(["-m", "not slow"])
        print("Running fast tests only")
    
    # Add coverage if requested
    if "--cov" in sys.argv:
        cmd.extend(["--cov=sarenv", "--cov-report=term", "--cov-report=html"])
        print("Coverage reporting enabled")
    
    # Add verbose output
    if "--verbose" in sys.argv or "-v" in sys.argv:
        cmd.append("--tb=long")
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [unit|integration|all|fast] [--cov] [--verbose]")
        print("Examples:")
        print("  python run_tests.py unit           # Run unit tests only")
        print("  python run_tests.py integration    # Run integration tests only")
        print("  python run_tests.py all            # Run all tests")
        print("  python run_tests.py fast           # Run fast tests (exclude slow)")
        print("  python run_tests.py unit --cov     # Run unit tests with coverage")
        return 1
    
    test_type = sys.argv[1]
    
    if test_type not in ["unit", "integration", "all", "fast"]:
        print(f"Invalid test type: {test_type}")
        print("Valid options: unit, integration, all, fast")
        return 1
    
    return run_tests(test_type)


if __name__ == "__main__":
    sys.exit(main())
