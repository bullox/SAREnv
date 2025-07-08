#!/bin/bash

# Local test script for the SAR dataset export workflow
# This script simulates the GitHub Actions workflow locally

set -e

echo "=== SAR Dataset Export - Local Test ==="
echo

# Check if required tools are available
command -v python >/dev/null 2>&1 || { echo "Python is required but not installed. Aborting." >&2; exit 1; }

# Set default size if not provided
SIZE_TO_LOAD=${SIZE_TO_LOAD:-"small"}
echo "Dataset size: $SIZE_TO_LOAD"

# Create timestamp
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
echo "Timestamp: $TIMESTAMP"

# Check if we're in the right directory
if [ ! -f "data/export_all_data.py" ]; then
    echo "Error: export_all_data.py not found. Please run this script from the repository root."
    exit 1
fi

echo
echo "=== Installing Dependencies ==="
pip install -r requirements.txt
pip install -e .

echo
echo "=== Running Dataset Export ==="
cd data
export SIZE_TO_LOAD
python export_all_data.py
cd ..

echo
echo "=== Creating Archives ==="
if [ -d "sarenv_dataset" ]; then
    # Create archives
    tar -czf "sar-dataset-${TIMESTAMP}.tar.gz" \
        --exclude='*.git*' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        sarenv_dataset/
    
    zip -r "sar-dataset-${TIMESTAMP}.zip" \
        sarenv_dataset/ \
        -x "*.git*" "*__pycache__*" "*.pyc"
    
    # Create checksums
    sha256sum "sar-dataset-${TIMESTAMP}.tar.gz" > checksums.txt
    sha256sum "sar-dataset-${TIMESTAMP}.zip" >> checksums.txt
    
    echo "Archives created:"
    ls -lh sar-dataset-${TIMESTAMP}.*
    echo
    echo "Dataset statistics:"
    echo "- Size: $SIZE_TO_LOAD"
    echo "- Locations: $(find sarenv_dataset -name "*.json" | wc -l)"
    echo "- Total files: $(find sarenv_dataset -type f | wc -l)"
    echo "- Archive size (tar.gz): $(du -h sar-dataset-${TIMESTAMP}.tar.gz | cut -f1)"
    echo "- Archive size (zip): $(du -h sar-dataset-${TIMESTAMP}.zip | cut -f1)"
    echo
    echo "Checksums:"
    cat checksums.txt
else
    echo "Error: sarenv_dataset directory not found. Export may have failed."
    exit 1
fi

echo
echo "=== Local Test Complete ==="
echo "Files created:"
echo "- sar-dataset-${TIMESTAMP}.tar.gz"
echo "- sar-dataset-${TIMESTAMP}.zip"
echo "- checksums.txt"
echo "- sarenv_dataset/ (directory)"
