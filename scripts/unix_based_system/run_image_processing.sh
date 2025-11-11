#!/bin/bash
# Convenience script for running image processing stage
# This script runs all image processing substages sequentially

set -e  # Exit on error

echo "══════════════════════════════════════════════════════════════════════"
echo " Running Image Processing Stage"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

echo "Step 1/3: Center Crop..."
python -m data_processing.cli image_processing process --substage center_crop

echo ""
echo "Step 2/3: Image Enhancement..."
python -m data_processing.cli image_processing process --substage image_enhancement

echo ""
echo "Step 3/3: Data Balancing..."
python -m data_processing.cli image_processing process --substage data_balancing

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Image Processing Complete"
echo "══════════════════════════════════════════════════════════════════════"

