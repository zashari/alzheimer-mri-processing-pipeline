#!/bin/bash
# Convenience script for running NIfTI processing stage
# This script runs all NIfTI processing substages sequentially

set -e  # Exit on error

echo "══════════════════════════════════════════════════════════════════════"
echo " Running NIfTI Processing Stage"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

echo "Step 1/4: Skull Stripping (Test)..."
python -m data_processing.cli nifti_processing test --substage skull_stripping

echo ""
echo "Step 2/4: Skull Stripping (Process)..."
python -m data_processing.cli nifti_processing process --substage skull_stripping

echo ""
echo "Step 3/4: Template Registration (Test)..."
python -m data_processing.cli nifti_processing test --substage template_registration

echo ""
echo "Step 4/4: Template Registration (Process)..."
python -m data_processing.cli nifti_processing process --substage template_registration

echo ""
echo "Step 5/6: Labelling..."
python -m data_processing.cli nifti_processing process --substage labelling

echo ""
echo "Step 6/6: 2D Conversion..."
python -m data_processing.cli nifti_processing process --substage twoD_conversion

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " NIfTI Processing Complete"
echo "══════════════════════════════════════════════════════════════════════"

