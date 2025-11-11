#!/bin/bash
# Convenience script for running data preparation stage
# This script runs both split and analyze actions sequentially

set -e  # Exit on error

echo "══════════════════════════════════════════════════════════════════════"
echo " Running Data Preparation Stage"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

echo "Step 1/2: Splitting data..."
python -m data_processing.cli data_preparation split

echo ""
echo "Step 2/2: Analyzing data..."
python -m data_processing.cli data_preparation analyze

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Data Preparation Complete"
echo "══════════════════════════════════════════════════════════════════════"

