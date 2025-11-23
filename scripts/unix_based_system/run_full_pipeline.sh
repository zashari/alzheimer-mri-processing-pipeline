#!/bin/bash
# Convenience script for running the complete pipeline end-to-end
# WARNING: This will run all stages sequentially and may take a very long time
# Ensure your machine has sufficient resources (RAM, disk space, GPU) before running

set -e  # Exit on error

echo "══════════════════════════════════════════════════════════════════════"
echo " Running Full Pipeline - All Stages"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "WARNING: This will run all stages sequentially."
echo "Estimated time: Several hours depending on dataset size and hardware."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

START_TIME=$(date +%s)

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Stage 1/4: Environment Setup"
echo "══════════════════════════════════════════════════════════════════════"
adp environment_setup setup --auto-install true --perf-test full

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Stage 2/4: Data Preparation"
echo "══════════════════════════════════════════════════════════════════════"
adp data_preparation split
adp data_preparation analyze

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Stage 3/4: NIfTI Processing"
echo "══════════════════════════════════════════════════════════════════════"
adp nifti_processing test --substage skull_stripping
adp nifti_processing process --substage skull_stripping
adp nifti_processing test --substage template_registration
adp nifti_processing process --substage template_registration
adp nifti_processing process --substage labelling
adp nifti_processing process --substage twoD_conversion

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Stage 4/4: Image Processing"
echo "══════════════════════════════════════════════════════════════════════"
adp image_processing process --substage center_crop
adp image_processing process --substage image_enhancement
adp image_processing process --substage data_balancing

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Full Pipeline Complete!"
echo "══════════════════════════════════════════════════════════════════════"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "══════════════════════════════════════════════════════════════════════"

