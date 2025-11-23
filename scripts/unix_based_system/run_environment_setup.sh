#!/bin/bash
# Convenience script for running environment setup stage
# This script wraps the Python CLI command for easier execution

set -e  # Exit on error

echo "══════════════════════════════════════════════════════════════════════"
echo " Running Environment Setup Stage"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

adp environment_setup setup --auto-install true --perf-test full

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Environment Setup Complete"
echo "══════════════════════════════════════════════════════════════════════"

