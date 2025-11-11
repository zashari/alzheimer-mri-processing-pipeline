# Convenience Scripts

This directory contains convenience wrapper scripts for running the Alzheimer MRI Processing Pipeline stages. These scripts simplify execution for non-technical users by wrapping the Python CLI commands.

## Directory Structure

- **`unix_based_system/`** - Shell scripts (`.sh`) for Linux and macOS
- **`windows_based_system/`** - Batch scripts (`.bat`) for Windows

## Available Scripts

### Individual Stage Scripts

Run these scripts if you want to execute stages one-by-one and monitor progress:

#### Unix/Linux/macOS:
```bash
./scripts/unix_based_system/run_environment_setup.sh
./scripts/unix_based_system/run_data_preparation.sh
./scripts/unix_based_system/run_nifti_processing.sh
./scripts/unix_based_system/run_image_processing.sh
```

#### Windows:
```cmd
scripts\windows_based_system\run_environment_setup.bat
scripts\windows_based_system\run_data_preparation.bat
scripts\windows_based_system\run_nifti_processing.bat
scripts\windows_based_system\run_image_processing.bat
```

### Full Pipeline Scripts

Run these scripts if you want to execute the entire pipeline end-to-end in one go. **Only use these if you have sufficient machine resources and can leave the process running for several hours.**

#### Unix/Linux/macOS:
```bash
./scripts/unix_based_system/run_full_pipeline.sh
```

#### Windows:
```cmd
scripts\windows_based_system\run_full_pipeline.bat
```

## What Each Script Does

### `run_environment_setup.sh` / `run_environment_setup.bat`
- Runs environment setup with GPU detection
- Auto-installs PyTorch with appropriate CUDA version
- Performs full performance test

### `run_data_preparation.sh` / `run_data_preparation.bat`
- Runs data splitting (train/val/test)
- Runs data analysis
- Generates manifest files

### `run_nifti_processing.sh` / `run_nifti_processing.bat`
Runs all NIfTI processing substages sequentially:
1. Skull Stripping (test)
2. Skull Stripping (process)
3. Template Registration (test)
4. Template Registration (process)
5. Labelling
6. 2D Conversion

### `run_image_processing.sh` / `run_image_processing.bat`
Runs all image processing substages sequentially:
1. Center Crop
2. Image Enhancement
3. Data Balancing

### `run_full_pipeline.sh` / `run_full_pipeline.bat`
Runs all stages in sequence:
1. Environment Setup
2. Data Preparation
3. NIfTI Processing (all substages)
4. Image Processing (all substages)

**Warning**: This will take several hours and requires significant computational resources.

## Prerequisites

Before running any script:

1. **Python 3.10+** must be installed
2. **Dependencies** should be installed (`pip install -r requirements.txt`)
3. **Configuration** files should be set up (see `configs/default.yaml`)
4. **Template files** should be downloaded (see `support_files/templates/*/README.md`)

## Usage Tips

### For Non-Technical Users
- Start with individual stage scripts
- Run stages one at a time
- Check outputs between stages
- Monitor system resources (RAM, disk space)

### For Advanced Users
- Use full pipeline script if you have powerful hardware
- Can leave running unattended
- Monitor logs in `.reports/` directory

## Troubleshooting

### Scripts won't execute (Unix/Linux/macOS)
Make scripts executable:
```bash
chmod +x scripts/unix_based_system/*.sh
```

### Python not found
Ensure Python is in your PATH:
```bash
# Check Python version
python --version

# Or try python3
python3 --version
```

### Script fails with error
- Check that you're in the project root directory
- Verify configuration files are set up correctly
- Check that required template files are downloaded
- Review error messages in the terminal output

## Alternative: Direct Python CLI

If you prefer more control or need to customize parameters, you can run the Python CLI directly:

```bash
python -m data_processing.cli <stage> <action> [options]
```

See the main README.md for detailed CLI usage examples.

