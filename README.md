[![License](https://img.shields.io/badge/license-MIT-informational)](./LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.1-blue)](./docs/CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blue)](./CITATION.cff)
[![Status](https://img.shields.io/badge/status-stable-success)](./docs/CHANGELOG.md)

# Alzheimer MRI Processing Pipeline

A **complete, production-ready 3D NIfTI preprocessing pipeline** for ADNI T1-weighted MRI data, designed to accelerate Alzheimer's disease research by automating the entire preprocessing workflow from raw NIfTI files to training-ready 2D image sequences.

## Table of Contents

- [ADNI/IDA Compliance Notice](#adniida-compliance-notice)
- [Overview](#overview)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Convenience Scripts](#convenience-scripts)
  - [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## ADNI/IDA Compliance Notice

> **Project status:** Stable release. Current version: **v1.1.1**.

> **This project is an independent, open-source effort. It is not affiliated with, endorsed by, or sponsored by the Alzheimer's Disease Neuroimaging Initiative ([ADNI](https://adni.loni.usc.edu/)) or the Imaging Data Archive ([IDA](https://ida.loni.usc.edu/)) operated by the Laboratory of Neuro Imaging ([LONI](https://loni.usc.edu/)). "ADNI" and "IDA-LONI" are trademarks of their respective owners and are used solely to indicate data compatibility.**
>
> **This repository contains code only. It does not host or distribute ADNI data. Access to ADNI/IDA is governed by their Data Use Agreements (DUAs). Users are solely responsible for compliance with all applicable terms.**
>
> **Before requesting data access, read the [ADNI Data Use Agreement](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp) and then follow the [ADNI data access instructions](https://adni.loni.usc.edu/data-samples/adni-data/#AccessData).**

### Scope of This Repository

**Included:**
- ✅ Processing framework and utilities (code only)
- ✅ Configuration templates for experiments
- ✅ Documentation and examples

**Explicitly Excluded:**
- ❌ ADNI participant-level data
- ❌ ADNI-derived datasets or processed outputs
- ❌ Participant identifiers or metadata

### Requirements for ADNI Data Users

If you use this pipeline with **ADNI data**, you must comply with the ADNI DUA: <https://ida.loni.usc.edu/collaboration/access/appLicense.jsp>

### Resources

- **ADNI Website:** <https://adni.loni.usc.edu/>
- **ADNI Data Use Agreement:** <https://ida.loni.usc.edu/collaboration/access/appLicense.jsp>
- **IDA-LONI Access Portal:** <https://ida.loni.usc.edu/>
- **ADNI Publication & Citation Guidelines:** <https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Manuscript_Citations.pdf>

**By using this pipeline with ADNI data, you confirm that you have read, understood, and agree to comply with all terms of the ADNI Data Use Agreement.**

---

## Overview

This repository provides a **complete 3D NIfTI preprocessing pipeline** for **ADNI T1-weighted MRI** data, transforming raw medical imaging files into training-ready 2D image sequences optimized for temporal deep learning models (e.g., CNN+LSTM architectures).

The pipeline was developed in **early 2025** as part of a final-year thesis project, with the goal of reducing time spent on data wrangling so researchers can focus on modeling and experimentation.

### Key Benefits

- **End-to-End Automation:** Complete preprocessing workflow from raw NIfTI to training-ready images
- **Modular Design:** Each stage can be run independently or as part of the full pipeline
- **Resume Capability:** Long-running processes can be resumed from checkpoints
- **Cross-Platform:** Windows (primary) and Unix-based systems (Linux/macOS)
- **Production-Ready:** Comprehensive error handling, logging, and progress tracking
- **GPU Acceleration:** Optimized for CUDA-enabled GPUs with automatic fallback to CPU

---

## Features

- ✅ **Complete 3D NIfTI Processing Pipeline** - From raw files to training-ready 2D sequences
- ✅ **Modular Stage Architecture** - Run stages independently or end-to-end
- ✅ **GPU Acceleration** - CUDA support with automatic CPU fallback
- ✅ **Resume Capability** - Checkpoint-based resumption for long-running processes
- ✅ **Rich Console Output** - Beautiful, informative progress indicators and summaries
- ✅ **JSON Reports** - Detailed execution reports for each stage
- ✅ **Configuration Management** - YAML-based configuration with CLI overrides
- ✅ **Convenience Scripts** - Pre-built scripts for easy execution (Windows/Unix)
- ✅ **Comprehensive Logging** - Detailed logs for debugging and monitoring
- ✅ **Template Support** - Pre-configured templates for MNI brain and hippocampus ROI

---

## Pipeline Architecture

The pipeline consists of **4 main stages**, each with multiple substages:

### Stage 1: Environment Setup
- GPU detection and verification
- Package dependency checking and installation
- Performance testing and optimization

### Stage 2: Data Preparation
- **Split:** Stratified train/validation/test splitting
- **Analyze:** Metadata analysis and statistics generation

### Stage 3: NIfTI Processing
- **Skull Stripping:** HD-BET-based brain extraction
- **Template Registration:** ANTs-based MNI template alignment
- **Labelling:** Temporal sequence organization
- **2D Conversion:** NIfTI to PNG conversion with slice extraction

### Stage 4: Image Processing
- **Center Crop:** Temporal sequence extraction and cropping
- **Image Enhancement:** Grey Wolf Optimizer-based enhancement
- **Data Balancing:** Augmentation and class balancing

---

## Prerequisites

### System Requirements

- **Python:** 3.11 or higher (3.12 recommended)
- **Operating System:** Windows 10/11, Linux, or macOS
- **RAM:** Minimum 8GB (16GB+ recommended for large datasets)
- **Storage:** Sufficient space for processed outputs (typically 2-3x input size)
- **GPU:** Optional but recommended (NVIDIA GPU with CUDA support)

### Required Software

- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **Git** - [Download](https://git-scm.com/downloads)
- **pip** - Usually included with Python

### External Dependencies (Optional)

- **HD-BET** - Automatically installed via pip if needed
- **ANTs** (antspyx) - Required for template registration stage
  - Install manually: `pip install antspyx` or via environment setup stage

### Template Files

Before running the pipeline, download the required template files:

1. **MNI Brain Template:**
   - Location: `support_files/templates/mni-brain/`
   - Download: [MNI152_T1_1mm_brain.nii.gz](https://git.fmrib.ox.ac.uk/fsl/data_standard/-/blob/master/MNI152_T1_1mm_brain.nii.gz?ref_type=heads)
   - See `support_files/templates/mni-brain/README.md` for details

2. **Hippocampus ROI Mask:**
   - Location: `support_files/templates/hippocampal-roi/`
   - Download: [NeuroVault Image 448213](https://neurovault.org/images/448213/)
   - See `support_files/templates/hippocampal-roi/README.md` for details

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zashari/alzheimer-mri-processing-pipeline.git
   cd alzheimer-mri-processing-pipeline
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

5. **Verify installation:**
   ```bash
   adp --help
   ```

### Post-Installation

1. **Download template files** (see [Prerequisites](#template-files))
   - MNI brain template: Place in `support_files/templates/mni-brain/MNI152_T1_1mm_brain.nii.gz`
   - Hippocampus ROI: Place in `support_files/templates/hippocampal-roi/hippho50.nii.gz`
2. **Configure paths** in `configs/default.yaml`:
   - Set `paths.data_root` to your raw dataset directory
   - Set `paths.metadata_csv` to your metadata CSV file path
   - Set `paths.output_root` (default: `outputs`)
3. **Run environment setup** to verify GPU and dependencies:
   ```bash
   adp environment_setup setup --auto-install true
   ```

---

## Quick Start

### 1. Basic Setup

```bash
# Activate your virtual environment
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Run environment setup
adp environment_setup setup --auto-install true --perf-test quick
```

### 2. Prepare Your Data

```bash
# Split data into train/val/test
adp data_preparation split

# Analyze metadata
adp data_preparation analyze
```

### 3. Process NIfTI Files

```bash
# Test each substage on samples before full processing
adp nifti_processing test --substage skull_stripping
adp nifti_processing test --substage template_registration
adp nifti_processing test --substage labelling
adp nifti_processing test --substage twoD_conversion

# Process all files (run after successful tests)
adp nifti_processing process --substage skull_stripping
adp nifti_processing process --substage template_registration
adp nifti_processing process --substage labelling
adp nifti_processing process --substage twoD_conversion
```

### 4. Process Images

```bash
# Test each substage on samples before full processing
adp image_processing test --substage center_crop
adp image_processing test --substage image_enhancement
adp image_processing test --substage data_balancing

# Process all files (run after successful tests)
adp image_processing process --substage center_crop
adp image_processing process --substage image_enhancement
adp image_processing process --substage data_balancing
```

### 5. Run Full Pipeline (Advanced)

For users with sufficient resources, run the complete pipeline:

```bash
# Windows
scripts\windows_based_system\run_full_pipeline.bat

# Linux/macOS
./scripts/unix_based_system/run_full_pipeline.sh
```

**Warning:** This will run all stages sequentially and may take several hours.

---

## Usage

### Command-Line Interface (CLI)

The pipeline provides a unified CLI through the `adp` command (or `python -m data_processing.cli`).

#### General Syntax

```bash
adp <stage> <action> [options]
```

#### Available Stages

- `environment_setup` - Environment configuration and verification
- `data_preparation` - Data splitting and analysis
- `nifti_processing` - NIfTI file processing (4 substages)
- `image_processing` - Image processing (3 substages)

#### Common Options

- `--config <path>` - Specify custom configuration file
- `--stage-config <path>` - Override stage-specific config file path
- `--set <key=value>` - Override configuration values (repeatable, use comma for arrays)
- `--debug` - Enable debug output
- `--quiet` - Suppress non-essential output
- `--dry-run` - Show what would be done without executing
- `--log-file <path>` - Write logs to file
- `--seed <int>` - Set random seed
- `--workers <int>` - Number of parallel workers
- `--version` - Show version information

#### Examples

```bash
# Environment setup with full performance test
adp environment_setup setup --auto-install true --perf-test full

# Data preparation with custom split ratios (must sum to 1.0, comma-separated)
adp data_preparation split --set data_preparation.split_ratios=0.7,0.2,0.1

# NIfTI processing with specific substage and debug output
adp nifti_processing process --substage skull_stripping --debug

# Image processing with custom output root (affects all stages)
adp image_processing process --substage center_crop --set paths.output_root=/custom/output/path

# Or override stage-specific output directory
adp image_processing process --substage center_crop --set image_processing.center_crop.output_dir=custom_center_crop
```

### Convenience Scripts

For non-technical users or quick execution, use the provided convenience scripts:

#### Individual Stage Scripts

Run stages one-by-one with progress monitoring:

**Windows:**
```cmd
scripts\windows_based_system\run_environment_setup.bat
scripts\windows_based_system\run_data_preparation.bat
scripts\windows_based_system\run_nifti_processing.bat
scripts\windows_based_system\run_image_processing.bat
```

**Linux/macOS:**
```bash
./scripts/unix_based_system/run_environment_setup.sh
./scripts/unix_based_system/run_data_preparation.sh
./scripts/unix_based_system/run_nifti_processing.sh
./scripts/unix_based_system/run_image_processing.sh
```

#### Full Pipeline Scripts

Run the complete pipeline end-to-end (requires significant resources):

**Windows:**
```cmd
scripts\windows_based_system\run_full_pipeline.bat
```

**Linux/macOS:**
```bash
./scripts/unix_based_system/run_full_pipeline.sh
```

For detailed script documentation, see [`scripts/README.md`](scripts/README.md).

### Pipeline Stages

#### Stage 1: Environment Setup

```bash
# Verify environment without installation
adp environment_setup verify

# Setup with automatic package installation
adp environment_setup setup --auto-install true

# Full setup with performance testing
adp environment_setup setup --auto-install true --perf-test full
```

**Actions:**
- `verify` - Check GPU, dependencies, and environment
- `setup` - Full environment setup with optional auto-install

#### Stage 2: Data Preparation

```bash
# Split data into train/val/test sets
adp data_preparation split

# Analyze metadata and generate statistics
adp data_preparation analyze
```

**Actions:**
- `split` - Stratified data splitting
- `analyze` - Metadata analysis and statistics
- `manifests` - Generate manifest CSV files without file copying

#### Stage 3: NIfTI Processing

```bash
# Test on a sample before full processing
adp nifti_processing test --substage skull_stripping

# Process all files
adp nifti_processing process --substage skull_stripping
adp nifti_processing process --substage template_registration
adp nifti_processing process --substage labelling
adp nifti_processing process --substage twoD_conversion
```

**Substages:**
- `skull_stripping` - HD-BET-based brain extraction
- `template_registration` - ANTs-based MNI template alignment
- `labelling` - Temporal sequence organization
- `twoD_conversion` - NIfTI to PNG conversion

#### Stage 4: Image Processing

```bash
# Process images sequentially
adp image_processing process --substage center_crop
adp image_processing process --substage image_enhancement
adp image_processing process --substage data_balancing
```

**Substages:**
- `center_crop` - Temporal sequence extraction
- `image_enhancement` - Grey Wolf Optimizer-based enhancement
- `data_balancing` - Augmentation and class balancing

---

## Configuration

The pipeline uses YAML-based configuration files located in the `configs/` directory.

### Configuration Files

- `configs/default.yaml` - Main configuration file
- `configs/stages/environment_setup.yaml` - Environment setup settings
- `configs/stages/data_preparation.yaml` - Data preparation settings
- `configs/stages/nifti_processing.yaml` - NIfTI processing settings
- `configs/stages/image_processing.yaml` - Image processing settings

### Key Configuration Sections

#### Paths

```yaml
paths:
  data_root: "path/to/raw/nifti/files"  # Root directory containing ADNI-like dataset structure
  output_root: "outputs"  # Base output directory (relative to project root)
  metadata_csv: "path/to/metadata.csv"  # Primary metadata CSV file path
```

#### Data Preparation

```yaml
data_preparation:
  required_visits: ["sc", "m06", "m12"]  # Required visits for complete sequences
  split_ratios: [0.7, 0.15, 0.15]  # Train/Val/Test ratios (must sum to 1.0)
  stratify_by: "Group"  # Column name for stratification
  shuffle: true  # Shuffle subjects before splitting
```

#### NIfTI Processing

```yaml
nifti_processing:
  skull_stripping:
    device: "cuda"  # Options: "cuda", "cpu", "mps"
    use_tta: false  # Test-time augmentation
  template_registration:
    mni_template_path: "support_files/templates/mni-brain/MNI152_T1_1mm_brain.nii.gz"
    hippocampus_roi_path: "support_files/templates/hippocampal-roi/hippho50.nii.gz"
    registration:
      type: "SyNAggro"  # Options: "SyN", "SyNRA", "SyNAggro", "SyNCC", "Affine", "Rigid"
      num_threads: 8
```

### Overriding Configuration

You can override configuration values via CLI:

```bash
# Single override - update split ratios (must sum to 1.0, comma-separated)
adp data_preparation split --set data_preparation.split_ratios=0.7,0.2,0.1

# Multiple overrides
adp nifti_processing process --substage skull_stripping \
  --set nifti_processing.skull_stripping.device=cpu \
  --set nifti_processing.skull_stripping.use_tta=true

# Override template paths
adp nifti_processing process --substage template_registration \
  --set nifti_processing.template_registration.mni_template_path=/custom/path/to/template.nii.gz
```

---

## Project Structure

```
alzheimer-mri-processing-pipeline/
├── configs/                 # Configuration files
│   ├── default.yaml        # Main configuration
│   └── stages/             # Stage-specific configurations
├── docs/                   # Documentation
│   ├── CHANGELOG.md        # Version history
│   └── THIRD_PARTY_NOTICES.md
├── scripts/                # Convenience scripts
│   ├── unix_based_system/  # Shell scripts (Linux/macOS)
│   └── windows_based_system/  # Batch scripts (Windows)
├── src/
│   └── data_processing/    # Main package
│       ├── cli.py          # CLI entry point
│       ├── config/         # Configuration management
│       ├── data_preparation/  # Data preparation stage
│       ├── environment_setup/  # Environment setup stage
│       ├── image_processing/   # Image processing stage
│       ├── nifti_processing/   # NIfTI processing stage
│       └── stages/         # Stage registry
├── support_files/
│   └── templates/          # Template files (MNI brain, ROI masks)
├── outputs/                # Generated outputs (gitignored)
├── .reports/               # Execution reports (gitignored)
├── pyproject.toml          # Package metadata and build configuration
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

---

## Output Structure

After running the pipeline, outputs are organized as follows:

```
outputs/
├── 1_splitted_sequential/     # Data preparation outputs
│   ├── train/
│   ├── val/
│   └── test/
├── manifests/                  # CSV manifests (at output root)
│   ├── metadata_split.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── 2_skull_stripping/         # Skull stripping outputs
├── 3_optimal_slices/          # Template registration outputs
│   ├── axial/
│   ├── coronal/
│   ├── sagittal/
│   └── hippocampus_masks_3D/
├── 4_labelling/                # Labelling outputs
├── 5_twoD/                    # 2D conversion outputs
├── 6_center_crop/              # Center crop outputs
├── 7_enhanced/                 # Image enhancement outputs
└── 8_balanced/                # Data balancing outputs

.reports/                      # JSON execution reports (at project root)
├── environment_setup_*.json
├── data_preparation_*.json
├── nifti_processing_*.json
└── image_processing_*.json

.visualizations/                # Visualization outputs (at output root)
├── data_preparation/
├── nifti_processing/
│   ├── skull_stripping/
│   ├── template_registration/
│   ├── labelling/
│   └── twoD_conversion/
└── image_processing/
    ├── center_crop/
    ├── image_enhancement/
    └── data_balancing/
```

---

## Troubleshooting

### Common Issues

#### 1. Stage Registration Fails

**Problem:** `Unknown stage: <stage_name>. Available: []`

**Solution:**
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Verify installation
python -c "from data_processing.stages import available_stages; print(list(available_stages().keys()))"
# Should output: ['environment_setup', 'data_preparation', 'nifti_processing', 'image_processing']
```

#### 2. GPU Not Detected

**Problem:** Pipeline runs on CPU despite GPU being available

**Solution:**
```bash
# Verify CUDA installation
nvidia-smi  # Should show GPU information

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 3. Template Files Missing

**Problem:** Template registration fails with file not found error

**Solution:**
1. Download MNI brain template to `support_files/templates/mni-brain/MNI152_T1_1mm_brain.nii.gz`
2. Download hippocampus ROI to `support_files/templates/hippocampal-roi/hippho50.nii.gz`
3. Verify template paths in `configs/stages/nifti_processing.yaml` under `template_registration` section, or override via CLI:
   ```bash
   adp nifti_processing process --substage template_registration \
     --set nifti_processing.template_registration.mni_template_path=/path/to/template.nii.gz
   ```

#### 4. Out of Memory Errors

**Problem:** GPU runs out of memory during processing

**Solution:**
- Reduce batch size in configuration
- Process files in smaller batches
- Use CPU mode: `--set nifti_processing.skull_stripping.device=cpu`
- Close other GPU-intensive applications

#### 5. Permission Errors on Windows

**Problem:** `PermissionError: [WinError 5] Access is denied`

**Solution:**
- Close Jupyter Lab or other Python processes using the environment
- Run terminal as administrator (if necessary)
- Ensure no files are locked by other processes

### Getting Help

- **Check Logs:** Review `.reports/*.json` for detailed error information
- **Enable Debug Mode:** Use `--debug` flag for detailed output
- **GitHub Issues:** Report bugs or ask questions on [GitHub Issues](https://github.com/zashari/alzheimer-mri-processing-pipeline/issues)
- **Email:** Contact `izzat.zaky@gmail.com` for direct support

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow code style** - Use consistent formatting and naming conventions
3. **Add tests** - Include tests for new features when possible
4. **Update documentation** - Keep README and docstrings up to date
5. **Submit a Pull Request** - Include a clear description of changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/alzheimer-mri-processing-pipeline.git
cd alzheimer-mri-processing-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Make changes and test
adp --help

# Run tests (if available)
pytest
```

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Citation

If you use this pipeline in your research, please cite it appropriately. This helps track the impact of this work and supports future development.

### Quick Citation

**BibTeX format:**
```bibtex
@misc{alzheimer_mri_processing_pipeline_2025,
  title        = {alzheimer-mri-processing-pipeline},
  author       = {Ashari, Zaky and contributors},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/zashari/alzheimer-mri-processing-pipeline}},
  note         = {Version v1.1.1}
}
```

**APA format:**
```
Ashari, Z. (2025). alzheimer-mri-processing-pipeline [Computer software]. GitHub. 
https://github.com/zashari/alzheimer-mri-processing-pipeline
```

### Citation File

- **[`CITATION.cff`](CITATION.cff)** — Citation File Format for automatic citation (GitHub will display a "Cite this repository" button)

### References

This pipeline is built upon and informed by the following research papers and methods. When publishing work that uses this pipeline, please cite the relevant references:

#### Skull Stripping
- Druzhinina, P., & Kondrateva, E. (2022). *The effect of skull-stripping on transfer learning for 3D MRI models: ADNI data.* Medical Imaging with Deep Learning (MIDL). [https://openreview.net/forum?id=IS1yeyiAFZS](https://openreview.net/forum?id=IS1yeyiAFZS)
- Isensee, F., Schell, M., Tursunova, I., *et al.* (2019). *Automated brain extraction of multi-sequence MRI using artificial neural networks.* Human Brain Mapping. [https://doi.org/10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750) — **HD-BET tool**: [https://github.com/MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET)

#### Alzheimer's Disease ROI
- Hassouneh, A., Bazuin, B., Danna-Dos-Santos, A., Acar, I., Abdel-Qader, I., & ADNI. (2024). *Feature Importance Analysis and Machine Learning for Alzheimer's Disease Early Detection: Feature Fusion of the Hippocampus, Entorhinal Cortex, and Standardized Uptake Value Ratio.* [https://doi.org/10.1159/000538486](https://doi.org/10.1159/000538486)

#### Data Augmentation / Domain Adaptation
- Llambias, S. N., Nielsen, M., & Mehdipour Ghazi, M. (2023). *Data Augmentation-Based Unsupervised Domain Adaptation In Medical Imaging.* arXiv preprint arXiv:2308.04395. [https://doi.org/10.48550/arXiv.2308.04395](https://doi.org/10.48550/arXiv.2308.04395)

#### Image Enhancement / Optimization
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Advances in Engineering Software, 69, 46–61. [https://doi.org/10.1016/j.advengsoft.2013.12.007](https://doi.org/10.1016/j.advengsoft.2013.12.007)

#### Dataset
- **ADNI Dataset**: If using ADNI data, follow [ADNI citation requirements](https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Manuscript_Citations.pdf)
- Imaging Data Archive (IDA) at LONI: [https://ida.loni.usc.edu/](https://ida.loni.usc.edu/)

---

## Acknowledgments

This work was developed as part of a final-year thesis project. Special thanks to:

- **Dr. Dani Suandi, S.Si., M.Si.** — Lecturer in Mathematics, Binus University; Lecturer, School of Computer Science, Binus University — for guidance and supervision throughout this project.
  - [Google Scholar](https://scholar.google.com/citations?user=LKUVKGEAAAAJ&hl=id)
  - [ResearchGate](https://www.researchgate.net/profile/Dani-Suandi)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Third-party dependencies** (e.g., HD-BET, ANTs) have their own licenses. Please review and comply with their respective license terms. See [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) for details.

---

## Additional Resources

- **[Changelog](docs/CHANGELOG.md)** - Version history and release notes
- **[Security Policy](SECURITY.md)** - Security reporting and supported versions
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to this project
- **[Scripts Documentation](scripts/README.md)** - Detailed script usage guide

---

**For questions or support, please open an issue on [GitHub](https://github.com/zashari/alzheimer-mri-processing-pipeline/issues) or contact `izzat.zaky@gmail.com`.**
