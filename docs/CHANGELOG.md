# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-11-12

### Added
- GPU utilization monitoring for `skull_stripping` sub-stage:
  - New `get_gpu_utilization()` function to query system-wide GPU utilization via `nvidia-smi`
  - Real-time GPU utilization display showing actual GPU compute activity (0-100%)
  - Periodic GPU status updates during processing in both `test` and `process` modes
  - System-wide GPU metrics that accurately reflect HD-BET's GPU usage across processes

### Changed
- **GPU status display**: Replaced PyTorch memory-based display with GPU utilization percentage
  - Now shows actual GPU compute utilization instead of process-specific memory allocation
  - More accurate representation of GPU usage for subprocess-based tools like HD-BET
  - Falls back to PyTorch memory tracking if `nvidia-smi` is unavailable (backward compatible)
- Enhanced GPU monitoring: GPU utilization updates are shown after each file/batch processing
- Improved GPU status visibility: Both GPU utilization and memory usage are displayed when available

### Technical Details
- `get_gpu_info()` now prioritizes `nvidia-smi` queries for system-wide GPU metrics
- Added `print_gpu_status_update()` method to `NiftiFormatter` for periodic GPU status updates
- GPU utilization accurately reflects HD-BET's GPU activity even though it runs as a separate subprocess

[1.3.0]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.3.0

## [1.2.0] - 2025-11-12

### Added
- Progress bar visualization for `skull_stripping` test mode:
  - Rich progress bar with spinner, percentage, and time remaining indicators
  - Dynamic description showing current file being processed
  - Consistent user experience matching the `process` action behavior
  - Visual feedback for better monitoring during test execution

### Changed
- Enhanced test mode UX: Test mode now provides the same level of progress visibility as process mode
- Improved status messages: Success/error messages now include filename for better traceability

[1.2.0]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.2.0

## [1.1.1] - 2025-11-11

### Fixed
- Fixed missing dependencies: Added `rich` (>=13.0.0) and `seaborn` (>=0.13.0) to `pyproject.toml`
- Resolved stage registration issue: All stages now register correctly after dependency installation
- Updated `requirements.txt` to include optional `antspyx` dependency with installation notes

### Changed
- Improved dependency management: Configured Poetry with PyTorch CUDA wheel source
- Updated `requirements.txt` format to use `--index-url` for PyTorch packages
- Enhanced documentation: Added comments in `requirements.txt` explaining optional dependencies

### Dependencies
- Added `rich==13.9.4` for enhanced console output formatting
- Added `seaborn==0.13.2` for visualization support
- Added `antspyx>=0.3.0` (optional) to `requirements.txt` for template registration stage

[1.1.1]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.1.1

## [1.1.0] - 2025-11-11

### Added
- Convenience wrapper scripts for easier pipeline execution:
  - Individual stage scripts for Unix/Linux/macOS (`.sh`) and Windows (`.bat`):
    - `run_environment_setup`: Environment setup with GPU detection
    - `run_data_preparation`: Data splitting and analysis
    - `run_nifti_processing`: All NIfTI processing substages sequentially
    - `run_image_processing`: All image processing substages sequentially
  - Full pipeline scripts for end-to-end execution:
    - `run_full_pipeline.sh` / `run_full_pipeline.bat`: Complete pipeline in one go
- Comprehensive scripts documentation (`scripts/README.md`) with usage instructions

### Changed
- Improved user experience for non-technical users with simplified command execution
- Scripts include error handling and progress indicators

### Documentation
- Added detailed README in scripts directory explaining script usage and alternatives
- Documented prerequisites and troubleshooting tips for script execution

[1.1.0]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.1.0

## [1.0.0] - 2025-11-11

### Added
- Complete image processing stage with three sub-stages:
  - Center crop: Temporal sequence extraction and cropping
  - Image enhancement: Grey Wolf Optimizer-based enhancement
  - Data balancing: Augmentation and class balancing
- Template download instructions for MNI brain and hippocampus ROI masks
- Documentation reorganization: moved CHANGELOG and THIRD_PARTY_NOTICES to docs/ directory

### Fixed
- Critical bug: Added missing `nifti_processing_stage` import in stage registry
- Removed empty `visualizations/` directory that was not being used

### Changed
- Project status updated from pre-release (WIP) to stable release
- Version bumped from 0.1.0 to 1.0.0
- Updated all version references across documentation files

### Removed
- Unused files: `config/validators.py`, `utils/timers.py`, `paths.py`
- Empty visualizations module directory

### Documentation
- Added README files in template directories with download instructions
- Reorganized documentation structure for better maintainability
- Updated cross-references after file reorganization

[1.0.0]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.0.0

## [0.1.0] - 2025-11-09

### Added
- Initial release of Alzheimer MRI processing pipeline
- Environment setup stage with GPU detection and package management
- Data preparation stage with metadata handling and stratified splitting
- NIfTI processing stage with 4 sub-stages:
  - Skull stripping (HD-BET integration)
  - Template registration (ANTs integration)
  - Labelling (temporal sequence organization)
  - 2D conversion (NIfTI to PNG)
- Comprehensive configuration system (YAML-based)
- Standardized output formatting with Rich library
- JSON report generation for all stages
- Visualization support for analysis
- Resume capability for long-running processes
- Full Windows and Unix support
- Documentation and citation files
- ADNI Data Use Agreement compliance documentation

### Documentation
- README with comprehensive usage instructions
- ADNI DUA compliance section
- Citation guide with references
- Acknowledgments section

[0.1.0]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v0.1.0


