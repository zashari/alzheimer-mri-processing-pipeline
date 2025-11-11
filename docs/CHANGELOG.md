# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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


