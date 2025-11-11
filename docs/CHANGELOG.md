# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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


