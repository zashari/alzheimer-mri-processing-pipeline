# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.5] - 2025-11-14

### Fixed
- **Critical Windows subprocess hanging issue in skull stripping**: Removed `close_fds=True` parameter on Windows
  - Root cause: `close_fds=True` is incompatible with file handles on Windows, causing indefinite hangs
  - Solution: Remove `close_fds=True` from Windows subprocess calls while keeping it for Unix systems
  - This fixes the 600-second timeout issue that prevented HD-BET from running on Windows
  - Maintains cross-platform compatibility (Unix systems still use `close_fds=True` for security)

### Technical Details
- Modified `HDBETProcessor.process_file()` in `processor.py` line 150
- Windows now uses `subprocess.Popen()` without `close_fds` parameter when using file handles
- Unix/Linux/macOS continue to use `close_fds=True` for proper file descriptor management
- Aligns implementation with the working Jupyter notebook reference code

## [1.4.4] - 2025-11-13

### Fixed
- **Windows timeout issue in skull stripping**: Fixed subprocess hanging indefinitely on Windows (600s timeout)
  - Changed from `subprocess.DEVNULL` for stdout to file handle (proper Windows handle inheritance)
  - Uses file handles for both stdout and stderr (matches working pattern from v1.4.1)
  - Maintains fast execution with `wait()` instead of `communicate()`
  - Proper handle inheritance ensures subprocess completes correctly on Windows
  - Processing completes successfully within expected time (30 seconds - 2 minutes per file)

### Changed
- **Subprocess execution strategy**: Use file handles for both stdout and stderr
  - File handles ensure proper handle inheritance on Windows (avoids timeout issues)
  - Both stdout and stderr use file handles (unlimited buffer, no deadlock risk)
  - Maintains `wait()` for fast execution (doesn't read output during execution)
  - Retry logic handles Windows file locking gracefully
  - Cross-platform compatible (works on Windows, Linux, macOS)

### Technical Details
- Modified `HDBETProcessor.process_file()` to use file handles for both stdout and stderr
- File handles are kept open during `wait()`, then closed after process completes
- Error messages read from stderr file only when `returncode != 0`
- Both temp stdout and stderr files cleaned up with retry logic in `finally` block
- Maintains all existing functionality while fixing Windows timeout issue

[1.4.4]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.4.4

## [1.4.3] - 2025-11-12

### Fixed
- **Performance regression in skull stripping**: Fixed severe slowdown (24+ minutes per file) caused by `subprocess.communicate()`
cls  - Changed from `subprocess.PIPE` + `communicate()` to hybrid approach: file handle for stderr + `wait()`
  - Uses `subprocess.DEVNULL` for stdout (not needed)
  - Uses file handle for stderr (unlimited buffer, no deadlock risk)
  - Uses `wait()` instead of `communicate()` for fast execution (matches notebook performance)
  - Processing time restored to expected 30 seconds - 2 minutes per file
  - Added `_safe_delete()` helper function with retry logic for Windows file locking

### Changed
- **Subprocess execution strategy**: Hybrid approach combining benefits of file handles and `wait()`
  - File handle for stderr provides unlimited buffer (no pipe deadlock risk)
  - `wait()` provides fast execution (doesn't read output during execution)
  - Retry logic handles Windows file locking gracefully
  - Maintains error capture capability (reads stderr file only on error)
  - Cross-platform compatible (works on Windows, Linux, macOS)

### Technical Details
- Modified `HDBETProcessor.process_file()` to use file handle + `wait()` instead of PIPE + `communicate()`
- Added `_safe_delete()` function with exponential backoff retry logic for file deletion
- File handle is kept open during `wait()`, then closed after process completes
- Error messages read from file only when `returncode != 0`
- Temp stderr file cleaned up with retry logic in `finally` block
- Maintains all existing functionality while dramatically improving performance

[1.4.3]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.4.3

## [1.4.2] - 2025-11-12

### Fixed
- **Windows file locking issue**: Fixed `PermissionError` when deleting temporary files on Windows
  - Changed `HDBETProcessor.process_file()` to use `subprocess.PIPE` instead of file handles
  - Eliminates Windows file locking issues where temp files couldn't be deleted after subprocess completion
  - Improved cross-platform compatibility (Windows, Linux, macOS)
  - No more `[WinError 32] The process cannot access the file` errors

### Changed
- **Subprocess output handling**: Refactored to use pipes instead of temporary files for stdout/stderr
  - Uses `subprocess.PIPE` for stdout and stderr redirection
  - Reads output directly from pipes using `process.communicate()`
  - Removed temporary file creation for subprocess output (stdout/stderr)
  - More efficient and avoids file system operations for output capture
- **Error message extraction**: Error messages now read directly from stderr pipe
  - No longer requires opening temp files for error reading
  - Cleaner error handling flow
  - Same error message format (first 500 chars of stderr)

### Technical Details
- Modified `process_file()` to use `subprocess.PIPE` for stdout/stderr
- Changed from `process.wait()` + file reading to `process.communicate()` for pipe-based output
- Updated `check_availability()` to also use `subprocess.PIPE` for consistency
- Removed `temp_stdout` and `temp_stderr` file creation and cleanup
- Added proper pipe cleanup in timeout handling
- Maintains all existing functionality while fixing Windows compatibility

[1.4.2]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.4.2

## [1.4.1] - 2025-11-12

### Fixed
- **Stability fix**: Removed real-time GPU utilization monitoring that was causing process crashes on Windows
  - Fixed silent process termination due to Rich Live display compatibility issues with Windows console
  - Resolved Unicode encoding errors with emoji rendering in GPU status panel
  - Eliminated threading-related instability from background GPU monitoring

### Removed
- **Real-time GPU monitoring**: Removed `GPUMonitor` class and Rich Live display integration
  - Removed misleading GPU utilization percentage display (showed 0% even when GPU was active)
  - Removed GPU memory usage percentage display (not meaningful for subprocess-based tools like HD-BET)
  - Simplified GPU status display to show only essential information: GPU name and total VRAM

### Changed
- **Simplified GPU status display**: GPU status now shows only device name and VRAM capacity
  - More stable and reliable across all platforms
  - No longer displays misleading utilization percentages
  - Cleaner, simpler output focused on essential GPU information
- **Reverted subprocess execution**: Changed `HDBETProcessor.process_file()` back to blocking `wait()` instead of polling loop
  - More stable and reliable subprocess handling
  - Eliminates potential race conditions from polling loop
  - Maintains proper timeout handling and error reporting

### Technical Details
- Removed `create_live_display()` method from `NiftiFormatter`
- Removed `print_gpu_status_update()` method from `NiftiFormatter`
- Removed `GPUMonitor` import and usage from `skull_stripping/runner.py`
- Simplified `hd_bet_status()` to display only GPU name and VRAM
- Reverted `processor.process_file()` to use `process.wait(timeout=...)` instead of polling loop

[1.4.1]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.4.1

## [1.4.0] - 2025-11-12

### Added
- **Real-time GPU utilization monitoring** for `skull_stripping` sub-stage:
  - New `GPUMonitor` class with background thread for continuous GPU status polling
  - Rich Live display integration combining progress bar and real-time GPU status panel
  - GPU status panel shows live GPU utilization percentage and memory usage during processing
  - Automatic GPU monitoring start/stop with context manager support
  - Thread-safe GPU information retrieval for concurrent access

### Changed
- **Non-blocking subprocess execution**: Modified `HDBETProcessor.process_file()` to use polling loop instead of blocking `wait()`
  - Enables real-time GPU monitoring during HD-BET execution
  - Allows Live display updates while subprocess is running
  - Maintains timeout handling and error reporting
- **Enhanced display system**: Added `create_live_display()` method to `NiftiFormatter`
  - Combines Rich Progress bar with GPU status panel in a single Live display
  - Updates GPU status every 0.5 seconds (configurable refresh rate)
  - Gracefully handles GPU unavailability (falls back to progress bar only)
- **Improved user experience**: Both `test` and `process` modes now show real-time GPU utilization
  - GPU status updates continuously during file processing
  - Visual feedback shows actual GPU activity, not just completion status
  - Better visibility into GPU resource usage during long-running operations

### Technical Details
- `GPUMonitor` uses `threading.Thread` with daemon flag for background monitoring
- Polling interval: 0.5 seconds (configurable via `update_interval` parameter)
- Live display refresh rate: 2.0 updates per second (configurable)
- GPU monitoring thread automatically stops when processing completes
- Thread-safe implementation using `threading.Lock` for concurrent access
- Backward compatible: Falls back gracefully if GPU monitoring unavailable

[1.4.0]: https://github.com/zashari/alzheimer-mri-processing-pipeline/releases/tag/v1.4.0

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


