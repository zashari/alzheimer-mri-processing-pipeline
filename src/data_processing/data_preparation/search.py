from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List


def find_nifti_file_comprehensive(
    subject_dir: str | Path,
    image_id: str,
    debug: bool = False
) -> Optional[Path]:
    """
    Comprehensive search for NIfTI files by Image ID.
    Handles ADNI directory structure where files are in:
    subject/scan_type/date_time/image_id/filename.nii

    Priority system:
    - Scaled_2 files have highest priority
    - Scaled files have medium priority
    - Other files have lowest priority

    Args:
        subject_dir: Path to subject directory
        image_id: Image ID to search for
        debug: If True, print debug information

    Returns:
        Path to the found NIfTI file, or None if not found
    """
    # Ensure we're working with strings for os.walk
    subject_dir_str = str(subject_dir)

    if not os.path.exists(subject_dir_str):
        if debug:
            print(f"  [DEBUG] Subject directory does not exist: {subject_dir_str}")
        return None

    # Clean the image ID
    image_id = str(image_id).strip()

    # Store all found files with priority
    found_files: List[Tuple[int, str]] = []

    # Walk through all directories
    for root, dirs, files in os.walk(subject_dir_str):
        # Get the current directory name
        current_dir = os.path.basename(root)

        # Check if this is our target image ID directory
        if current_dir == image_id:
            # Look for NIfTI files
            for filename in files:
                if filename.endswith((".nii", ".nii.gz")):
                    # Build the full path using os.path.join
                    full_path = os.path.join(root, filename)

                    # Double-check the file exists
                    if os.path.isfile(full_path):
                        # Determine priority based on path
                        # Check entire path, not just immediate directory
                        path_str = str(root)
                        if "Scaled_2" in path_str:
                            priority = 2
                        elif "Scaled" in path_str and "Scaled_2" not in path_str:
                            priority = 1
                        else:
                            priority = 0

                        found_files.append((priority, full_path))

                        if debug:
                            print(f"  [DEBUG] Found: {full_path} (priority={priority})")

    # If we found any files, return the highest priority one
    if found_files:
        # Sort by priority (highest first)
        found_files.sort(key=lambda x: x[0], reverse=True)
        best_file = Path(found_files[0][1])

        if debug:
            print(f"  [DEBUG] Selected: {best_file} (from {len(found_files)} options)")

        return best_file

    # If we get here, no file was found
    if debug:
        print(f"  [DEBUG] No NIfTI files found for image_id={image_id}")

    return None


def find_all_nifti_files(subject_dir: str | Path, extensions: Tuple[str, ...] = (".nii", ".nii.gz")) -> List[Path]:
    """
    Find all NIfTI files in a subject directory.

    Args:
        subject_dir: Path to subject directory
        extensions: Tuple of file extensions to search for

    Returns:
        List of paths to NIfTI files
    """
    subject_dir = Path(subject_dir)
    nifti_files = []

    if not subject_dir.exists():
        return nifti_files

    for ext in extensions:
        pattern = f"**/*{ext}"
        found = subject_dir.glob(pattern)
        nifti_files.extend(found)

    # Filter to only existing files
    nifti_files = [f for f in nifti_files if f.is_file()]

    return sorted(nifti_files)

