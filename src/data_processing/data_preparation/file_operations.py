"""File operations for copying/linking NIfTI files."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from .search import find_nifti_file_comprehensive


class FileOperations:
    """Handles file copying and linking operations for NIfTI files."""

    def __init__(self, use_symlinks: bool = False, debug_mode: bool = False):
        """
        Initialize file operations handler.

        Args:
            use_symlinks: If True, create symbolic links instead of copying
            debug_mode: If True, print detailed debug information
        """
        self.use_symlinks = use_symlinks
        self.debug_mode = debug_mode
        self.copy_stats = defaultdict(lambda: defaultdict(int))
        self.errors = []

    def mirror_nifti_files(
        self,
        df: pd.DataFrame,
        raw_dir: Path,
        output_dir: Path,
        formatter=None,
        dry_run: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """
        Copy or link NIfTI files to output directory with standardized naming.

        Directory structure:
        output_dir/
            train/
                subject_id/
                    subject_id_visit.nii[.gz]
            val/
                subject_id/
                    subject_id_visit.nii[.gz]
            test/
                subject_id/
                    subject_id_visit.nii[.gz]

        Args:
            df: DataFrame with columns: Subject, Visit, Image Data ID, Split
            raw_dir: Root directory of raw ADNI data
            output_dir: Output directory for organized files
            formatter: Optional formatter for progress display
            dry_run: If True, simulate operations without actually copying

        Returns:
            Dictionary with copy statistics per split
        """
        if dry_run:
            if formatter:
                formatter.console.print("[yellow][DRY RUN][/yellow] Simulating file operations...")
            return self.copy_stats

        # Reset statistics
        self.copy_stats.clear()
        self.errors.clear()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Count total files
        total_files = len(df)

        # Create progress bar if formatter available
        progress = None
        if formatter and not formatter.json_only and not formatter.quiet:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=formatter.console
            )
            task = progress.add_task("Copying files", total=total_files)
            progress.start()
        else:
            # Use tqdm for terminal progress if no formatter
            progress = tqdm(total=total_files, desc="Copying files", unit="file")

        try:
            for _, row in df.iterrows():
                subject = row["Subject"]
                visit = row["Visit"]
                image_id = row.get("Image Data ID", row.get("Image ID", ""))
                split = row["Split"]

                # Update progress description
                if hasattr(progress, "update"):
                    # Rich progress
                    progress.update(task, description=f"Copying {split}/{subject}_{visit}")
                else:
                    # tqdm progress
                    progress.set_description(f"Copying {split}/{subject}_{visit}")

                # Create output directories
                subset_dir = output_dir / split
                subset_dir.mkdir(exist_ok=True)
                dest_subj_dir = subset_dir / subject
                dest_subj_dir.mkdir(parents=True, exist_ok=True)

                # Find source file
                subject_dir = raw_dir / subject
                src = find_nifti_file_comprehensive(subject_dir, image_id, debug=self.debug_mode)

                if not src:
                    error_msg = f"[{split}] {subject}_{visit}: No NIfTI file found for image ID {image_id}"
                    self.errors.append(error_msg)
                    self.copy_stats[split]["errors"] += 1

                    if self.debug_mode:
                        print(f"\n[DEBUG] Failed to find: {subject}/{image_id}")
                        self._debug_directory_structure(subject_dir, image_id)

                    if hasattr(progress, "update"):
                        progress.update(task, advance=1)
                    else:
                        progress.update(1)
                    continue

                # Create destination filename
                if src.suffix == ".gz":
                    dest_filename = f"{subject}_{visit}.nii.gz"
                else:
                    dest_filename = f"{subject}_{visit}.nii"
                dest = dest_subj_dir / dest_filename

                # Copy or link file
                try:
                    if dest.exists():
                        self.copy_stats[split]["skipped"] += 1
                    else:
                        if self.use_symlinks:
                            self._create_symlink(src, dest)
                        else:
                            shutil.copy2(src, dest)
                        self.copy_stats[split]["copied"] += 1

                except Exception as e:
                    error_msg = f"[{split}] Could not copy {src} → {dest}: {e}"
                    self.errors.append(error_msg)
                    self.copy_stats[split]["errors"] += 1

                    if self.debug_mode:
                        print(f"\n[DEBUG] Copy error: {e}")

                # Update progress
                if hasattr(progress, "update"):
                    progress.update(task, advance=1)
                else:
                    progress.update(1)

        finally:
            # Close progress bar
            if hasattr(progress, "stop"):
                progress.stop()
            else:
                progress.close()

        # Convert defaultdict to regular dict for return
        return dict(self.copy_stats)

    def _create_symlink(self, src: Path, dest: Path) -> None:
        """
        Create symbolic link with Windows fallback.

        Args:
            src: Source file path
            dest: Destination link path
        """
        try:
            os.symlink(src.resolve(), dest)
        except OSError as e:
            # Windows requires admin privileges for symlinks
            if "[WinError 1314]" in str(e) or "privilege" in str(e).lower():
                # Fall back to copying on Windows without admin rights
                shutil.copy2(src, dest)
            else:
                raise

    def _debug_directory_structure(self, subject_dir: Path, image_id: str) -> None:
        """
        Print debug information about directory structure.

        Args:
            subject_dir: Subject directory to analyze
            image_id: Image ID being searched for
        """
        if not subject_dir.exists():
            print(f"  Subject directory does not exist: {subject_dir}")
            return

        # Look for directories with the image ID
        id_dirs = list(subject_dir.glob(f"**/{image_id}"))
        if id_dirs:
            print(f"  Found {len(id_dirs)} directories named '{image_id}':")
            for d in id_dirs[:3]:  # Show first 3
                nii_files = list(d.glob("*.nii*"))
                print(f"    - {d.relative_to(subject_dir)}: {len(nii_files)} NIfTI files")
        else:
            print(f"  No directories found named '{image_id}'")

            # Show sample of what exists
            all_dirs = []
            for root, dirs, _ in os.walk(subject_dir):
                for d in dirs:
                    if d.startswith("I"):  # Image IDs typically start with I
                        all_dirs.append(d)
                        if len(all_dirs) >= 5:
                            break
                if len(all_dirs) >= 5:
                    break

            if all_dirs:
                print(f"  Sample image ID directories found: {all_dirs}")

    def verify_copied_files(self, output_dir: Path, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Verify that all expected files were copied successfully.

        Args:
            output_dir: Output directory to check
            df: DataFrame with expected files

        Returns:
            Dictionary with verification results
        """
        verification = {
            "missing": [],
            "corrupt": [],
            "ok": []
        }

        for _, row in df.iterrows():
            subject = row["Subject"]
            visit = row["Visit"]
            split = row["Split"]

            # Check both possible extensions
            dest_subj_dir = output_dir / split / subject
            dest_nii = dest_subj_dir / f"{subject}_{visit}.nii"
            dest_niigz = dest_subj_dir / f"{subject}_{visit}.nii.gz"

            if dest_nii.exists():
                dest = dest_nii
            elif dest_niigz.exists():
                dest = dest_niigz
            else:
                verification["missing"].append(f"{split}/{subject}_{visit}")
                continue

            # Check file size (basic corruption check)
            try:
                size = dest.stat().st_size
                if size < 1000:  # Less than 1KB is likely corrupt
                    verification["corrupt"].append(str(dest))
                else:
                    verification["ok"].append(str(dest))
            except Exception:
                verification["corrupt"].append(str(dest))

        return verification

    def get_error_summary(self) -> Dict[str, List[str]]:
        """
        Get categorized error summary.

        Returns:
            Dictionary with errors grouped by type
        """
        error_types = defaultdict(list)

        for error in self.errors:
            if "Subject directory not found" in error:
                error_types["missing_subject_dir"].append(error)
            elif "No NIfTI file found" in error:
                error_types["missing_nifti"].append(error)
            elif "Could not copy" in error:
                error_types["copy_failed"].append(error)
            else:
                error_types["other"].append(error)

        return dict(error_types)