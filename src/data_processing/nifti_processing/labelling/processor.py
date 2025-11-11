"""Labelling processor for organizing temporal sequences by subject and group."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class LabellingProcessor:
    """Handles organization of temporal sequences by subject and diagnostic group."""

    def __init__(
        self,
        metadata_csv: Path,
        required_visits: List[str],
        groups: List[str],
        splits: List[str],
        duplicate_strategy: str = "largest",
        remove_empty_files: bool = True,
        verify_copies: bool = True,
        verbose: bool = False
    ):
        """
        Initialize labelling processor.

        Args:
            metadata_csv: Path to metadata CSV with Subject, Group, Split columns
            required_visits: List of required visits (e.g., ["sc", "m06", "m12"])
            groups: List of groups to process (e.g., ["AD", "CN"])
            splits: List of splits to process (e.g., ["train", "val", "test"])
            duplicate_strategy: Strategy for handling duplicates ("largest", "newest", "skip")
            remove_empty_files: Whether to remove empty files (0 bytes)
            verify_copies: Whether to verify files after copying
            verbose: Show detailed output
        """
        self.metadata_csv = Path(metadata_csv)
        self.required_visits = required_visits
        self.groups = groups
        self.splits = splits
        self.duplicate_strategy = duplicate_strategy
        self.remove_empty_files = remove_empty_files
        self.verify_copies = verify_copies
        self.verbose = verbose

        # Will be populated after loading metadata
        self.subject_to_group: Dict[str, str] = {}
        self.subject_to_split: Dict[str, str] = {}
        self.metadata_loaded = False

    def load_metadata(self) -> Tuple[bool, Optional[str]]:
        """
        Load metadata CSV and create subject mappings.

        Returns:
            Tuple of (success, error_message)
        """
        if not PANDAS_AVAILABLE:
            return False, "pandas is required for metadata operations"

        if not self.metadata_csv.exists():
            return False, f"Metadata CSV not found: {self.metadata_csv}"

        try:
            df = pd.read_csv(self.metadata_csv)

            # Check required columns
            required_cols = ["Subject", "Group", "Split"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"

            # Create mappings
            self.subject_to_group = dict(zip(df["Subject"], df["Group"]))
            self.subject_to_split = dict(zip(df["Subject"], df["Split"]))

            self.metadata_loaded = True
            return True, None

        except Exception as e:
            return False, f"Failed to load metadata: {e}"

    @staticmethod
    def extract_subject_id_temporal(filename: str) -> Optional[str]:
        """
        Extract subject ID from temporal filename.

        Args:
            filename: Filename like "002_S_0295_sc_optimal_axial_x95.nii.gz"

        Returns:
            Subject ID like "002_S_0295" or None if not found
        """
        clean_name = filename.replace('.nii.gz', '').replace('.nii', '')
        parts = clean_name.split('_')

        for i in range(len(parts) - 2):
            if (parts[i+1] == 'S' and
                parts[i].isdigit() and
                parts[i+2].isdigit()):
                return f"{parts[i]}_S_{parts[i+2]}"
        return None

    def extract_visit_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract visit code from temporal filename.

        Args:
            filename: Filename like "002_S_0295_sc_optimal_axial_x95.nii.gz"

        Returns:
            Visit code like "sc" or None if not found
        """
        for visit in self.required_visits:
            if f"_{visit}_" in filename:
                return visit
        return None

    def scan_slice_files(
        self,
        slice_root: Path,
        pattern: str
    ) -> Tuple[Dict[str, Dict[str, List[Path]]], List[Tuple[str, str, List[Tuple[Path, int]]]], List[Path]]:
        """
        Scan for slice files and organize by subject and visit.

        Args:
            slice_root: Root directory containing split subdirectories
            pattern: File pattern to match (e.g., "*_optimal_axial_x*.nii.gz")

        Returns:
            Tuple of:
            - subject_files: Dict[subject_id][visit] -> List[Path]
            - duplicates: List of (subject_id, visit, [(filepath, size), ...])
            - empty_files: List of empty file paths
        """
        subject_files: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
        duplicates: List[Tuple[str, str, List[Tuple[Path, int]]]] = []
        empty_files: List[Path] = []

        # Track all files found for duplicate detection
        file_tracker: Dict[Tuple[str, str], List[Path]] = defaultdict(list)

        # Scan all splits
        for split in self.splits:
            split_dir = slice_root / split
            if not split_dir.exists():
                continue

            for subject_dir in split_dir.iterdir():
                if not subject_dir.is_dir():
                    continue

                subject_id = subject_dir.name
                slice_files = list(subject_dir.glob(pattern))

                for filepath in slice_files:
                    filename = filepath.name
                    visit = self.extract_visit_from_filename(filename)

                    if visit:
                        file_key = (subject_id, visit)
                        file_tracker[file_key].append(filepath)

        # Process file tracker and handle duplicates
        for (subject_id, visit), files in file_tracker.items():
            if len(files) > 1:
                # Handle duplicates based on strategy
                files_with_size = [(f, f.stat().st_size) for f in files]
                files_with_size.sort(key=lambda x: (-x[1], x[0].name))

                duplicates.append((subject_id, visit, files_with_size))

                # Select file based on strategy
                if self.duplicate_strategy == "largest":
                    selected = files_with_size[0][0]
                    subject_files[subject_id][visit] = [selected]
                elif self.duplicate_strategy == "newest":
                    # Sort by modification time (newest first)
                    files_with_mtime = [(f, f.stat().st_mtime) for f in files]
                    files_with_mtime.sort(key=lambda x: -x[1])
                    selected = files_with_mtime[0][0]
                    subject_files[subject_id][visit] = [selected]
                elif self.duplicate_strategy == "skip":
                    # Skip all duplicates
                    continue
                else:
                    # Default to largest
                    selected = files_with_size[0][0]
                    subject_files[subject_id][visit] = [selected]

                # Track empty files for removal
                if self.remove_empty_files:
                    for f, size in files_with_size[1:]:
                        if size == 0:
                            empty_files.append(f)
            else:
                # Single file - check if empty
                filepath = files[0]
                if self.remove_empty_files and filepath.stat().st_size == 0:
                    empty_files.append(filepath)
                else:
                    subject_files[subject_id][visit] = [filepath]

        return subject_files, duplicates, empty_files

    def process_slice_type(
        self,
        slice_type: str,
        slice_root: Path,
        pattern: str,
        output_base: Path
    ) -> Dict:
        """
        Process a single slice type (axial, coronal, sagittal).

        Args:
            slice_type: Name of slice type (e.g., "axial")
            slice_root: Root directory for this slice type
            pattern: File pattern to match
            output_base: Base output directory

        Returns:
            Dictionary with processing statistics
        """
        stats = defaultdict(int)
        unmatched_subjects = []
        unknown_groups = []
        processed_subjects = defaultdict(lambda: defaultdict(int))
        temporal_stats = defaultdict(lambda: defaultdict(int))

        # Scan for files
        subject_files, duplicates, empty_files = self.scan_slice_files(slice_root, pattern)

        # Process each subject
        for subject_id, visit_files in subject_files.items():
            # Check if subject has complete temporal sequence
            available_visits = set(visit_files.keys())
            if available_visits != set(self.required_visits):
                stats["incomplete_sequence"] += 1
                continue

            # Get subject metadata
            if subject_id not in self.subject_to_group:
                unmatched_subjects.append(subject_id)
                stats["unmatched"] += 1
                continue

            group = self.subject_to_group[subject_id]
            split = self.subject_to_split[subject_id]

            if group not in self.groups:
                unknown_groups.append((subject_id, group))
                stats["unknown_group"] += 1
                continue

            if split not in self.splits:
                stats["unknown_split"] += 1
                continue

            # Create output directory
            subject_output_dir = output_base / slice_type / split / group / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy all temporal files for this subject
            subject_file_count = 0
            for visit in self.required_visits:
                for filepath in visit_files[visit]:
                    filename = filepath.name
                    dst = subject_output_dir / filename

                    if not dst.exists():
                        try:
                            shutil.copy2(filepath, dst)

                            # Verify copy if enabled
                            if self.verify_copies:
                                if dst.stat().st_size != filepath.stat().st_size:
                                    stats["copy_errors"] += 1
                                    continue

                            subject_file_count += 1
                            temporal_stats[split][group] += 1
                        except Exception as e:
                            if self.verbose:
                                print(f"Error copying {filepath} to {dst}: {e}")
                            stats["copy_errors"] += 1

            if subject_file_count > 0:
                processed_subjects[split][group] += 1
                stats[(split, group)] += subject_file_count
                stats["successful_subjects"] += 1

        return {
            "stats": dict(stats),
            "unmatched_subjects": unmatched_subjects,
            "unknown_groups": unknown_groups,
            "processed_subjects": dict(processed_subjects),
            "temporal_stats": {k: dict(v) for k, v in temporal_stats.items()},
            "duplicates": duplicates,
            "empty_files": empty_files,
            "total_files_scanned": sum(len(visit_files) for visit_files in subject_files.values())
        }


