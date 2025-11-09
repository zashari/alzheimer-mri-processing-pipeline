"""2D conversion processor for converting NIfTI slices to PNG images."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    NIBABEL_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    PIL_AVAILABLE = False


class TwoDConversionProcessor:
    """Handles conversion of NIfTI slices to PNG images."""

    def __init__(
        self,
        required_visits: List[str],
        groups: List[str],
        splits: List[str],
        intensity_percentile: Tuple[int, int] = (1, 99),
        target_size: Tuple[int, int] = (256, 256),
        interpolation_method: str = "LANCZOS",
        verify_outputs: bool = True,
        preserve_original_size: bool = False,
        verbose: bool = False
    ):
        """
        Initialize 2D conversion processor.

        Args:
            required_visits: List of required visits (e.g., ["sc", "m06", "m12"])
            groups: List of groups to process (e.g., ["AD", "CN"])
            splits: List of splits to process (e.g., ["train", "val", "test"])
            intensity_percentile: Percentile range for normalization (low, high)
            target_size: Target resolution for PNG images (width, height)
            interpolation_method: Interpolation method ("LANCZOS", "BILINEAR", "NEAREST", "BICUBIC")
            verify_outputs: Whether to verify PNG files after saving
            preserve_original_size: Whether to track original sizes for reporting
            verbose: Show detailed output
        """
        self.required_visits = required_visits
        self.groups = groups
        self.splits = splits
        self.intensity_percentile = intensity_percentile
        self.target_size = target_size
        self.interpolation_method = interpolation_method
        self.verify_outputs = verify_outputs
        self.preserve_original_size = preserve_original_size
        self.verbose = verbose

        # Map interpolation method string to PIL constant
        self.interpolation_map = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "BILINEAR": Image.Resampling.BILINEAR,
            "NEAREST": Image.Resampling.NEAREST,
            "BICUBIC": Image.Resampling.BICUBIC,
        }

        if interpolation_method not in self.interpolation_map:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

    @staticmethod
    def extract_visit_from_filename(filename: str, required_visits: List[str]) -> Optional[str]:
        """
        Extract visit code from filename.

        Args:
            filename: Filename like "002_S_0295_sc_optimal_axial_x95.nii.gz"
            required_visits: List of required visits

        Returns:
            Visit code like "sc" or None if not found
        """
        for visit in required_visits:
            if f"_{visit}_" in filename:
                return visit
        return None

    @staticmethod
    def extract_coordinate_position(filename: str, pattern: str) -> Optional[int]:
        """
        Extract coordinate position from filename.

        Args:
            filename: Filename like "002_S_0295_sc_optimal_axial_x95.nii.gz"
            pattern: Pattern to match (e.g., "_optimal_axial_x")

        Returns:
            Coordinate position (slice index) or None if not found
        """
        try:
            # The pattern is like: {subject}_{visit}_optimal_{plane}_x{number}.nii.gz
            # We need to extract the number after 'x'
            if "_x" in filename:
                coord_part = filename.split("_x")[1].split(".")[0]
                return int(coord_part)
            else:
                return None
        except (IndexError, ValueError):
            return None

    @staticmethod
    def extract_subject_id_from_filename(filename: str) -> Optional[str]:
        """
        Extract subject ID from temporal filename.

        Args:
            filename: Filename like "002_S_0295_sc_optimal_axial_x95.nii.gz"

        Returns:
            Subject ID like "002_S_0295" or None if not found
        """
        clean_name = filename.replace(".nii.gz", "").replace(".nii", "").replace(".png", "")
        parts = clean_name.split("_")

        for i in range(len(parts) - 2):
            if parts[i + 1] == "S" and parts[i].isdigit() and parts[i + 2].isdigit():
                return f"{parts[i]}_S_{parts[i+2]}"
        return None

    def normalize_image(self, data: np.ndarray, p_range: Tuple[int, int]) -> np.ndarray:
        """
        Normalize image intensity for PNG conversion.

        Args:
            data: 2D numpy array with image data
            p_range: Percentile range (low, high)

        Returns:
            Normalized uint8 array (0-255)
        """
        p_low, p_high = np.percentile(data, p_range)
        data = np.clip(data, p_low, p_high)
        data = (data - p_low) / (p_high - p_low + 1e-8)
        return (data * 255).astype(np.uint8)

    def resize_to_target(self, img: Image.Image) -> Image.Image:
        """
        Resize PIL Image to target size using configured interpolation.

        Args:
            img: PIL Image to resize

        Returns:
            Resized PIL Image
        """
        interpolation = self.interpolation_map[self.interpolation_method]
        return img.resize(self.target_size, interpolation)

    def convert_nifti_to_png(
        self,
        nii_file: Path,
        output_path: Path,
        pattern: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Convert a single NIfTI file to PNG.

        Args:
            nii_file: Path to input NIfTI file
            output_path: Path to output PNG file
            pattern: Pattern to extract coordinate position

        Returns:
            Tuple of (success, error_message, original_size_string)
        """
        if not NIBABEL_AVAILABLE:
            return False, "nibabel is required for NIfTI operations", None

        if not PIL_AVAILABLE:
            return False, "PIL/Pillow is required for PNG conversion", None

        try:
            # Extract coordinate position from filename
            coord_pos = self.extract_coordinate_position(nii_file.name, pattern)
            if coord_pos is None:
                return False, f"Could not extract coordinate position from filename: {nii_file.name}", None

            # Load the NIfTI file (already a single 2D slice)
            slice_img = nib.load(str(nii_file))
            slice_data = slice_img.get_fdata()

            # Remove singleton dimensions to get 2D slice
            slice_2d = np.squeeze(slice_data)

            if slice_2d.ndim != 2:
                return False, f"Expected 2D slice, got {slice_2d.ndim}D data", None

            # Normalize slice
            slice_normalized = self.normalize_image(slice_2d, self.intensity_percentile)

            # Convert to PIL Image (transpose for correct orientation)
            img = Image.fromarray(slice_normalized.T, mode="L")

            # Track original size if requested
            original_size = None
            if self.preserve_original_size:
                original_size = f"{img.size[0]}×{img.size[1]}"

            # Resize to target resolution
            img_resized = self.resize_to_target(img)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save PNG
            img_resized.save(output_path)

            # Verify output if enabled
            if self.verify_outputs:
                if not output_path.exists() or output_path.stat().st_size == 0:
                    return False, "Output PNG file is empty or missing", original_size

            return True, None, original_size

        except Exception as e:
            return False, f"Error processing {nii_file.name}: {str(e)}", None

    def scan_input_files(
        self,
        input_root: Path,
        slice_type: str,
        pattern: str
    ) -> Tuple[List[Tuple[str, str, str, str, Path]], Dict[str, Dict[str, int]]]:
        """
        Scan input directory for NIfTI files to convert.

        Args:
            input_root: Root directory containing slice type subdirectories
            slice_type: Name of slice type (e.g., "axial")
            pattern: File pattern to match

        Returns:
            Tuple of:
            - tasks: List of (split, group, subject_id, visit, nii_file) tuples
            - subject_stats: Dictionary of subject counts by split and group
        """
        tasks = []
        subject_stats = defaultdict(lambda: defaultdict(int))

        for split in self.splits:
            split_dir = input_root / slice_type / split
            if not split_dir.exists():
                continue

            for group in self.groups:
                group_dir = split_dir / group
                if not group_dir.exists():
                    continue

                # Process each subject directory
                subject_dirs = [d for d in group_dir.iterdir() if d.is_dir()]

                for subj_dir in subject_dirs:
                    subject_id = subj_dir.name

                    # Find NIfTI files in this subject directory
                    nii_files = list(subj_dir.glob(f"*{pattern}*.nii*"))

                    for nii_file in nii_files:
                        # Extract visit from filename
                        visit = self.extract_visit_from_filename(nii_file.name, self.required_visits)
                        if visit:
                            tasks.append((split, group, subject_id, visit, nii_file))

                subject_stats[split][group] = len(subject_dirs)

        return tasks, dict(subject_stats)

    def process_slice_type(
        self,
        slice_type: str,
        input_root: Path,
        output_root: Path,
        pattern: str
    ) -> Dict:
        """
        Process a single slice type (axial, coronal, sagittal).

        Args:
            slice_type: Name of slice type (e.g., "axial")
            input_root: Root directory for input files
            output_root: Root directory for output PNG files
            pattern: File pattern to match

        Returns:
            Dictionary with processing statistics
        """
        stats = defaultdict(lambda: defaultdict(int))
        errors = []
        original_sizes = set()
        visit_counts = defaultdict(int)
        processed_subjects = defaultdict(lambda: defaultdict(set))

        # Scan for input files
        tasks, subject_stats = self.scan_input_files(input_root, slice_type, pattern)

        if not tasks:
            return {
                "stats": dict(stats),
                "errors": errors,
                "original_sizes": list(original_sizes),
                "visit_counts": dict(visit_counts),
                "processed_subjects": {},
                "subject_stats": subject_stats,
                "total_tasks": 0
            }

        # Process each task
        for split, group, subject_id, visit, nii_file in tasks:
            # Create output filename: {subject_id}_{visit}_{slice_type}_x{position}.png
            coord_pos = self.extract_coordinate_position(nii_file.name, pattern)
            if coord_pos is None:
                errors.append(f"Subject {subject_id}: Could not extract coordinate from {nii_file.name}")
                stats[group]["coord_parse_error"] += 1
                continue

            out_name = f"{subject_id}_{visit}_{slice_type}_x{coord_pos}.png"
            out_path = output_root / slice_type / split / group / out_name

            # Skip if already exists (resume capability)
            if out_path.exists() and out_path.stat().st_size > 0:
                stats[group]["skipped"] += 1
                processed_subjects[split][group].add(subject_id)
                visit_counts[visit] += 1
                continue

            # Convert NIfTI to PNG
            success, error_msg, original_size = self.convert_nifti_to_png(
                nii_file, out_path, pattern
            )

            if success:
                stats[group]["saved"] += 1
                visit_counts[visit] += 1
                processed_subjects[split][group].add(subject_id)

                if original_size:
                    original_sizes.add(original_size)
            else:
                stats[group]["load_error"] += 1
                errors.append(f"Subject {subject_id}: {error_msg}")

        # Count unique subjects processed
        processed_subjects_dict = {}
        for split in self.splits:
            processed_subjects_dict[split] = {}
            for group in self.groups:
                processed_subjects_dict[split][group] = len(processed_subjects.get(split, {}).get(group, set()))

        return {
            "stats": dict(stats),
            "errors": errors,
            "original_sizes": sorted(list(original_sizes)) if original_sizes else [],
            "visit_counts": dict(visit_counts),
            "processed_subjects": processed_subjects_dict,
            "subject_stats": subject_stats,
            "total_tasks": len(tasks)
        }

