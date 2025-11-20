"""Center crop processor for image processing stage."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from PIL import Image
    NUMPY_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    PIL_AVAILABLE = False


class CenterCropProcessor:
    """Handles center cropping, resizing, and rotation of PNG images."""

    def __init__(
        self,
        slice_types: List[str],
        splits: List[str],
        groups: List[str],
        crop_padding: int = 5,
        target_size: Tuple[int, int] = (256, 256),
        rotation_angle: int = 180,
        required_visits: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """
        Initialize center crop processor.

        Args:
            slice_types: List of slice types to process (e.g., ["axial", "coronal", "sagittal"])
            splits: List of splits to process (e.g., ["train", "val", "test"])
            groups: List of groups to process (e.g., ["AD", "CN", "MCI"])
            crop_padding: Padding to add around brain bounding box (pixels)
            target_size: Target resolution for output images (width, height)
            rotation_angle: Rotation angle in degrees
            required_visits: Optional list of required visits for filtering
            verbose: Show detailed output
        """
        self.slice_types = slice_types
        self.splits = splits
        self.groups = groups
        self.crop_padding = crop_padding
        self.target_size = target_size
        self.rotation_angle = rotation_angle
        self.required_visits = required_visits or []
        self.verbose = verbose

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """
        Check if all prerequisites are met.

        Returns:
            (success, list of error messages)
        """
        errors = []

        if not NUMPY_AVAILABLE:
            errors.append("NumPy is required. Install with: pip install numpy")

        if not PIL_AVAILABLE:
            errors.append("Pillow is required. Install with: pip install pillow")

        return len(errors) == 0, errors

    def scan_input_files(self, input_root: Path) -> Tuple[List[Path], Dict[str, Dict[str, Dict[str, int]]]]:
        """
        Scan input directory for PNG files.

        Args:
            input_root: Root directory containing PNG files

        Returns:
            Tuple of:
            - List of PNG file paths
            - Distribution dictionary: {slice_type: {split: {group: count}}}
        """
        png_paths = list(input_root.rglob("*.png"))
        distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Organize by slice_type/split/group structure
        # Handle both flat structure ({slice_type}/{split}/{group}/*.png) 
        # and nested structure ({slice_type}/{split}/{group}/{subject}/*.png)
        for png_path in png_paths:
            try:
                # Extract structure from path
                rel_path = png_path.relative_to(input_root)
                parts = rel_path.parts

                # Find slice_type, split, group in path
                # Path can be: {slice_type}/{split}/{group}/{filename}.png
                # or: {slice_type}/{split}/{group}/{subject}/{filename}.png
                if len(parts) >= 3:
                    slice_type = parts[0]
                    split = parts[1]
                    group = parts[2]

                    if slice_type in self.slice_types and split in self.splits and group in self.groups:
                        distribution[slice_type][split][group] += 1
            except Exception:
                # Skip files that don't match expected structure
                continue

        return png_paths, dict(distribution)

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        input_root: Path
    ) -> Tuple[str, Optional[str]]:
        """
        Process a single PNG file: crop, resize, rotate.

        Args:
            input_path: Path to input PNG file
            output_path: Path for output PNG file
            input_root: Root input directory for relative path calculation

        Returns:
            Tuple of (status, error_message)
            Status can be: "success", "skip", "error"
        """
        # Check if output already exists
        if output_path.exists() and output_path.stat().st_size > 0:
            return "skip", None

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load as grayscale
            img = Image.open(input_path).convert("L")
            arr = np.array(img)

            # Find bright (brain) pixels
            ys, xs = np.where(arr > 0)

            if len(ys) == 0:
                # Nothing to crop: just resize, rotate, save
                img.resize(self.target_size, Image.Resampling.BILINEAR) \
                   .rotate(self.rotation_angle, expand=True) \
                   .save(output_path)
                return "success", None

            # Bounding box of the brain
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()

            # Add padding
            y0 = max(0, y0 - self.crop_padding)
            x0 = max(0, x0 - self.crop_padding)
            y1 = min(arr.shape[0], y1 + self.crop_padding)
            x1 = min(arr.shape[1], x1 + self.crop_padding)

            # Crop to brain + padding, then resize and rotate
            cropped = img.crop((x0, y0, x1, y1))
            resized = cropped.resize(self.target_size, Image.Resampling.BILINEAR)
            rotated = resized.rotate(self.rotation_angle, expand=True)

            # Save
            rotated.save(output_path)

            # Verify output
            if not output_path.exists() or output_path.stat().st_size == 0:
                return "error", "Output file is empty or missing"

            return "success", None

        except Exception as e:
            return "error", str(e)

    def process_batch(
        self,
        input_root: Path,
        output_root: Path
    ) -> Dict:
        """
        Process all PNG files in input directory.

        Args:
            input_root: Root directory for input PNG files
            output_root: Root directory for output PNG files

        Returns:
            Dictionary with processing statistics
        """
        # Scan input files
        png_paths, input_distribution = self.scan_input_files(input_root)

        # Process files
        stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "error_details": []
        }
        output_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for input_path in png_paths:
            try:
                # Compute relative path, mirror directory tree
                rel = input_path.relative_to(input_root)
                output_path = output_root / rel

                # Process file
                status, error_msg = self.process_file(input_path, output_path, input_root)

                if status == "success":
                    stats["processed"] += 1
                    # Update output distribution
                    parts = rel.parts
                    if len(parts) >= 3:
                        slice_type, split, group = parts[0], parts[1], parts[2]
                        if slice_type in self.slice_types and split in self.splits and group in self.groups:
                            output_distribution[slice_type][split][group] += 1
                elif status == "skip":
                    stats["skipped"] += 1
                    # Count skipped files in distribution
                    parts = rel.parts
                    if len(parts) >= 3:
                        slice_type, split, group = parts[0], parts[1], parts[2]
                        if slice_type in self.slice_types and split in self.splits and group in self.groups:
                            output_distribution[slice_type][split][group] += 1
                else:
                    stats["errors"] += 1
                    if error_msg:
                        stats["error_details"].append(f"{input_path.name}: {error_msg}")

            except Exception as e:
                stats["errors"] += 1
                stats["error_details"].append(f"{input_path.name}: {str(e)}")

        return {
            "stats": stats,
            "input_distribution": dict(input_distribution),
            "output_distribution": dict(output_distribution),
            "total_files": len(png_paths)
        }

