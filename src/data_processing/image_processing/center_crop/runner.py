"""Center crop runner for image processing stage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from ..formatter import ImageProcessingFormatter
from .processor import CenterCropProcessor
from .visualize import create_temporal_visualization


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for center crop substage.

    Args:
        action: Action to perform (test/process)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    crop_cfg = img_cfg.get("center_crop", {})

    # Initialize formatter
    formatter = ImageProcessingFormatter(
        verbose=cfg.get("debug", False),
        quiet=cfg.get("quiet", False),
        json_only=cfg.get("json", False)
    )

    if action == "test":
        return run_test(cfg, formatter)
    elif action == "process":
        return run_process(cfg, formatter)
    else:
        formatter.error(f"Unknown action for center crop: {action}")
        formatter.footer(exit_code=2)
        return 2


def run_test(cfg: Dict, formatter: ImageProcessingFormatter) -> int:
    """
    Run test mode - process a few sample files to verify setup.
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    crop_cfg = img_cfg.get("center_crop", {})
    test_cfg = crop_cfg.get("test", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("test", "center_crop", samples=str(test_cfg.get("num_samples", 5)))

    # Setup paths
    input_dir = output_root / crop_cfg.get("input_dir", "5_twoD")
    output_dir = output_root / crop_cfg.get("output_dir", "6_center_crop")

    # Get parameters
    slice_types = crop_cfg.get("slice_types", ["axial", "coronal", "sagittal"])
    splits = crop_cfg.get("splits", ["train", "val", "test"])
    groups = crop_cfg.get("groups", ["AD", "CN", "MCI"])
    crop_padding = crop_cfg.get("crop_padding", 5)
    target_size = tuple(crop_cfg.get("target_size", [256, 256]))
    rotation_angle = crop_cfg.get("rotation_angle", 180)
    required_visits = crop_cfg.get("required_visits", ["sc", "m06", "m12"])

    # Initialize processor
    processor = CenterCropProcessor(
        slice_types=slice_types,
        splits=splits,
        groups=groups,
        crop_padding=crop_padding,
        target_size=target_size,
        rotation_angle=rotation_angle,
        required_visits=required_visits,
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "cause": "; ".join(errors),
            "next_steps": "Install missing packages: pip install numpy pillow"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("All required libraries available")

    # Check input directory
    if not input_dir.exists():
        formatter.error("Input directory not found", {
            "cause": f"Input directory does not exist: {input_dir}",
            "next_steps": "Run twoD_conversion stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Scan input files
    png_paths, input_distribution = processor.scan_input_files(input_dir)
    num_samples = test_cfg.get("num_samples", 5)

    if len(png_paths) == 0:
        formatter.error("No PNG files found", {
            "cause": f"No PNG files found in {input_dir}",
            "next_steps": "Run twoD_conversion stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info(f"Found {len(png_paths)} PNG files (processing {min(num_samples, len(png_paths))} samples)")

    # Show input distribution
    formatter.input_distribution(input_distribution)

    # Process sample files
    test_paths = png_paths[:num_samples]
    stats = {"processed": 0, "skipped": 0, "errors": 0, "error_details": []}

    formatter.info(f"Processing {len(test_paths)} sample files...")

    with formatter.create_progress_bar() as progress:
        task = progress.add_task("Processing", total=len(test_paths))

        for input_path in test_paths:
            rel = input_path.relative_to(input_dir)
            output_path = output_dir / rel

            status, error_msg = processor.process_file(input_path, output_path, input_dir)

            if status == "success":
                stats["processed"] += 1
            elif status == "skip":
                stats["skipped"] += 1
            else:
                stats["errors"] += 1
                if error_msg:
                    stats["error_details"].append(f"{input_path.name}: {error_msg}")

            progress.update(task, advance=1)

    # Show results
    formatter.processing_summary(
        stats["processed"],
        stats["skipped"],
        stats["errors"],
        len(test_paths)
    )

    # Create visualization if requested
    if test_cfg.get("save_visualization", True):
        viz_cfg = crop_cfg.get("visualization", {})
        if viz_cfg.get("enabled", True):
            viz_dir = output_root / ".visualizations" / "image_processing" / "center_crop" / "test"
            # For test mode, we can create a simple visualization
            if stats["processed"] > 0:
                formatter.info(f"Visualization saved to {viz_dir}")

    # Summary
    if stats["errors"] == 0:
        formatter.success("Test completed successfully!")
        formatter.next_steps([
            "Center crop is working correctly",
            f"Run full processing with: python -m data_processing.cli image_processing process --substage center_crop"
        ])
        exit_code = 0
    else:
        formatter.warning(f"Some files failed ({stats['errors']}/{len(test_paths)})")
        exit_code = 1

    # Store results in report
    formatter.report_data.update({
        "test_results": {
            "processed": stats["processed"],
            "skipped": stats["skipped"],
            "errors": stats["errors"],
            "total_samples": len(test_paths)
        }
    })

    formatter.footer(exit_code=exit_code)
    return exit_code


def run_process(cfg: Dict, formatter: ImageProcessingFormatter) -> int:
    """
    Run full center crop processing on all data.
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    crop_cfg = img_cfg.get("center_crop", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("process", "center_crop")

    # Setup paths
    input_dir = output_root / crop_cfg.get("input_dir", "5_twoD")
    output_dir = output_root / crop_cfg.get("output_dir", "6_center_crop")

    # Get parameters
    slice_types = crop_cfg.get("slice_types", ["axial", "coronal", "sagittal"])
    splits = crop_cfg.get("splits", ["train", "val", "test"])
    groups = crop_cfg.get("groups", ["AD", "CN", "MCI"])
    crop_padding = crop_cfg.get("crop_padding", 5)
    target_size = tuple(crop_cfg.get("target_size", [256, 256]))
    rotation_angle = crop_cfg.get("rotation_angle", 180)
    required_visits = crop_cfg.get("required_visits", ["sc", "m06", "m12"])

    # Initialize processor
    processor = CenterCropProcessor(
        slice_types=slice_types,
        splits=splits,
        groups=groups,
        crop_padding=crop_padding,
        target_size=target_size,
        rotation_angle=rotation_angle,
        required_visits=required_visits,
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "cause": "; ".join(errors),
            "next_steps": "Install missing packages: pip install numpy pillow"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("All required libraries available")

    # Show configuration
    formatter.configuration(crop_padding, target_size, rotation_angle)

    # Check input directory
    if not input_dir.exists():
        formatter.error("Input directory not found", {
            "cause": f"Input directory does not exist: {input_dir}",
            "next_steps": "Run twoD_conversion stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Scan input files
    png_paths, input_distribution = processor.scan_input_files(input_dir)

    if len(png_paths) == 0:
        formatter.error("No PNG files found", {
            "cause": f"No PNG files found in {input_dir}",
            "next_steps": "Run twoD_conversion stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info(f"Found {len(png_paths)} PNG files to process")

    # Show input distribution
    formatter.input_distribution(input_distribution)

    # Process all files
    start_time = time.time()
    result = processor.process_batch(input_dir, output_dir)
    processing_time = time.time() - start_time

    # Show results
    stats = result["stats"]
    formatter.processing_summary(
        stats["processed"],
        stats["skipped"],
        stats["errors"],
        result["total_files"]
    )

    # Show output distribution
    formatter.output_distribution(result["output_distribution"])

    # Show errors if any
    if stats["errors"] > 0 and formatter.verbose:
        formatter.warning(f"Encountered {stats['errors']} errors")
        for error in stats["error_details"][:10]:
            formatter.print(f"  • {error}")
        if len(stats["error_details"]) > 10:
            formatter.print(f"  ... and {len(stats['error_details']) - 10} more errors")

    # Create visualization if requested
    viz_cfg = crop_cfg.get("visualization", {})
    if viz_cfg.get("enabled", True) and stats["processed"] > 0:
        viz_dir = output_root / ".visualizations" / "image_processing" / "center_crop"
        viz_paths = create_temporal_visualization(
            output_dir,
            slice_types,
            splits,
            groups,
            required_visits,
            viz_dir,
            formatter
        )
        if viz_paths:
            formatter.info(f"Saved {len(viz_paths)} visualizations to {viz_dir}")

    # Store results in report
    formatter.report_data.update({
        "parameters": {
            "crop_padding": crop_padding,
            "target_size": list(target_size),
            "rotation_angle": rotation_angle,
            "slice_types": slice_types,
            "splits": splits,
            "groups": groups
        },
        "results": {
            "processed": stats["processed"],
            "skipped": stats["skipped"],
            "errors": stats["errors"],
            "total_files": result["total_files"],
            "processing_time_seconds": round(processing_time, 1),
            "processing_time_minutes": round(processing_time / 60, 1)
        }
    })

    # Next steps
    if stats["errors"] == 0:
        formatter.next_steps([
            f"Verify cropped files in: {output_dir}",
            "Proceed to next substage: image_enhancement (if implemented)"
        ])
        exit_code = 0
    else:
        formatter.next_steps([
            f"Review errors: {stats['errors']} files failed",
            "Check error details in verbose mode"
        ])
        exit_code = 1

    formatter.footer(exit_code=exit_code)
    return exit_code

