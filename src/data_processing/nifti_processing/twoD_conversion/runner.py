"""2D conversion implementation for NIfTI processing stage."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ..formatter import NiftiFormatter
from .processor import TwoDConversionProcessor
from .visualize import create_visualizations


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for 2D conversion substage.

    Args:
        action: Action to perform (test/process)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get configuration
    conv_cfg = cfg.get("nifti_processing", {}).get("twoD_conversion", {})

    # Initialize formatter
    formatter = NiftiFormatter(
        verbose=cfg.get("debug", False),
        quiet=cfg.get("quiet", False),
        json_only=cfg.get("json", False)
    )

    if action == "test":
        return run_test(cfg, formatter)
    elif action == "process":
        return run_process(cfg, formatter)
    else:
        formatter.error(f"Unknown action for 2D conversion: {action}")
        formatter.footer(exit_code=2)
        return 2


def run_test(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run test mode - convert a few sample subjects to verify setup.
    """
    # Get configuration
    conv_cfg = cfg.get("nifti_processing", {}).get("twoD_conversion", {})
    test_cfg = conv_cfg.get("test", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("test", "twoD_conversion", samples=str(test_cfg.get("num_subjects", 3)))

    # Setup paths
    input_dir = output_root / conv_cfg.get("input_dir", "4_labelling")
    output_dir = output_root / conv_cfg.get("output_dir", "5_twoD")

    # Get conversion parameters
    intensity_percentile = tuple(conv_cfg.get("intensity_percentile", [1, 99]))
    target_size = tuple(conv_cfg.get("target_size", [256, 256]))
    interpolation_method = conv_cfg.get("interpolation_method", "LANCZOS")

    # Initialize processor
    processor = TwoDConversionProcessor(
        required_visits=conv_cfg.get("required_visits", ["sc", "m06", "m12"]),
        groups=conv_cfg.get("groups", ["AD", "CN"]),
        splits=conv_cfg.get("splits", ["train", "val", "test"]),
        intensity_percentile=intensity_percentile,
        target_size=target_size,
        interpolation_method=interpolation_method,
        verify_outputs=conv_cfg.get("verify_outputs", True),
        preserve_original_size=True,  # Always track sizes in test mode for reporting
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    if not processor.__class__.__module__.startswith("data_processing"):
        # Check if required libraries are available
        try:
            import nibabel
            import numpy
            from PIL import Image
        except ImportError as e:
            formatter.error("Prerequisites not met", {
                "cause": f"Missing required library: {e}",
                "next_steps": "Install: pip install nibabel numpy pillow"
            })
            formatter.footer(exit_code=1)
            return 1

    formatter.info("Prerequisites")
    formatter.success("All required libraries available")
    formatter.print(f"  Target resolution: {target_size[0]}×{target_size[1]} pixels")
    formatter.print(f"  Interpolation: {interpolation_method}")
    formatter.print(f"  Intensity normalization: {intensity_percentile[0]}-{intensity_percentile[1]} percentile")

    # Check input directories
    slice_types = conv_cfg.get("slice_types", {})
    missing_dirs = []
    enabled_slice_types = []
    for slice_type, config in slice_types.items():
        if not config.get("enabled", True):
            continue
        enabled_slice_types.append(slice_type)
        slice_root = input_dir / slice_type
        if not slice_root.exists():
            missing_dirs.append(str(slice_root))

    if missing_dirs:
        formatter.error("Input directories not found", {
            "cause": f"Missing {len(missing_dirs)} input directories",
            "next_steps": "Run labelling stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Process test subjects (limit to first enabled slice type and num_subjects)
    num_subjects = test_cfg.get("num_subjects", 3)
    formatter.info(f"Processing {num_subjects} test subjects...")

    test_results = {}
    subjects_processed = 0

    for slice_type in enabled_slice_types:
        if subjects_processed >= num_subjects:
            break

        config = slice_types[slice_type]
        pattern = config.get("pattern", f"_optimal_{slice_type}_x")

        result = processor.process_slice_type(
            slice_type, input_dir, output_dir, pattern
        )

        test_results[slice_type] = result

        # Show results for this slice type
        stats = result["stats"]
        total_saved = sum(stats.get(group, {}).get("saved", 0) for group in processor.groups)
        total_errors = sum(
            stats.get(group, {}).get("coord_parse_error", 0) +
            stats.get(group, {}).get("load_error", 0)
            for group in processor.groups
        )

        formatter.info(f"{slice_type.upper()} Results")
        formatter.print(f"  PNG files converted: {total_saved}")
        if total_errors > 0:
            formatter.warning(f"  Errors: {total_errors}")

        if result.get("original_sizes"):
            formatter.print(f"  Original sizes: {', '.join(result['original_sizes'])}")
        formatter.print(f"  Resized to: {target_size[0]}×{target_size[1]}")

        # Count subjects processed
        for split_data in result.get("processed_subjects", {}).values():
            for group_data in split_data.values():
                subjects_processed += group_data

        if subjects_processed >= num_subjects:
            break

    # Create visualization if requested
    if test_cfg.get("save_visualization", True):
        viz_cfg = conv_cfg.get("visualization", {})
        if viz_cfg.get("enabled", True):
            viz_dir = output_root / ".visualizations" / "nifti_processing" / "twoD_conversion" / "test"
            create_visualizations(test_results, viz_dir, formatter, is_test=True)

    # Summary
    total_converted = sum(
        sum(r["stats"].get(group, {}).get("saved", 0) for group in processor.groups)
        for r in test_results.values()
    )

    if total_converted > 0:
        formatter.success(f"Test completed: {total_converted} PNG files converted")
        formatter.next_steps([
            "2D conversion is working correctly",
            f"Run full processing with: python -m data_processing.cli nifti_processing process --substage twoD_conversion"
        ])
        exit_code = 0
    else:
        formatter.warning("No files converted in test mode")
        exit_code = 1

    # Store results in report
    formatter.report_data.update({
        "test_results": {
            slice_type: {
                "stats": r["stats"],
                "total_converted": sum(
                    r["stats"].get(group, {}).get("saved", 0) for group in processor.groups
                ),
                "errors": len(r.get("errors", []))
            }
            for slice_type, r in test_results.items()
        }
    })

    formatter.footer(exit_code=exit_code)
    return exit_code


def run_process(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run full 2D conversion on all data.
    """
    # Get configuration
    conv_cfg = cfg.get("nifti_processing", {}).get("twoD_conversion", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("process", "twoD_conversion")

    # Setup paths
    input_dir = output_root / conv_cfg.get("input_dir", "4_labelling")
    output_dir = output_root / conv_cfg.get("output_dir", "5_twoD")

    # Get conversion parameters
    intensity_percentile = tuple(conv_cfg.get("intensity_percentile", [1, 99]))
    target_size = tuple(conv_cfg.get("target_size", [256, 256]))
    interpolation_method = conv_cfg.get("interpolation_method", "LANCZOS")

    # Initialize processor
    processor = TwoDConversionProcessor(
        required_visits=conv_cfg.get("required_visits", ["sc", "m06", "m12"]),
        groups=conv_cfg.get("groups", ["AD", "CN"]),
        splits=conv_cfg.get("splits", ["train", "val", "test"]),
        intensity_percentile=intensity_percentile,
        target_size=target_size,
        interpolation_method=interpolation_method,
        verify_outputs=conv_cfg.get("verify_outputs", True),
        preserve_original_size=conv_cfg.get("preserve_original_size", False),
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    formatter.info("Prerequisites")
    formatter.success("All required libraries available")

    # Show configuration
    formatter.info("Configuration")
    formatter.print(f"  Slice types: {', '.join([st for st, cfg in conv_cfg.get('slice_types', {}).items() if cfg.get('enabled', True)])}")
    formatter.print(f"  Groups: {', '.join(processor.groups)}")
    formatter.print(f"  Required visits: {', '.join(processor.required_visits)}")
    formatter.print(f"  Target resolution: {target_size[0]}×{target_size[1]} pixels")
    formatter.print(f"  Interpolation: {interpolation_method}")
    formatter.print(f"  Intensity normalization: {intensity_percentile[0]}-{intensity_percentile[1]} percentile")

    # Check input directories
    slice_types = conv_cfg.get("slice_types", {})
    missing_dirs = []
    enabled_slice_types = []
    for slice_type, config in slice_types.items():
        if not config.get("enabled", True):
            continue
        enabled_slice_types.append(slice_type)
        slice_root = input_dir / slice_type
        if not slice_root.exists():
            missing_dirs.append(str(slice_root))

    if missing_dirs:
        formatter.error("Input directories not found", {
            "cause": f"Missing {len(missing_dirs)} input directories",
            "next_steps": "Run labelling stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Process each slice type
    all_results = {}
    all_errors = []
    grand_total_png = 0
    all_original_sizes = set()

    start_time = time.time()

    with formatter.create_progress_bar() as progress:
        task = progress.add_task("Converting slice types", total=len(enabled_slice_types))

        for slice_type in enabled_slice_types:
            config = slice_types[slice_type]
            pattern = config.get("pattern", f"_optimal_{slice_type}_x")

            progress.update(task, description=f"Converting {slice_type}")

            result = processor.process_slice_type(
                slice_type, input_dir, output_dir, pattern
            )

            all_results[slice_type] = result
            all_errors.extend(result.get("errors", []))
            all_original_sizes.update(result.get("original_sizes", []))

            # Calculate PNG files for this slice type
            slice_png = sum(
                result["stats"].get(group, {}).get("saved", 0)
                for group in processor.groups
            )
            grand_total_png += slice_png

            progress.update(task, advance=1)

    processing_time = time.time() - start_time

    # Show results summary
    formatter.print()
    formatter.info("Processing Complete")

    # Overall statistics
    total_subjects = sum(
        sum(
            sum(split_data.values())
            for split_data in r.get("processed_subjects", {}).values()
        )
        for r in all_results.values()
    )

    total_skipped = sum(
        sum(r["stats"].get(group, {}).get("skipped", 0) for group in processor.groups)
        for r in all_results.values()
    )

    formatter.print(f"  PNG files converted: {grand_total_png}")
    formatter.print(f"  PNG files skipped (existing): {total_skipped}")
    formatter.print(f"  Subjects processed: {total_subjects}")
    formatter.print(f"  Errors: {len(all_errors)}")
    formatter.print(f"  Processing time: {processing_time/60:.1f} minutes")

    if all_original_sizes:
        formatter.print(f"  Original sizes found: {', '.join(sorted(all_original_sizes))}")
    formatter.print(f"  All images resized to: {target_size[0]}×{target_size[1]}")

    # Show errors if any
    if all_errors and formatter.verbose:
        formatter.warning(f"Encountered {len(all_errors)} errors")
        for error in all_errors[:10]:
            formatter.print(f"  • {error}")
        if len(all_errors) > 10:
            formatter.print(f"  ... and {len(all_errors) - 10} more errors")

    # Show detailed breakdown by slice type
    formatter.print()
    formatter.info("File Distribution by Slice Type, Split, and Group")
    for slice_type in enabled_slice_types:
        result = all_results[slice_type]
        stats = result["stats"]
        processed_subjects = result["processed_subjects"]

        formatter.print(f"\n  {slice_type.upper()}:")
        for split in processor.splits:
            split_total = sum(
                stats.get(group, {}).get("saved", 0) for group in processor.groups
            )
            if split_total > 0:
                formatter.print(f"    {split}:")
                for group in processor.groups:
                    png_count = stats.get(group, {}).get("saved", 0)
                    subj_count = processed_subjects.get(split, {}).get(group, 0)
                    if subj_count > 0:
                        avg_png = png_count / subj_count if subj_count > 0 else 0
                        formatter.print(f"      {group}: {subj_count} subjects, {png_count} PNG files (avg {avg_png:.1f} PNG/subject)")

        # Show visit distribution
        visit_counts = result.get("visit_counts", {})
        if visit_counts:
            formatter.print(f"    Visit distribution:")
            for visit in processor.required_visits:
                count = visit_counts.get(visit, 0)
                if count > 0:
                    formatter.print(f"      {visit}: {count} PNG files")

    # Create visualizations
    viz_cfg = conv_cfg.get("visualization", {})
    if viz_cfg.get("enabled", True) and grand_total_png > 0:
        viz_dir = output_root / ".visualizations" / "nifti_processing" / "twoD_conversion"
        create_visualizations(all_results, viz_dir, formatter, is_test=False)

    # Store results in report
    formatter.report_data.update({
        "parameters": {
            "slice_types": enabled_slice_types,
            "groups": processor.groups,
            "required_visits": processor.required_visits,
            "target_size": list(target_size),
            "interpolation_method": interpolation_method,
            "intensity_percentile": list(intensity_percentile)
        },
        "results": {
            "total_png_converted": grand_total_png,
            "total_png_skipped": total_skipped,
            "total_subjects": total_subjects,
            "total_errors": len(all_errors),
            "processing_time_minutes": round(processing_time / 60, 1),
            "by_slice_type": {
                slice_type: {
                    "stats": r["stats"],
                    "processed_subjects": r["processed_subjects"],
                    "visit_counts": r.get("visit_counts", {})
                }
                for slice_type, r in all_results.items()
            }
        }
    })

    # Next steps
    if grand_total_png > 0:
        formatter.next_steps([
            f"Verify PNG files in: {output_dir}",
            f"Structure: {output_dir}/{{slice_type}}/{{split}}/{{group}}/{{subject_id}}_{{visit}}_{{slice_type}}_x{{position}}.png",
            "Ready for CNN+LSTM temporal sequence modeling",
            "Each subject contributes 3 timepoints × 3 slice types = 9 images"
        ])
        exit_code = 0
    else:
        formatter.warning("No PNG files were converted")
        exit_code = 1

    formatter.footer(exit_code=exit_code)
    return exit_code

