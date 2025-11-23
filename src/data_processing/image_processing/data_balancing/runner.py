"""Data balancing runner for image processing stage."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..formatter import ImageProcessingFormatter
from .processor import (
    DataBalancingProcessor,
    convert_numpy_types,
    generate_alphabetical_id,
    get_subject_id,
    get_timepoint
)


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for data balancing substage.

    Args:
        action: Action to perform (test/process)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    bal_cfg = img_cfg.get("data_balancing", {})

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
        formatter.error(f"Unknown action for data balancing: {action}")
        formatter.footer(exit_code=2)
        return 2


def run_test(cfg: Dict, formatter: ImageProcessingFormatter) -> int:
    """
    Run test mode - balance a few sample classes to verify setup.
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    bal_cfg = img_cfg.get("data_balancing", {})
    test_cfg = bal_cfg.get("test", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("test", "data_balancing", samples=str(test_cfg.get("num_samples", 2)))

    # Setup paths
    input_dir = output_root / bal_cfg.get("input_dir", "7_enhanced")
    output_dir = output_root / bal_cfg.get("output_dir", "8_balanced")

    # Get parameters
    slice_types = bal_cfg.get("slice_types", ["axial", "coronal", "sagittal"])
    groups = bal_cfg.get("groups", ["AD", "CN", "MCI"])
    required_visits = bal_cfg.get("required_visits", ["sc", "m06", "m12"])
    augmentation_targets = bal_cfg.get("augmentation_targets", {})
    augmentation_params = bal_cfg.get("augmentation_params", {})
    seed = cfg.get("seed", 42)

    # For test mode, use smaller targets
    test_targets = {}
    for group in groups:
        if group in augmentation_targets:
            current = augmentation_targets[group].get("current", 0)
            target = min(
                current + test_cfg.get("num_samples", 2),
                augmentation_targets[group].get("target", current)
            )
            test_targets[group] = {"current": current, "target": target}

    # Initialize processor
    processor = DataBalancingProcessor(
        slice_types=slice_types,
        groups=groups,
        required_visits=required_visits,
        augmentation_targets=test_targets,
        augmentation_params=augmentation_params,
        seed=seed,
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "cause": "; ".join(errors),
            "next_steps": "Install missing packages: pip install opencv-python numpy scipy"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("All required libraries available")

    # Check input directory
    if not input_dir.exists():
        formatter.error("Input directory not found", {
            "cause": f"Input directory does not exist: {input_dir}",
            "next_steps": "Run image_enhancement stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Process only first slice type for test
    test_slice_type = slice_types[0] if slice_types else "axial"
    formatter.info(f"Processing test augmentation for {test_slice_type} slice type")

    augmentation_log = {}
    stats_dict = {"augmented": 0, "copied": 0, "errors": 0}

    # Check which groups have data available
    missing_groups_test = []
    for group in groups:
        if group not in test_targets:
            continue
        input_class_path = input_dir / test_slice_type / "train" / group
        if not input_class_path.exists():
            missing_groups_test.append(group)

    if missing_groups_test:
        formatter.warning(f"Missing input data for groups: {', '.join(missing_groups_test)}")
        formatter.info("These groups will be skipped as there are no source images to augment from.")

    try:
        for group in groups:
            if group not in test_targets:
                continue

            input_class_path = input_dir / test_slice_type / "train" / group
            output_class_path = output_dir / test_slice_type / "train" / group

            if not input_class_path.exists():
                continue

            result = processor.augment_class(
                input_class_path,
                output_class_path,
                group
            )

            stats_dict["augmented"] += result["augmented_subjects"]
            augmentation_log[test_slice_type] = {group: result}

    except Exception as e:
        formatter.error(f"Test augmentation failed: {e}")
        stats_dict["errors"] += 1

    # Show results
    formatter.info("Test Results")
    formatter.print(f"  • Augmented subjects: {stats_dict['augmented']}")
    formatter.print(f"  • Errors: {stats_dict['errors']}")

    # Store results in report
    formatter.report_data.update({
        "test_results": {
            "augmented_subjects": stats_dict["augmented"],
            "errors": stats_dict["errors"],
            "slice_type": test_slice_type
        }
    })

    if stats_dict["errors"] == 0:
        formatter.success("Test completed successfully!")
        formatter.next_steps([
            "Data balancing is working correctly",
            f"Run full processing with: python -m data_processing.cli image_processing process --substage data_balancing"
        ])
        exit_code = 0
    else:
        formatter.warning(f"Some errors occurred ({stats_dict['errors']})")
        exit_code = 1

    formatter.footer(exit_code=exit_code)
    return exit_code


def run_process(cfg: Dict, formatter: ImageProcessingFormatter) -> int:
    """
    Run full data balancing processing on all data.
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    bal_cfg = img_cfg.get("data_balancing", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("process", "data_balancing")

    # Setup paths
    input_dir = output_root / bal_cfg.get("input_dir", "7_enhanced")
    output_dir = output_root / bal_cfg.get("output_dir", "8_balanced")

    # Get parameters
    slice_types = bal_cfg.get("slice_types", ["axial", "coronal", "sagittal"])
    groups = bal_cfg.get("groups", ["AD", "CN", "MCI"])
    required_visits = bal_cfg.get("required_visits", ["sc", "m06", "m12"])
    augmentation_targets = bal_cfg.get("augmentation_targets", {})
    augmentation_params = bal_cfg.get("augmentation_params", {})
    seed = cfg.get("seed", 42)

    # Initialize processor
    processor = DataBalancingProcessor(
        slice_types=slice_types,
        groups=groups,
        required_visits=required_visits,
        augmentation_targets=augmentation_targets,
        augmentation_params=augmentation_params,
        seed=seed,
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "cause": "; ".join(errors),
            "next_steps": "Install missing packages: pip install opencv-python numpy scipy"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("All required libraries available")

    # Show configuration
    formatter.info("Configuration")
    formatter.print(f"  • Slice types: {', '.join(slice_types)}")
    formatter.print(f"  • Groups: {', '.join(groups)}")
    formatter.print("")

    # Check input directory
    if not input_dir.exists():
        formatter.error("Input directory not found", {
            "cause": f"Input directory does not exist: {input_dir}",
            "next_steps": "Run image_enhancement stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Process augmentation for training set
    start_time = time.time()
    augmentation_log = {}
    stats_dict = {
        "original_subjects": {},
        "augmented_subjects": {},
        "total_subjects": {},
        "copied_files": 0,
        "errors": 0
    }

    formatter.info("Processing training set augmentation...")

    # First, check which groups have input data available
    missing_groups = []
    available_groups = {}

    for group in groups:
        if group not in augmentation_targets:
            continue

        input_class_path = input_dir / "axial" / "train" / group
        if input_class_path.exists():
            subjects = processor.organize_subjects_by_class(input_class_path)
            available_groups[group] = subjects
        else:
            missing_groups.append(group)

    # Report missing groups and show augmentation targets
    if missing_groups:
        formatter.warning(f"Missing input data for groups: {', '.join(missing_groups)}")
        formatter.info("These groups will be skipped as there are no source images to augment from.")
        formatter.info("To process these groups, first run the image_enhancement stage with all groups configured.")

    # Show augmentation targets with actual counts
    formatter.info("Augmentation targets:")
    for group in groups:
        if group in augmentation_targets:
            target = augmentation_targets[group].get("target", 0)
            if group in available_groups:
                current = len(available_groups[group])
                formatter.print(f"  • {group}: {current} → {target} subjects")
            else:
                formatter.print(f"  • {group}: No input data (target: {target} subjects)")
    formatter.print("")

    # First pass: Generate augmentation params for axial plane (reference plane)
    # These params will be reused for other planes to ensure consistency
    axial_augmentation_params = {}
    for group, subjects in available_groups.items():
        current_count = len(subjects)
        target_count = augmentation_targets[group].get("target", current_count)
        subjects_to_augment = max(0, target_count - current_count)

        if subjects_to_augment > 0:
            if not NUMPY_AVAILABLE:
                formatter.error("NumPy is required for data balancing")
                return 1
            subject_ids = list(subjects.keys())
            selected_subjects = np.random.choice(
                subject_ids, size=subjects_to_augment, replace=True
            )

            axial_augmentation_params[group] = {}
            for i, source_subject in enumerate(selected_subjects):
                alpha_id = generate_alphabetical_id(i)
                axial_augmentation_params[group][alpha_id] = {
                    "source": source_subject,
                    "params": processor.generate_augmentation_params_for_class(),
                }

    # Process each plane
    for plane_idx, plane in enumerate(slice_types):
        formatter.info(f"Processing {plane.upper()} plane ({plane_idx + 1}/{len(slice_types)})")

        plane_log = {}

        for group in groups:
            if group not in augmentation_targets:
                continue

            input_class_path = input_dir / plane / "train" / group
            output_class_path = output_dir / plane / "train" / group

            if not input_class_path.exists():
                # Skip silently as we already reported missing groups at the start
                continue

            try:
                # Use augmentation params from axial plane for consistency
                augmentation_params_dict = axial_augmentation_params.get(group, {})

                result = processor.augment_class(
                    input_class_path,
                    output_class_path,
                    group,
                    augmentation_params_dict
                )

                plane_log[group] = result

                # Update statistics (only count once per group, not per plane)
                if plane == "axial":  # Only count from axial plane to avoid double counting
                    if group not in stats_dict["original_subjects"]:
                        stats_dict["original_subjects"][group] = 0
                        stats_dict["augmented_subjects"][group] = 0
                        stats_dict["total_subjects"][group] = 0

                    stats_dict["original_subjects"][group] = result["original_subjects"]
                    stats_dict["augmented_subjects"][group] = result["augmented_subjects"]
                    stats_dict["total_subjects"][group] = result["total_subjects"]

            except Exception as e:
                formatter.error(f"Failed to augment {group} in {plane}: {e}")
                if formatter.verbose:
                    import traceback
                    formatter.print(traceback.format_exc())
                stats_dict["errors"] += 1

        augmentation_log[plane] = plane_log

    # Copy validation and test sets
    formatter.info("Copying validation and test sets...")
    copy_stats = processor.copy_non_train_sets(input_dir, output_dir)
    stats_dict["copied_files"] = copy_stats["copied"]

    processing_time = time.time() - start_time

    # Save augmentation log
    log_path = output_dir / "augmentation_log.json"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(convert_numpy_types(augmentation_log), f, indent=2)
        formatter.info(f"Augmentation log saved to {log_path}")
    except Exception as e:
        formatter.warning(f"Could not save augmentation log: {e}")

    # Show results
    formatter.info("Augmentation Results")
    for group in groups:
        if group in stats_dict["original_subjects"]:
            orig = stats_dict["original_subjects"][group]
            aug = stats_dict["augmented_subjects"][group]
            total = stats_dict["total_subjects"][group]
            formatter.print(f"  {group}: {orig} original + {aug} augmented = {total} total subjects")

    formatter.info(f"Copied {stats_dict['copied_files']} files from validation/test sets")

    # Create visualization if requested
    viz_cfg = bal_cfg.get("visualization", {})
    if viz_cfg.get("enabled", True):
        from .visualize import create_augmentation_effects_visualization
        viz_dir = output_root / ".visualizations" / "image_processing" / "data_balancing"
        viz_paths = create_augmentation_effects_visualization(
            input_dir,
            output_dir,
            slice_types,
            groups,
            required_visits,
            augmentation_log,
            viz_dir,
            formatter
        )
        if viz_paths:
            formatter.info(f"Saved {len(viz_paths)} visualizations to {viz_dir}")

    # Store results in report
    formatter.report_data.update({
        "parameters": {
            "slice_types": slice_types,
            "groups": groups,
            "augmentation_targets": augmentation_targets
        },
        "results": {
            "original_subjects": stats_dict["original_subjects"],
            "augmented_subjects": stats_dict["augmented_subjects"],
            "total_subjects": stats_dict["total_subjects"],
            "copied_files": stats_dict["copied_files"],
            "errors": stats_dict["errors"],
            "processing_time_seconds": round(processing_time, 1),
            "processing_time_minutes": round(processing_time / 60, 1)
        },
        "augmentation_log_path": str(log_path)
    })

    # Next steps
    if stats_dict["errors"] == 0:
        formatter.next_steps([
            f"Verify balanced files in: {output_dir}",
            f"Check augmentation log: {log_path}",
            "Dataset is now balanced and ready for training"
        ])
        exit_code = 0
    else:
        formatter.next_steps([
            f"Review errors: {stats_dict['errors']} classes failed",
            "Check error details in verbose mode"
        ])
        exit_code = 1

    formatter.footer(exit_code=exit_code)
    return exit_code

