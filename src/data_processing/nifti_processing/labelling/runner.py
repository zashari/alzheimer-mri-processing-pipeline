"""Labelling implementation for NIfTI processing stage."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..formatter import NiftiFormatter
from .processor import LabellingProcessor
from .visualize import create_visualizations


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for labelling substage.

    Args:
        action: Action to perform (test/process)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get configuration
    label_cfg = cfg.get("nifti_processing", {}).get("labelling", {})

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
        formatter.error(f"Unknown action for labelling: {action}")
        formatter.footer(exit_code=2)
        return 2


def run_test(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run test mode - process a few sample subjects to verify setup.
    """
    # Get configuration
    label_cfg = cfg.get("nifti_processing", {}).get("labelling", {})
    test_cfg = label_cfg.get("test", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("test", "labelling", samples=str(test_cfg.get("num_subjects", 3)))

    # Setup paths
    metadata_csv = output_root / label_cfg.get("metadata_csv", "manifests/metadata_split.csv")
    input_dir = output_root / label_cfg.get("input_dir", "3_optimal_slices")
    output_dir = output_root / label_cfg.get("output_dir", "4_labelling")

    # Initialize processor
    processor = LabellingProcessor(
        metadata_csv=metadata_csv,
        required_visits=label_cfg.get("required_visits", ["sc", "m06", "m12"]),
        groups=label_cfg.get("groups", ["AD", "CN"]),
        splits=label_cfg.get("splits", ["train", "val", "test"]),
        duplicate_strategy=label_cfg.get("duplicate_strategy", "largest"),
        remove_empty_files=label_cfg.get("remove_empty_files", True),
        verify_copies=label_cfg.get("verify_copies", True),
        verbose=cfg.get("debug", False)
    )

    # Load metadata
    success, error_msg = processor.load_metadata()
    if not success:
        formatter.error("Failed to load metadata", {
            "cause": error_msg,
            "next_steps": f"Check metadata CSV path: {metadata_csv}"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success(f"Metadata loaded: {len(processor.subject_to_group)} subjects")
    formatter.success(f"Groups: {', '.join(processor.groups)}")
    formatter.success(f"Required visits: {', '.join(processor.required_visits)}")

    # Check input directories
    slice_types = label_cfg.get("slice_types", {})
    missing_dirs = []
    for slice_type, config in slice_types.items():
        if not config.get("enabled", True):
            continue
        slice_root = input_dir / slice_type
        if not slice_root.exists():
            missing_dirs.append(str(slice_root))

    if missing_dirs:
        formatter.error("Input directories not found", {
            "cause": f"Missing {len(missing_dirs)} input directories",
            "next_steps": "Run template_registration stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Process test subjects (limit to num_subjects)
    num_subjects = test_cfg.get("num_subjects", 3)
    formatter.info(f"Processing {num_subjects} test subjects...")

    # Process first enabled slice type only for test
    test_results = {}
    for slice_type, config in slice_types.items():
        if not config.get("enabled", True):
            continue

        slice_root = input_dir / slice_type
        pattern = config.get("pattern", f"*_optimal_{slice_type}_x*.nii.gz")
        output_base = output_dir

        result = processor.process_slice_type(
            slice_type, slice_root, pattern, output_base
        )

        test_results[slice_type] = result

        # Show results for this slice type
        stats = result["stats"]
        formatter.info(f"{slice_type.upper()} Results")
        formatter.print(f"  Successful subjects: {stats.get('successful_subjects', 0)}")
        formatter.print(f"  Files organized: {sum(stats.get((s, g), 0) for s in processor.splits for g in processor.groups)}")

        # Limit to num_subjects for test
        if stats.get('successful_subjects', 0) >= num_subjects:
            break

    # Create visualization if requested
    if test_cfg.get("save_visualization", True):
        viz_cfg = label_cfg.get("visualization", {})
        if viz_cfg.get("enabled", True):
            viz_dir = output_root / ".visualizations" / "nifti_processing" / "labelling" / "test"
            create_visualizations(test_results, viz_dir, formatter, is_test=True)

    # Summary
    total_subjects = sum(r["stats"].get("successful_subjects", 0) for r in test_results.values())
    if total_subjects > 0:
        formatter.success(f"Test completed: {total_subjects} subjects processed")
        formatter.next_steps([
            "Labelling is working correctly",
            f"Run full processing with: python -m data_processing.cli nifti_processing process --substage labelling"
        ])
        exit_code = 0
    else:
        formatter.warning("No subjects processed in test mode")
        exit_code = 1

    # Store results in report
    formatter.report_data.update({
        "test_results": {
            slice_type: {
                "stats": r["stats"],
                "unmatched": len(r["unmatched_subjects"]),
                "unknown_groups": len(r["unknown_groups"])
            }
            for slice_type, r in test_results.items()
        }
    })

    formatter.footer(exit_code=exit_code)
    return exit_code


def run_process(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run full labelling on all data.
    """
    # Get configuration
    label_cfg = cfg.get("nifti_processing", {}).get("labelling", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("process", "labelling")

    # Setup paths
    metadata_csv = output_root / label_cfg.get("metadata_csv", "manifests/metadata_split.csv")
    input_dir = output_root / label_cfg.get("input_dir", "3_optimal_slices")
    output_dir = output_root / label_cfg.get("output_dir", "4_labelling")

    # Initialize processor
    processor = LabellingProcessor(
        metadata_csv=metadata_csv,
        required_visits=label_cfg.get("required_visits", ["sc", "m06", "m12"]),
        groups=label_cfg.get("groups", ["AD", "CN"]),
        splits=label_cfg.get("splits", ["train", "val", "test"]),
        duplicate_strategy=label_cfg.get("duplicate_strategy", "largest"),
        remove_empty_files=label_cfg.get("remove_empty_files", True),
        verify_copies=label_cfg.get("verify_copies", True),
        verbose=cfg.get("debug", False)
    )

    # Load metadata
    success, error_msg = processor.load_metadata()
    if not success:
        formatter.error("Failed to load metadata", {
            "cause": error_msg,
            "next_steps": f"Check metadata CSV path: {metadata_csv}"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success(f"Metadata loaded: {len(processor.subject_to_group)} subjects")

    # Show metadata distribution
    from collections import Counter
    group_dist = Counter(processor.subject_to_group.values())
    split_dist = Counter(processor.subject_to_split.values())
    formatter.print(f"  Group distribution: {dict(group_dist)}")
    formatter.print(f"  Split distribution: {dict(split_dist)}")

    # Check input directories
    slice_types = label_cfg.get("slice_types", {})
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
            "next_steps": "Run template_registration stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Show configuration
    formatter.info("Configuration")
    formatter.print(f"  Slice types: {', '.join(enabled_slice_types)}")
    formatter.print(f"  Groups: {', '.join(processor.groups)}")
    formatter.print(f"  Required visits: {', '.join(processor.required_visits)}")
    formatter.print(f"  Duplicate strategy: {processor.duplicate_strategy}")
    formatter.print(f"  Remove empty files: {processor.remove_empty_files}")
    formatter.print(f"  Verify copies: {processor.verify_copies}")

    # Process each slice type
    all_results = {}
    all_duplicates = {}
    all_empty_files = {}
    all_unmatched_subjects = set()
    all_unknown_groups = []
    grand_total_files = 0

    start_time = time.time()

    with formatter.create_progress_bar() as progress:
        task = progress.add_task("Processing slice types", total=len(enabled_slice_types))

        for slice_type in enabled_slice_types:
            config = slice_types[slice_type]
            slice_root = input_dir / slice_type
            pattern = config.get("pattern", f"*_optimal_{slice_type}_x*.nii.gz")
            output_base = output_dir

            progress.update(task, description=f"Processing {slice_type}")

            result = processor.process_slice_type(
                slice_type, slice_root, pattern, output_base
            )

            all_results[slice_type] = result
            all_unmatched_subjects.update(result["unmatched_subjects"])
            all_unknown_groups.extend(result["unknown_groups"])

            if result["duplicates"]:
                all_duplicates[slice_type] = result["duplicates"]
            if result["empty_files"]:
                all_empty_files[slice_type] = result["empty_files"]

            # Calculate files for this slice type
            slice_files = sum(
                result["temporal_stats"].get(split, {}).get(group, 0)
                for split in processor.splits
                for group in processor.groups
            )
            grand_total_files += slice_files

            progress.update(task, advance=1)

    processing_time = time.time() - start_time

    # Show results summary
    formatter.print()
    formatter.info("Processing Complete")

    # Overall statistics
    total_successful_subjects = sum(
        r["stats"].get("successful_subjects", 0) for r in all_results.values()
    )
    total_incomplete = sum(
        r["stats"].get("incomplete_sequence", 0) for r in all_results.values()
    )

    formatter.print(f"  Successful subjects: {total_successful_subjects}")
    formatter.print(f"  Incomplete sequences: {total_incomplete}")
    formatter.print(f"  Unmatched subjects: {len(all_unmatched_subjects)}")
    formatter.print(f"  Unknown groups: {len(all_unknown_groups)}")
    formatter.print(f"  Total files organized: {grand_total_files}")
    formatter.print(f"  Processing time: {processing_time/60:.1f} minutes")

    # Show duplicates summary
    if all_duplicates:
        formatter.warning(f"Duplicate files detected: {sum(len(d) for d in all_duplicates.values())} cases")
        if formatter.verbose:
            for slice_type, dups in all_duplicates.items():
                formatter.print(f"  {slice_type}: {len(dups)} duplicate cases")

    # Show empty files summary
    if all_empty_files:
        total_empty = sum(len(files) for files in all_empty_files.values())
        formatter.warning(f"Empty files found: {total_empty}")
        if processor.remove_empty_files:
            formatter.print(f"  Empty files will be removed")

    # Show detailed breakdown by slice type
    formatter.print()
    formatter.info("File Distribution by Slice Type, Split, and Group")
    for slice_type in enabled_slice_types:
        result = all_results[slice_type]
        temporal_stats = result["temporal_stats"]
        processed_subjects = result["processed_subjects"]

        formatter.print(f"\n  {slice_type.upper()}:")
        for split in processor.splits:
            split_total = sum(temporal_stats.get(split, {}).get(g, 0) for g in processor.groups)
            if split_total > 0:
                formatter.print(f"    {split}:")
                for group in processor.groups:
                    file_count = temporal_stats.get(split, {}).get(group, 0)
                    subj_count = processed_subjects.get(split, {}).get(group, 0)
                    if subj_count > 0:
                        avg_files = file_count / subj_count if subj_count > 0 else 0
                        formatter.print(f"      {group}: {subj_count} subjects, {file_count} files (avg {avg_files:.1f} files/subject)")

    # Create visualizations
    viz_cfg = label_cfg.get("visualization", {})
    if viz_cfg.get("enabled", True) and total_successful_subjects > 0:
        viz_dir = output_root / ".visualizations" / "nifti_processing" / "labelling"
        create_visualizations(all_results, viz_dir, formatter, is_test=False)

    # Store results in report
    formatter.report_data.update({
        "parameters": {
            "slice_types": enabled_slice_types,
            "groups": processor.groups,
            "required_visits": processor.required_visits,
            "duplicate_strategy": processor.duplicate_strategy
        },
        "results": {
            "total_successful_subjects": total_successful_subjects,
            "total_incomplete_sequences": total_incomplete,
            "total_unmatched_subjects": len(all_unmatched_subjects),
            "total_unknown_groups": len(all_unknown_groups),
            "total_files_organized": grand_total_files,
            "processing_time_minutes": round(processing_time / 60, 1),
            "by_slice_type": {
                slice_type: {
                    "stats": r["stats"],
                    "temporal_stats": r["temporal_stats"],
                    "processed_subjects": r["processed_subjects"]
                }
                for slice_type, r in all_results.items()
            }
        }
    })

    # Next steps
    if total_successful_subjects > 0:
        formatter.next_steps([
            f"Verify labelled files in: {output_dir}",
            f"Structure: {output_dir}/{{slice_type}}/{{split}}/{{group}}/{{subject_id}}/",
            "Proceed to next substage: twoD_conversion"
        ])
        exit_code = 0
    else:
        formatter.warning("No subjects were successfully processed")
        exit_code = 1

    formatter.footer(exit_code=exit_code)
    return exit_code

