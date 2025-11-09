from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

from .metadata import ensure_columns, load_metadata, value_counts_summary
from .splitting import (
    subjects_with_complete_visits,
    analyze_visit_completeness,
    check_split_feasibility,
    stratified_split_with_redistribution,
    calculate_split_statistics,
    check_leakage,
    materialize_manifests
)
from .deduplication import deduplicate_scans, analyze_duplicates
from .file_operations import FileOperations
from .formatter import DataPrepFormatter
from ..utils.randomness import set_seed


REQUIRED_COLS = ["Subject", "Visit", "Group", "Acq Date"]


def _resolve_paths(cfg: Dict) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve data paths from configuration."""
    paths = cfg.get("paths", {})
    data_root = paths.get("data_root")
    output_root = paths.get("output_root")
    metadata_csv = paths.get("metadata_csv")
    return (
        Path(data_root) if data_root else None,
        Path(output_root) if output_root else None,
        Path(metadata_csv) if metadata_csv else None,
    )


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for data preparation stage.

    Args:
        action: One of 'analyze', 'split', 'manifests'
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Initialize formatter
    verbose = cfg.get("debug", False)
    quiet = cfg.get("quiet", False)
    json_only = cfg.get("json", False)
    formatter = DataPrepFormatter(verbose=verbose, quiet=quiet, json_only=json_only)

    # Extract configuration
    dry_run = bool(cfg.get("dry_run", False) or cfg.get("global", {}).get("dry_run", False))
    seed = cfg.get("seed") or cfg.get("global", {}).get("seed", 42)
    set_seed(seed)
    show_all = cfg.get("show_all", True)  # Default to True (show all items)

    # Resolve paths
    data_root, output_root, metadata_csv = _resolve_paths(cfg)
    if not data_root or not output_root or not metadata_csv:
        formatter.error(
            "Missing required paths",
            {
                "cause": "Configs missing paths.data_root, paths.output_root, or paths.metadata_csv",
                "next_steps": "Update configs/default.yaml or pass via --set"
            }
        )
        formatter.footer(exit_code=2)
        return 2

    # Get data_preparation config
    dp = cfg.get("data_preparation", {})
    required_visits = dp.get("required_visits", ["sc", "m06", "m12"])
    split_ratios = dp.get("split_ratios", [0.7, 0.15, 0.15])
    stratify_by = dp.get("stratify_by", "Group")
    shuffle = bool(dp.get("shuffle", True))
    prefer_scaled2 = bool(dp.get("prefer_scaled2", True))
    use_symlinks = bool(dp.get("use_symlinks", False))
    debug_mode = bool(dp.get("debug_mode", False))
    copy_nifti = bool(dp.get("copy_nifti_files", True))
    min_subjects = dp.get("min_subjects_per_group", 7)

    # Visualization settings
    viz = dp.get("visualization", {})
    viz_enabled = bool(viz.get("enabled", False))
    viz_samples = viz.get("show_samples", 3)

    # Load metadata
    if not metadata_csv.exists():
        formatter.error(
            f"Metadata CSV not found: {metadata_csv}",
            {"next_steps": "Check path in configs/default.yaml"}
        )
        formatter.footer(exit_code=2)
        return 2

    try:
        df = load_metadata(metadata_csv, parse_dates=["Acq Date"])
        ensure_columns(df, REQUIRED_COLS)
    except Exception as e:
        formatter.error(
            f"Failed to load metadata: {e}",
            {"next_steps": "Check CSV format and required columns"}
        )
        formatter.footer(exit_code=2)
        return 2

    # Check if Subject column exists
    if "Subject" not in df.columns:
        formatter.error(
            "Missing 'Subject' column in metadata",
            {"next_steps": "Check if CSV has correct column names. Expected: Subject, Visit, Group, Acq Date"}
        )
        formatter.footer(exit_code=2)
        return 2

    # Store initial subject count for retention analysis
    initial_subjects = df["Subject"].nunique()

    # Apply deduplication if enabled
    if prefer_scaled2 and "Description" in df.columns and "Subject" in df.columns and "Visit" in df.columns:
        # Analyze duplicates first
        dup_info = analyze_duplicates(df)
        if dup_info["total_duplicates"] > 0:
            df_before = len(df)
            df = deduplicate_scans(df, verbose=verbose)
            df_after = len(df)
            if not quiet and df_before != df_after:
                formatter.console.print(f"[green]✓[/green] Removed {df_before - df_after} duplicate scans (kept Scaled_2 where available)")

    # Execute action
    if action == "analyze":
        return run_analyze(df, data_root, formatter, viz_enabled, viz_samples, initial_subjects,
                          viz, output_root, show_all)

    elif action == "split":
        return run_split(
            df, data_root, output_root, formatter,
            required_visits, split_ratios, stratify_by,
            shuffle, seed, dry_run, copy_nifti,
            use_symlinks, debug_mode, min_subjects,
            initial_subjects
        )

    elif action == "manifests":
        # Simplified manifests action (no file copying)
        formatter.header("manifests")

        # Get complete subjects
        complete_subjects = subjects_with_complete_visits(df, required_visits)

        # Perform split
        split = stratified_split_with_redistribution(
            complete_subjects, df, stratify_by, tuple(split_ratios), seed, shuffle
        )

        # Calculate statistics
        total_subjects = len(complete_subjects)
        formatter.dataset_info(
            total_subjects,
            len(df),
            "MRI",
            str(output_root / "manifests")
        )

        formatter.split_results(
            len(split["train"]),
            len(split["val"]),
            len(split["test"]),
            total_subjects
        )

        if not dry_run:
            out_dir = output_root / "manifests"
            file_counts = materialize_manifests(df, split, required_visits, out_dir)
            formatter.files_summary(file_counts, str(out_dir))
        else:
            formatter.console.print("[yellow][DRY RUN][/yellow] No manifests written")

        formatter.footer(exit_code=0)
        return 0

    else:
        formatter.error(f"Unknown action: {action}")
        formatter.footer(exit_code=2)
        return 2


def run_analyze(df, data_root: Path, formatter: DataPrepFormatter,
               viz_enabled: bool, viz_samples: int, initial_subjects: int,
               viz_config: Dict, output_root: Path, show_all: bool = True) -> int:
    """Run the analyze action."""
    formatter.header("analyze")

    # Calculate overview metrics
    total_rows = len(df)
    total_subjects = df["Subject"].nunique()

    # Modality distribution (assuming MRI for ADNI)
    modalities = {"MRI": 100.0}  # Can be enhanced if Modality column exists

    formatter.overview(total_rows, total_subjects, modalities)

    # Group distribution
    if "Group" in df.columns:
        group_counts = df["Group"].value_counts()
        groups = {}
        for group, count in group_counts.items():
            pct = count / total_rows * 100
            groups[group] = (count, pct)
        formatter.group_distribution(groups)

    # Sex distribution
    if "Sex" in df.columns:
        sex_counts = df["Sex"].value_counts()
        sex_dist = {}
        for sex, count in sex_counts.items():
            pct = count / total_rows * 100
            sex_dist[sex] = (count, pct)
        formatter.sex_distribution(sex_dist)

    # Visit distribution
    if "Visit" in df.columns:
        visit_counts = df["Visit"].value_counts().to_dict()
        formatter.visit_distribution(visit_counts)

    # Description summary
    if "Description" in df.columns:
        desc_counts = df["Description"].value_counts()
        descriptions = [(desc, count) for desc, count in desc_counts.items()]
        formatter.description_summary(descriptions, show_all=show_all)

    # Validation summary
    rare_categories = 0
    if "Group" in df.columns:
        group_counts = df["Group"].value_counts()
        total = group_counts.sum()
        rare_categories = sum(1 for count in group_counts.values if count / total < 0.005)

    missing_values = df.isnull().sum().sum()

    # Calculate Gini coefficient for class imbalance
    if "Group" in df.columns:
        from numpy import cumsum
        counts = sorted(group_counts.values)
        n = len(counts)
        cumulative = cumsum(counts)
        gini = (n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n
        if gini > 0.5:
            imbalance = f"high (Gini={gini:.2f})"
        elif gini > 0.3:
            imbalance = f"moderate (Gini={gini:.2f})"
        else:
            imbalance = f"low (Gini={gini:.2f})"
    else:
        imbalance = None

    formatter.validation_summary(rare_categories, missing_values, imbalance)

    # Visualization
    if viz_enabled and not formatter.json_only:
        try:
            from .visualize import plot_subject_visits
            # Use the configured output directory or default to outputs/.visualizations/data_preparation/
            viz_output_dir = viz_config.get("output_dir")
            if viz_output_dir is None:
                # Default to the standard visualization directory (hidden with dot prefix)
                viz_output_dir = output_root / ".visualizations" / "data_preparation"
            else:
                viz_output_dir = Path(viz_output_dir)

            if not formatter.quiet:
                formatter.console.print(f"[blue][INFO][/blue] Generating visualizations to {viz_output_dir}")
            plot_subject_visits(df, data_root, "Group", viz_output_dir)
        except Exception as e:
            if formatter.verbose:
                formatter.warning(f"Visualization failed: {e}")
            else:
                # Show error even in non-verbose mode for debugging
                formatter.warning(f"Visualization failed: {e}")

    formatter.footer(exit_code=0)
    return 0


def run_split(df, data_root: Path, output_root: Path, formatter: DataPrepFormatter,
             required_visits: list, split_ratios: list, stratify_by: str,
             shuffle: bool, seed: int, dry_run: bool, copy_nifti: bool,
             use_symlinks: bool, debug_mode: bool, min_subjects: int,
             initial_subjects: int) -> int:
    """Run the split action with comprehensive analysis and file copying."""

    formatter.header("split", seed=seed, stratify_by=stratify_by)

    # Analyze visit completeness
    complete_subjects, incomplete_subjects, complete_by_group = analyze_visit_completeness(
        df, required_visits
    )

    # Show data availability analysis
    formatter.data_availability_analysis(
        complete_subjects, incomplete_subjects, complete_by_group, initial_subjects
    )

    # Check feasibility
    feasible_groups = check_split_feasibility(complete_by_group, min_subjects)
    formatter.feasibility_check(feasible_groups, min_subjects)

    # Filter to complete subjects only
    df_complete = df[df["Subject"].isin(complete_subjects)].copy()

    # Dataset info
    formatter.dataset_info(
        len(complete_subjects),
        len(df_complete),
        "MRI",
        str(output_root / "manifests")
    )

    # Perform split
    split = stratified_split_with_redistribution(
        complete_subjects, df_complete, stratify_by, tuple(split_ratios), seed, shuffle
    )

    # Show split results
    formatter.split_results(
        len(split["train"]),
        len(split["val"]),
        len(split["test"]),
        len(complete_subjects)
    )

    # Calculate and show class balance
    split_stats = calculate_split_statistics(split, df_complete, stratify_by)
    overall_dist = {}
    for group in df_complete[stratify_by].unique():
        count = df_complete[df_complete[stratify_by] == group]["Subject"].nunique()
        overall_dist[group] = count / len(complete_subjects) * 100

    formatter.class_balance_table(split_stats, overall_dist)

    # Check for leakage
    overlap = check_leakage(split)
    formatter.integrity_check(subject_overlap=overlap)

    if not dry_run:
        # Write manifests
        out_dir = output_root / "manifests"
        file_counts = materialize_manifests(df_complete, split, required_visits, out_dir)
        formatter.files_summary(file_counts, str(out_dir))

        # Copy NIfTI files if enabled
        if copy_nifti:
            # Prepare DataFrame for file operations
            df_for_copy = df_complete.copy()

            # Add Split column
            def _split_label(subj: str) -> str:
                if subj in split["train"]:
                    return "train"
                elif subj in split["val"]:
                    return "val"
                else:
                    return "test"

            df_for_copy["Split"] = df_for_copy["Subject"].apply(_split_label)

            # Filter to required visits
            required_lower = [v.lower() for v in required_visits]
            df_for_copy = df_for_copy[df_for_copy["Visit"].str.lower().isin(required_lower)]

            # Check for Image Data ID column
            if "Image Data ID" not in df_for_copy.columns and "Image ID" not in df_for_copy.columns:
                formatter.warning("No 'Image Data ID' column found - skipping file copy")
            else:
                # Initialize file operations
                file_ops = FileOperations(use_symlinks=use_symlinks, debug_mode=debug_mode)

                # Copy files
                nifti_dir = output_root / "1_splitted_sequential"
                copy_stats = file_ops.mirror_nifti_files(
                    df_for_copy, data_root, nifti_dir, formatter, dry_run
                )

                # Show file operation summary
                total_copied = sum(s["copied"] for s in copy_stats.values())
                total_skipped = sum(s["skipped"] for s in copy_stats.values())
                total_errors = sum(s["errors"] for s in copy_stats.values())

                formatter.file_operations_summary(
                    total_copied, total_skipped, total_errors,
                    file_ops.errors[:10] if formatter.verbose else None
                )

                # Show next steps
                if total_errors == 0:
                    formatter.next_steps([
                        f"Verify file structure: {nifti_dir}/{{split}}/{{subject}}/{{subject}}_{{visit}}.nii[.gz]",
                        f"Each subject has complete temporal sequence: {' → '.join(required_visits)}",
                        "Proceed with skull stripping and ROI extraction on this sequential dataset"
                    ])

    else:
        formatter.console.print("[yellow][DRY RUN][/yellow] No files written")

    formatter.footer(exit_code=0)
    return 0