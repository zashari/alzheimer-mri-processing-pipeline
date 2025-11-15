"""Skull stripping implementation for NIfTI processing stage."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..formatter import NiftiFormatter
from .processor import HDBETProcessor
from ..gpu_utils import (
    get_gpu_info,
    get_gpu_memory_info,
    cleanup_gpu_memory,
    setup_gpu_environment
)
from .visualize import create_batch_visualization


def check_existing_progress(output_root: Path, required_visits: List[str]) -> Tuple[int, int, Dict[str, int]]:
    """
    Check for existing skull-stripped files.

    Returns:
        Tuple of (completed_files, completed_subjects, breakdown_by_split)
    """
    completed_files = 0
    completed_subjects = set()
    breakdown = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        if not split_dir.exists():
            continue

        for subject_dir in split_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name
            subject_completed = False

            for visit in required_visits:
                brain_file = subject_dir / f"{subject}_{visit}_brain.nii.gz"
                if brain_file.exists():
                    completed_files += 1
                    breakdown[split] += 1
                    subject_completed = True

            if subject_completed:
                completed_subjects.add(subject)

    return completed_files, len(completed_subjects), breakdown


def build_task_list(
    input_root: Path,
    output_root: Path,
    required_visits: List[str]
) -> Tuple[List[Tuple], List[Tuple], List[str]]:
    """
    Build list of processing tasks.

    Returns:
        Tuple of (tasks_to_process, tasks_to_skip, missing_files)
    """
    tasks_to_process = []
    tasks_to_skip = []
    missing_files = []

    for split in ["train", "val", "test"]:
        split_input = input_root / split
        if not split_input.exists():
            continue

        for subject_dir in split_input.iterdir():
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name

            for visit in required_visits:
                # Look for input file
                input_patterns = [
                    f"{subject}_{visit}.nii",
                    f"{subject}_{visit}.nii.gz"
                ]

                input_file = None
                for pattern in input_patterns:
                    candidate = subject_dir / pattern
                    if candidate.exists():
                        input_file = candidate
                        break

                if not input_file:
                    missing_files.append(f"{split}/{subject}/{subject}_{visit}")
                    continue

                # Define output paths
                output_brain = output_root / split / subject / f"{subject}_{visit}_brain.nii.gz"
                output_mask = output_root / split / subject / f"{subject}_{visit}_mask.nii.gz"

                # Create task tuple
                task = (split, subject, visit, input_file, output_brain, output_mask)

                if output_brain.exists():
                    tasks_to_skip.append(task)
                else:
                    tasks_to_process.append(task)

    return tasks_to_process, tasks_to_skip, missing_files


def run_test(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run test mode - process only a few sample files to verify setup.
    """
    # Get configuration
    skull_cfg = cfg.get("nifti_processing", {}).get("skull_stripping", {})
    test_cfg = skull_cfg.get("test", {})

    device = skull_cfg.get("device", "cuda")
    use_tta = skull_cfg.get("use_tta", False)
    timeout_sec = skull_cfg.get("timeout_sec", 600)
    execution_method = skull_cfg.get("execution_method", "auto")
    num_samples = test_cfg.get("num_samples", 2)
    save_viz = test_cfg.get("save_visualization", True)

    # Setup paths
    output_root = Path(cfg.get("paths", {}).get("output_root", "outputs"))
    input_dir = output_root / skull_cfg.get("input_dir", "1_splitted_sequential")
    output_dir = output_root / skull_cfg.get("output_dir", "2_skull_stripping")
    test_output = output_dir / "test"
    test_output.mkdir(parents=True, exist_ok=True)

    formatter.header("test", "skull_stripping", device=device, samples=num_samples)

    # Initialize processor with test mode and verbose flags
    processor = HDBETProcessor(
        device=device,
        use_tta=use_tta,
        timeout_sec=timeout_sec,
        verbose=formatter.verbose,
        is_test_mode=True,
        execution_method=execution_method
    )

    # Check HD-BET availability
    hd_bet_available = processor.check_availability()
    gpu_info = get_gpu_info() if device == "cuda" else None
    formatter.hd_bet_status(hd_bet_available, gpu_info)

    if not hd_bet_available:
        formatter.error("HD-BET is not available", {
            "cause": "HD-BET command-line tool not found",
            "next_steps": "Install with: pip install hd-bet"
        })
        formatter.footer(exit_code=1)
        return 1

    # Find sample files
    formatter.console.print("[blue]🔍 Finding sample files...[/blue]")
    sample_files = []

    for split in ["train", "val", "test"]:
        split_dir = input_dir / split
        if split_dir.exists():
            for nii_file in split_dir.rglob("*.nii*"):
                sample_files.append(nii_file)
                if len(sample_files) >= num_samples:
                    break
            if len(sample_files) >= num_samples:
                break

    if len(sample_files) < num_samples:
        formatter.warning(f"Only found {len(sample_files)} files (requested {num_samples})")

    # Process sample files
    results = []
    with formatter.create_progress_bar() as progress:
        task = progress.add_task("Processing test files", total=len(sample_files))

        for i, input_file in enumerate(sample_files):
            progress.update(task, description=f"Processing {input_file.name}")

            # Create output paths
            output_brain = test_output / f"test_{i}_brain.nii.gz"
            output_mask = test_output / f"test_{i}_mask.nii.gz"

            # Process file
            start_time = time.time()
            status, error_msg = processor.process_file(
                input_file, output_brain, output_mask, f"test_{i}"
            )
            process_time = time.time() - start_time

            # Store result
            result = {
                "input_file": input_file.name,
                "success": status == "success",
                "time": process_time,
                "brain_file": str(output_brain) if output_brain.exists() else None,
                "mask_file": str(output_mask) if output_mask.exists() else None,
                "error": error_msg
            }
            results.append(result)

            # Update progress bar
            progress.update(task, advance=1)

            if status == "success":
                formatter.success(f"Completed {input_file.name} in {process_time:.1f}s")
            else:
                formatter.error(f"Failed {input_file.name}: {error_msg}")

    # Show results
    formatter.test_results(results)

    # Create visualization if requested
    if save_viz and any(r["success"] for r in results):
        viz_dir = output_root / ".visualizations" / "nifti_processing" / "skull_stripping" / "test"
        successful_results = [
            (Path(sample_files[i]), Path(r["brain_file"]), Path(r["mask_file"]) if r["mask_file"] else None)
            for i, r in enumerate(results)
            if r["success"] and r["brain_file"]
        ]

        viz_paths = create_batch_visualization(successful_results, viz_dir)
        if viz_paths:
            formatter.console.print(f"[blue]📸 Saved {len(viz_paths)} visualizations to {viz_dir}[/blue]")

    # Summary
    successful = sum(1 for r in results if r["success"])
    if successful == len(results):
        formatter.success("All test files processed successfully!")
        formatter.next_steps([
            "HD-BET is working correctly",
            "Run full processing with: python -m data_processing.cli nifti_processing process"
        ])
        exit_code = 0
    else:
        formatter.warning(f"Some files failed ({len(results) - successful}/{len(results)})")
        exit_code = 1

    # Cleanup
    processor.cleanup()

    formatter.footer(exit_code=exit_code)
    return exit_code


def run_process(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run full skull stripping on all data.
    """
    # Get configuration
    skull_cfg = cfg.get("nifti_processing", {}).get("skull_stripping", {})

    device = skull_cfg.get("device", "cuda")
    use_tta = skull_cfg.get("use_tta", False)
    timeout_sec = skull_cfg.get("timeout_sec", 600)
    execution_method = skull_cfg.get("execution_method", "auto")
    subjects_per_batch = skull_cfg.get("subjects_per_batch", 5)
    enable_gpu_cleanup = skull_cfg.get("enable_gpu_cleanup", True)
    cleanup_wait_time = skull_cfg.get("cleanup_wait_time", 5)
    required_visits = skull_cfg.get("required_visits", ["sc", "m06", "m12"])

    # Setup paths
    output_root = Path(cfg.get("paths", {}).get("output_root", "outputs"))
    input_dir = output_root / skull_cfg.get("input_dir", "1_splitted_sequential")
    output_dir = output_root / skull_cfg.get("output_dir", "2_skull_stripping")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up test files before processing
    test_dir = output_dir / "test"
    if test_dir.exists():
        # Count files before deletion for reporting
        test_files = list(test_dir.glob("test_*"))
        all_files = list(test_dir.glob("*"))

        if test_files or all_files:
            # Remove the entire test directory and its contents
            shutil.rmtree(test_dir)
            formatter.console.print(f"[yellow]🧹 Cleaned up test directory with {len(all_files)} files from previous run[/yellow]")
        else:
            # Empty directory, just remove it
            test_dir.rmdir()
            formatter.console.print("[yellow]🧹 Removed empty test directory[/yellow]")

    formatter.header("process", "skull_stripping", device=device)

    # Setup GPU environment
    if device == "cuda":
        setup_gpu_environment(device)

    # Initialize processor with verbose flag
    processor = HDBETProcessor(
        device=device,
        use_tta=use_tta,
        timeout_sec=timeout_sec,
        verbose=formatter.verbose,
        is_test_mode=False,
        execution_method=execution_method,
        formatter=formatter  # Pass formatter for consistent output
    )

    # Check HD-BET availability
    hd_bet_available = processor.check_availability()
    gpu_info = get_gpu_info() if device == "cuda" else None
    formatter.hd_bet_status(hd_bet_available, gpu_info)

    if not hd_bet_available:
        formatter.error("HD-BET is not available", {
            "cause": "HD-BET command-line tool not found",
            "next_steps": "Install with: pip install hd-bet"
        })
        formatter.footer(exit_code=1)
        return 1

    # Show configuration
    formatter.configuration(
        device, None,  # Profile not implemented yet
        use_tta, subjects_per_batch, enable_gpu_cleanup
    )

    # Check existing progress (but don't display to avoid long output)
    completed_files, completed_subjects, breakdown = check_existing_progress(output_dir, required_visits)

    # Build task list
    tasks_to_process, tasks_to_skip, missing_files = build_task_list(
        input_dir, output_dir, required_visits
    )

    total_tasks = len(tasks_to_process) + len(tasks_to_skip)
    formatter.task_summary(total_tasks, len(tasks_to_process), len(tasks_to_skip))

    if len(tasks_to_process) == 0:
        formatter.success("All files already processed!")
        formatter.footer(exit_code=0)
        return 0

    # Process in batches
    all_success = []
    all_failed = []
    all_errors = []
    all_skipped = []
    batch_num = 0

    # Initialize counters for real-time tracking
    current_success = 0
    current_failed = 0
    current_skipped = len(tasks_to_skip)  # Pre-existing skips

    with formatter.create_progress_bar() as progress:
        task = progress.add_task("[cyan]Initializing...[/cyan]", total=len(tasks_to_process))

        for i in range(0, len(tasks_to_process), subjects_per_batch):
            batch_num += 1
            batch = tasks_to_process[i:i + subjects_per_batch]

            # Convert tasks to processor format
            processor_tasks = [
                (t[3], t[4], t[5], f"{t[1]}_{t[2]}")  # input, brain, mask, task_id
                for t in batch
            ]

            # Enhanced callback with real-time updates
            def progress_callback(event, current_idx, total, **kwargs):
                nonlocal current_success, current_failed, current_skipped

                if event == 'start':
                    # Show current file being processed
                    file_name = kwargs.get('file_name', 'Unknown')
                    progress.update(task,
                        description=f"Processing {file_name}")

                elif event == 'complete':
                    # Update counters based on status
                    status = kwargs.get('status')
                    if status == 'success':
                        current_success += 1
                    elif status == 'skip':
                        current_skipped += 1
                    else:
                        current_failed += 1

                    # Advance progress bar
                    progress.update(task, advance=1)

                    # Update description with live statistics
                    stats_text = (f"[green]{current_success} done[/green] | "
                                 f"[yellow]{current_skipped} skipped[/yellow] | "
                                 f"[red]{current_failed} failed[/red]")
                    progress.update(task, description=stats_text)

            batch_results = processor.process_batch(processor_tasks, progress_callback)

            # Track results for final summary
            all_success.extend(batch_results["success"])
            all_failed.extend(batch_results["failed"])
            all_errors.extend(batch_results["errors"])
            all_skipped.extend(batch_results.get("skipped", []))

            # Show batch results
            formatter.batch_results(
                batch_num,
                len(batch_results["success"]),
                len(batch_results["skipped"]),
                len(batch_results["failed"]),
                batch_results["errors"]
            )

            # GPU cleanup if enabled
            if enable_gpu_cleanup and device == "cuda" and (i + subjects_per_batch) < len(tasks_to_process):
                cleanup_result = cleanup_gpu_memory(cleanup_wait_time)
                if cleanup_result["success"]:
                    formatter.gpu_cleanup(
                        cleanup_result["before_mb"],
                        cleanup_result["after_mb"],
                        cleanup_result["freed_mb"]
                    )

    # Calculate average processing time
    avg_time = None
    if all_success:
        # Rough estimate based on batch processing
        total_time = time.time() - formatter.start_time
        avg_time = total_time / len(all_success)

    # Final summary
    formatter.final_summary(
        total_tasks,
        len(all_success),
        len(tasks_to_skip),
        len(all_failed),
        avg_time,
        all_errors
    )

    # Create visualizations
    viz_cfg = skull_cfg.get("visualization", {})
    if viz_cfg.get("enabled", True) and all_success:
        viz_dir = output_root / ".visualizations" / "nifti_processing" / "skull_stripping"
        sample_size = viz_cfg.get("sample_size", 3)

        # Get sample results for visualization
        sample_results = []
        for split in ["train", "val", "test"]:
            split_samples = [s for s in all_success if str(split) in str(s[1])][:sample_size]
            for orig, brain, mask in split_samples:
                # Find original file for visualization
                subject = brain.stem.split("_")[0]
                visit = brain.stem.split("_")[1].replace("_brain", "")
                orig_path = input_dir / split / subject / f"{subject}_{visit}.nii.gz"
                if not orig_path.exists():
                    orig_path = input_dir / split / subject / f"{subject}_{visit}.nii"

                if orig_path.exists():
                    sample_results.append((orig_path, brain, mask))

        if sample_results:
            viz_paths = create_batch_visualization(sample_results, viz_dir, sample_size)
            if viz_paths:
                formatter.console.print(f"\n[blue]📸 Saved {len(viz_paths)} visualizations to {viz_dir}[/blue]")

    # Next steps
    if len(all_failed) == 0:
        formatter.next_steps([
            f"Verify skull-stripped files in: {output_dir}",
            "Proceed with template registration",
            "Run: python -m data_processing.cli nifti_processing template_registration"
        ])
        exit_code = 0
    else:
        formatter.next_steps([
            f"Review failed files: {len(all_failed)} failures",
            f"Check error logs in .reports/",
            "Retry with different settings if needed"
        ])
        exit_code = 1 if len(all_failed) > len(all_success) else 0

    # Cleanup
    processor.cleanup()

    formatter.footer(exit_code=exit_code)
    return exit_code


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for skull stripping substage.
    """
    # Initialize formatter
    verbose = cfg.get("debug", False)
    quiet = cfg.get("quiet", False)
    json_only = cfg.get("json", False)
    formatter = NiftiFormatter(verbose=verbose, quiet=quiet, json_only=json_only)

    if action == "test":
        return run_test(cfg, formatter)
    elif action == "process":
        return run_process(cfg, formatter)
    else:
        formatter.error(f"Unknown action for skull stripping: {action}")
        formatter.footer(exit_code=2)
        return 2