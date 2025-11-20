"""Image enhancement runner for image processing stage."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from ..formatter import ImageProcessingFormatter
from .processor import ImageEnhancementProcessor
from .visualize import create_enhancement_comparison_visualization


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for image enhancement substage.

    Args:
        action: Action to perform (test/process)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    enh_cfg = img_cfg.get("image_enhancement", {})

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
        formatter.error(f"Unknown action for image enhancement: {action}")
        formatter.footer(exit_code=2)
        return 2


def run_test(cfg: Dict, formatter: ImageProcessingFormatter) -> int:
    """
    Run test mode - enhance a few sample files to verify setup.
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    enh_cfg = img_cfg.get("image_enhancement", {})
    test_cfg = enh_cfg.get("test", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("test", "image_enhancement", samples=str(test_cfg.get("num_samples", 3)))

    # Setup paths
    input_dir = output_root / enh_cfg.get("input_dir", "6_center_crop")
    output_dir = output_root / enh_cfg.get("output_dir", "7_enhanced")

    # Get parameters
    method = enh_cfg.get("method", "adaptive")
    gwo_iterations = enh_cfg.get("gwo_iterations", 20)
    num_wolves = enh_cfg.get("num_wolves", 10)
    slice_types = enh_cfg.get("slice_types", ["axial", "coronal", "sagittal"])
    splits = enh_cfg.get("splits", ["train", "val", "test"])
    groups = enh_cfg.get("groups", ["AD", "CN", "MCI"])
    required_visits = enh_cfg.get("required_visits", ["sc", "m06", "m12"])
    seed = cfg.get("seed", 42)

    # Initialize processor
    processor = ImageEnhancementProcessor(
        method=method,
        gwo_iterations=gwo_iterations,
        num_wolves=num_wolves,
        slice_types=slice_types,
        splits=splits,
        groups=groups,
        required_visits=required_visits,
        max_images_per_class=test_cfg.get("num_samples", 3),
        seed=seed,
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "cause": "; ".join(errors),
            "next_steps": "Install missing packages: pip install opencv-python numpy scipy scikit-image"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("All required libraries available")

    # Check input directory
    if not input_dir.exists():
        formatter.error("Input directory not found", {
            "cause": f"Input directory does not exist: {input_dir}",
            "next_steps": "Run center_crop stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Scan input files
    png_paths, scan_stats = processor.scan_input_files(input_dir)

    if len(png_paths) == 0:
        formatter.error("No PNG files found", {
            "cause": f"No PNG files found in {input_dir}",
            "next_steps": "Run center_crop stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Create tasks with output paths
    tasks = [(in_path, output_dir / in_path.relative_to(input_dir)) for in_path in png_paths]

    num_samples = min(test_cfg.get("num_samples", 3), len(tasks))
    test_tasks = tasks[:num_samples]

    formatter.info(f"Found {len(tasks)} PNG files (processing {len(test_tasks)} samples)")

    # Process sample files
    results = []
    stats_dict = {"processed": 0, "skipped": 0, "errors": 0, "error_details": []}

    formatter.info(f"Processing {len(test_tasks)} sample files...")

    with formatter.create_progress_bar() as progress:
        task = progress.add_task("Enhancing", total=len(test_tasks))

        for input_path, output_path in test_tasks:
            result = processor.enhance_single_image(input_path, output_path)
            results.append(result)

            if result.get("skipped", False):
                stats_dict["skipped"] += 1
            elif result.get("success", False):
                stats_dict["processed"] += 1
            else:
                stats_dict["errors"] += 1
                if "error" in result:
                    stats_dict["error_details"].append(f"{input_path.name}: {result['error']}")

            progress.update(task, advance=1)

    # Show results
    formatter.processing_summary(
        stats_dict["processed"],
        stats_dict["skipped"],
        stats_dict["errors"],
        len(test_tasks)
    )

    # Create visualization if requested
    if test_cfg.get("save_visualization", True):
        viz_cfg = enh_cfg.get("visualization", {})
        if viz_cfg.get("enabled", True) and stats_dict["processed"] > 0:
            viz_dir = output_root / ".visualizations" / "image_processing" / "image_enhancement" / "test"
            viz_paths = create_enhancement_comparison_visualization(
                input_dir,
                output_dir,
                slice_types,
                splits,
                groups,
                required_visits,
                viz_dir,
                formatter,
                num_samples=min(6, stats_dict["processed"])
            )
            if viz_paths:
                formatter.info(f"Visualization saved to {viz_dir}")

    # Summary
    if stats_dict["errors"] == 0:
        formatter.success("Test completed successfully!")
        formatter.next_steps([
            "Image enhancement is working correctly",
            f"Run full processing with: python -m data_processing.cli image_processing process --substage image_enhancement"
        ])
        exit_code = 0
    else:
        formatter.warning(f"Some files failed ({stats_dict['errors']}/{len(test_tasks)})")
        exit_code = 1

    # Store results in report
    formatter.report_data.update({
        "test_results": {
            "processed": stats_dict["processed"],
            "skipped": stats_dict["skipped"],
            "errors": stats_dict["errors"],
            "total_samples": len(test_tasks),
            "method": method
        }
    })

    formatter.footer(exit_code=exit_code)
    return exit_code


def run_process(cfg: Dict, formatter: ImageProcessingFormatter) -> int:
    """
    Run full image enhancement processing on all data.
    """
    # Get configuration
    img_cfg = cfg.get("image_processing", {})
    enh_cfg = img_cfg.get("image_enhancement", {})
    paths_cfg = cfg.get("paths", {})
    output_root = Path(paths_cfg.get("output_root", "outputs"))

    formatter.header("process", "image_enhancement")

    # Setup paths
    input_dir = output_root / enh_cfg.get("input_dir", "6_center_crop")
    output_dir = output_root / enh_cfg.get("output_dir", "7_enhanced")

    # Get parameters
    method = enh_cfg.get("method", "adaptive")
    gwo_iterations = enh_cfg.get("gwo_iterations", 30)
    num_wolves = enh_cfg.get("num_wolves", 15)
    max_workers = enh_cfg.get("max_workers", 4)
    slice_types = enh_cfg.get("slice_types", ["axial", "coronal", "sagittal"])
    splits = enh_cfg.get("splits", ["train", "val", "test"])
    groups = enh_cfg.get("groups", ["AD", "CN", "MCI"])
    required_visits = enh_cfg.get("required_visits", ["sc", "m06", "m12"])
    max_images_per_class = enh_cfg.get("max_images_per_class", None)
    seed = cfg.get("seed", 42)

    # Initialize processor
    processor = ImageEnhancementProcessor(
        method=method,
        gwo_iterations=gwo_iterations,
        num_wolves=num_wolves,
        slice_types=slice_types,
        splits=splits,
        groups=groups,
        required_visits=required_visits,
        max_images_per_class=max_images_per_class,
        seed=seed,
        verbose=cfg.get("debug", False)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "cause": "; ".join(errors),
            "next_steps": "Install missing packages: pip install opencv-python numpy scipy scikit-image"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("All required libraries available")

    # Show configuration
    formatter.info("Configuration")
    formatter.print(f"  • Method: {method}")
    formatter.print(f"  • GWO iterations: {gwo_iterations}")
    formatter.print(f"  • Number of wolves: {num_wolves}")
    formatter.print(f"  • Max workers: {max_workers}")
    formatter.print("")

    # Check input directory
    if not input_dir.exists():
        formatter.error("Input directory not found", {
            "cause": f"Input directory does not exist: {input_dir}",
            "next_steps": "Run center_crop stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Scan input files
    png_paths, scan_stats = processor.scan_input_files(input_dir)

    if len(png_paths) == 0:
        formatter.error("No PNG files found", {
            "cause": f"No PNG files found in {input_dir}",
            "next_steps": "Run center_crop stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    # Create tasks with output paths
    tasks = [(in_path, output_dir / in_path.relative_to(input_dir)) for in_path in png_paths]

    formatter.info(f"Found {len(tasks)} PNG files to enhance")

    # Show temporal statistics
    temporal_stats = scan_stats.get("temporal_stats", {})
    if temporal_stats:
        formatter.info("Temporal Dataset Statistics")
        for slice_t in slice_types:
            if slice_t in temporal_stats:
                formatter.print(f"  {slice_t.upper()} slice type:")
                for key, visit_dist in temporal_stats[slice_t].items():
                    split, cls = key.split("_", 1)
                    visit_str = ", ".join([f"{v}:{visit_dist.get(v, 0)}" for v in required_visits])
                    formatter.print(f"    {split}/{cls}: visits({visit_str})")
        formatter.print("")

    # Process all files in parallel
    start_time = time.time()
    results = []
    stats_dict = {"processed": 0, "skipped": 0, "errors": 0, "error_details": []}

    def worker(task_tuple):
        """Worker function for parallel processing."""
        input_path, output_path = task_tuple
        return processor.enhance_single_image(input_path, output_path)

    formatter.info(f"Starting enhancement with {max_workers} workers...")

    with formatter.create_progress_bar() as progress:
        task_id = progress.add_task("Enhancing", total=len(tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, t): t for t in tasks}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result.get("skipped", False):
                    stats_dict["skipped"] += 1
                elif result.get("success", False):
                    stats_dict["processed"] += 1
                else:
                    stats_dict["errors"] += 1
                    if "error" in result:
                        stats_dict["error_details"].append(
                            f"{Path(result['input_path']).name}: {result.get('error', 'Unknown error')}"
                        )

                progress.update(task_id, advance=1)

    processing_time = time.time() - start_time

    # Show results
    formatter.processing_summary(
        stats_dict["processed"],
        stats_dict["skipped"],
        stats_dict["errors"],
        len(tasks)
    )

    # Show errors if any
    if stats_dict["errors"] > 0 and formatter.verbose:
        formatter.warning(f"Encountered {stats_dict['errors']} errors")
        for error in stats_dict["error_details"][:10]:
            formatter.print(f"  • {error}")
        if len(stats_dict["error_details"]) > 10:
            formatter.print(f"  ... and {len(stats_dict['error_details']) - 10} more errors")

    # Create visualization if requested
    viz_cfg = enh_cfg.get("visualization", {})
    if viz_cfg.get("enabled", True) and stats_dict["processed"] > 0:
        viz_dir = output_root / ".visualizations" / "image_processing" / "image_enhancement"
        viz_paths = create_enhancement_comparison_visualization(
            input_dir,
            output_dir,
            slice_types,
            splits,
            groups,
            required_visits,
            viz_dir,
            formatter,
            num_samples=viz_cfg.get("sample_size", 6)
        )
        if viz_paths:
            formatter.info(f"Saved {len(viz_paths)} visualizations to {viz_dir}")

    # Store results in report
    formatter.report_data.update({
        "parameters": {
            "method": method,
            "gwo_iterations": gwo_iterations,
            "num_wolves": num_wolves,
            "max_workers": max_workers,
            "slice_types": slice_types,
            "splits": splits,
            "groups": groups
        },
        "results": {
            "processed": stats_dict["processed"],
            "skipped": stats_dict["skipped"],
            "errors": stats_dict["errors"],
            "total_files": len(tasks),
            "processing_time_seconds": round(processing_time, 1),
            "processing_time_minutes": round(processing_time / 60, 1)
        }
    })

    # Next steps
    if stats_dict["errors"] == 0:
        formatter.next_steps([
            f"Verify enhanced files in: {output_dir}",
            "Proceed to next substage: data_balancing (if implemented)"
        ])
        exit_code = 0
    else:
        formatter.next_steps([
            f"Review errors: {stats_dict['errors']} files failed",
            "Check error details in verbose mode"
        ])
        exit_code = 1

    formatter.footer(exit_code=exit_code)
    return exit_code

