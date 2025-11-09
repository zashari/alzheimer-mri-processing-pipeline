"""Template registration implementation for NIfTI processing stage."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..formatter import NiftiFormatter
from ..gpu_utils import setup_gpu_environment, cleanup_gpu_memory
from .processor import RegistrationProcessor
from .visualize import create_batch_visualization


def run(action: str, cfg: Dict) -> int:
    """
    Main entry point for template registration substage.

    Args:
        action: Action to perform (test/process)
        cfg: Configuration dictionary

    Returns:
        Exit code
    """
    # Get configuration
    reg_cfg = cfg.get("nifti_processing", {}).get("template_registration", {})

    # Initialize formatter
    formatter = NiftiFormatter(
        verbose=cfg.get("verbose", False),
        quiet=cfg.get("quiet", False),
        json_only=cfg.get("json_only", False)
    )

    # Print header
    formatter.header(
        "NIfTI Processing",
        action=action.title(),
        substage="template_registration"
    )

    # Setup environment
    if reg_cfg.get("itk_threads"):
        import os
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(reg_cfg["itk_threads"])

    if action == "test":
        return run_test(cfg, formatter)
    elif action == "process":
        return run_process(cfg, formatter)
    else:
        formatter.error(f"Unknown action: {action}")
        formatter.footer(exit_code=1)
        return 1


def run_test(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run test mode - process single subject to verify setup.
    """
    # Get configuration
    reg_cfg = cfg.get("nifti_processing", {}).get("template_registration", {})
    test_cfg = reg_cfg.get("test", {})
    output_root = Path(cfg.get("output_root", "outputs"))

    # Get template paths
    project_root = Path.cwd()
    mni_template = project_root / reg_cfg["mni_template_path"]
    hippo_roi = project_root / reg_cfg["hippocampus_roi_path"]

    # Initialize processor
    processor = RegistrationProcessor(
        mni_template_path=mni_template,
        hippocampus_roi_path=hippo_roi,
        registration_type=reg_cfg.get("registration", {}).get("type", "SyNAggro"),
        num_threads=reg_cfg.get("registration", {}).get("num_threads", 8),
        verbose=cfg.get("verbose", False),
        min_hippo_volume=reg_cfg.get("min_hippo_volume", 100)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "errors": errors,
            "next_steps": "Install missing packages and ensure template files exist"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("ANTs Python package found")
    formatter.success(f"MNI template found: {mni_template.name}")
    formatter.success(f"Hippocampus ROI found: {hippo_roi.name}")

    # Find test subject
    input_dir = output_root / reg_cfg["input_dir"]
    test_task = find_test_subject(input_dir, reg_cfg.get("required_visits", ["sc"]))

    if not test_task:
        formatter.error("No input files found for testing", {
            "cause": f"No skull-stripped brains in {input_dir}",
            "next_steps": "Run skull_stripping stage first"
        })
        formatter.footer(exit_code=1)
        return 1

    split, subject, visit, brain_path = test_task
    formatter.info(f"Testing on: [{split}] {subject}_{visit}")

    # Prepare output paths
    output_base = output_root / reg_cfg["output_dir"]
    brain_slices = {}

    for plane in ['axial', 'sagittal', 'coronal']:
        plane_dir = output_base / reg_cfg["plane_dirs"][plane] / split / subject
        brain_slices[plane] = plane_dir / f"{subject}_{visit}_optimal_{plane}_x0.nii.gz"

    mask_3d_dir = output_base / reg_cfg["hippo_3d_dir"] / split / subject
    mask_3d_path = mask_3d_dir / f"{subject}_{visit}_hippocampus_3D.nii.gz"

    # Process subject
    formatter.info("Processing...")
    result = processor.process_subject(
        brain_path=Path(brain_path),
        output_brain_slices=brain_slices,
        output_mask_3d=mask_3d_path if reg_cfg.get("save_3d_masks", True) else None
    )

    # Show results
    if result['status'] == 'success':
        formatter.success(f"Test successful in {result['processing_time']:.1f}s")
        formatter.info(f"Hippocampus volume: {result['hippo_volume']:.0f} voxels")

        # Show slice info
        formatter.info("Optimal slices extracted:")
        for plane, info in result['slices'].items():
            formatter.print(f"  • {plane}: slice {info['slice_idx']} (area: {info['hippo_area']:.0f})")

        # Create visualization if requested
        if test_cfg.get("save_visualization", True):
            viz_cfg = reg_cfg.get("visualization", {})
            viz_dir = output_root / ".visualizations" / "nifti_processing" / "template_registration" / "test"

            from .visualize import visualize_registration_results

            viz_path = viz_dir / f"test_{subject}_{visit}.png"
            success = visualize_registration_results(
                subject, visit,
                brain_slices,
                mask_3d_path=mask_3d_path if reg_cfg.get("save_3d_masks", True) else None,
                output_path=viz_path,
                alpha=viz_cfg.get("alpha", 0.3),
                show_contour=viz_cfg.get("show_contour", True)
            )

            if success:
                formatter.info(f"Visualization saved to: {viz_path}")

        # Next steps
        formatter.next_steps([
            "Template registration is working correctly",
            "Run full processing with: python -m data_processing.cli nifti_processing process --substage template_registration"
        ])

        # Save report
        formatter.report_data.update({
            "test_subject": f"{subject}_{visit}",
            "results": result
        })

        formatter.footer(exit_code=0)
        return 0

    else:
        formatter.error(f"Test failed: {result['error']}")
        formatter.footer(exit_code=1)
        return 1


def run_process(cfg: Dict, formatter: NiftiFormatter) -> int:
    """
    Run full template registration on all data.
    """
    # Get configuration
    reg_cfg = cfg.get("nifti_processing", {}).get("template_registration", {})
    output_root = Path(cfg.get("output_root", "outputs"))

    # Get template paths
    project_root = Path.cwd()
    mni_template = project_root / reg_cfg["mni_template_path"]
    hippo_roi = project_root / reg_cfg["hippocampus_roi_path"]

    # Initialize processor
    processor = RegistrationProcessor(
        mni_template_path=mni_template,
        hippocampus_roi_path=hippo_roi,
        registration_type=reg_cfg.get("registration", {}).get("type", "SyNAggro"),
        num_threads=reg_cfg.get("registration", {}).get("num_threads", 8),
        verbose=cfg.get("verbose", False),
        min_hippo_volume=reg_cfg.get("min_hippo_volume", 100)
    )

    # Check prerequisites
    success, errors = processor.check_prerequisites()
    if not success:
        formatter.error("Prerequisites not met", {
            "errors": errors,
            "next_steps": "Install missing packages and ensure template files exist"
        })
        formatter.footer(exit_code=1)
        return 1

    formatter.info("Prerequisites")
    formatter.success("All prerequisites met")

    # Show configuration
    formatter.info("Configuration")
    formatter.print(f"  Registration type: {reg_cfg.get('registration', {}).get('type', 'SyNAggro')}")
    formatter.print(f"  ANTs threads: {reg_cfg.get('registration', {}).get('num_threads', 8)}")
    formatter.print(f"  Min hippo volume: {reg_cfg.get('min_hippo_volume', 100)} voxels")
    formatter.print(f"  Save 3D masks: {reg_cfg.get('save_3d_masks', True)}")
    formatter.print(f"  Parallel processing: {reg_cfg.get('use_parallel', True)}")

    if reg_cfg.get('use_parallel', True):
        formatter.print(f"  Max workers: {reg_cfg.get('max_workers', 2)}")

    # Scan for tasks
    input_dir = output_root / reg_cfg["input_dir"]
    output_base = output_root / reg_cfg["output_dir"]

    tasks, existing_files = scan_for_tasks(
        input_dir,
        output_base,
        reg_cfg.get("required_visits", ["sc", "m06", "m12"]),
        reg_cfg.get("hippo_3d_dir", "hippocampus_masks_3D"),
        reg_cfg.get("plane_dirs", {})
    )

    # Show existing progress
    formatter.info("Existing Progress")
    formatter.print(f"  Completed files: {existing_files['complete_count']}")
    formatter.print(f"  Completed subjects: {len(existing_files['complete_subjects'])}")
    formatter.print(f"  Breakdown: train: {existing_files['train']}, val: {existing_files['val']}, test: {existing_files['test']}")

    # Show task summary
    formatter.info("Task Summary")
    formatter.print(f"  Total tasks: {len(tasks['to_process']) + len(tasks['to_skip'])}")
    formatter.print(f"  To process: {len(tasks['to_process'])}")
    formatter.print(f"  To skip: {len(tasks['to_skip'])}")

    if not tasks['to_process']:
        formatter.success("All files already processed!")
        formatter.footer(exit_code=0)
        return 0

    # Process tasks
    formatter.info(f"Processing {len(tasks['to_process'])} files...")

    results = []
    stats = {'success': 0, 'skip': 0, 'error': 0}
    start_time = time.time()

    # Load progress if it exists
    progress_file = output_base / reg_cfg.get("progress_file", "processing_progress.json")
    progress = load_progress(progress_file) if reg_cfg.get("save_progress", True) else {}

    # Process based on configuration
    if reg_cfg.get("use_parallel", True):
        results = process_parallel(
            tasks['to_process'],
            processor,
            reg_cfg,
            output_root,
            formatter,
            stats,
            progress,
            max_workers=reg_cfg.get("max_workers", 2)
        )
    else:
        results = process_sequential(
            tasks['to_process'],
            processor,
            reg_cfg,
            output_root,
            formatter,
            stats,
            progress
        )

    # Save final progress
    if reg_cfg.get("save_progress", True) and progress:
        save_progress(progress_file, progress)

    # Calculate statistics
    total_time = time.time() - start_time
    formatter.print()
    formatter.info("Processing Complete")
    formatter.print(f"  Success: {stats['success']} files")
    formatter.print(f"  Skipped: {stats['skip']} files")
    formatter.print(f"  Failed: {stats['error']} files")
    formatter.print(f"  Total time: {total_time/60:.1f} minutes")

    if stats['success'] > 0:
        formatter.print(f"  Average time per file: {total_time/stats['success']:.1f}s")

    # Create visualizations
    if reg_cfg.get("visualization", {}).get("enabled", True) and stats['success'] > 0:
        viz_dir = output_root / ".visualizations" / "nifti_processing" / "template_registration"
        viz_files = create_batch_visualization(
            [r for r in results if r.get('status') == 'success'],
            viz_dir,
            max_samples=reg_cfg.get("visualization", {}).get("sample_size", 3)
        )

        if viz_files:
            formatter.info(f"Saved {len(viz_files)} visualizations to {viz_dir}")

    # Output verification
    formatter.info("Output Verification")
    final_scan = scan_existing_files(
        output_base,
        reg_cfg.get("hippo_3d_dir", "hippocampus_masks_3D"),
        reg_cfg.get("plane_dirs", {})
    )

    for plane in ['axial', 'coronal', 'sagittal']:
        count = final_scan['planes'].get(plane, 0)
        formatter.print(f"  {plane} slices: {count}")

    formatter.print(f"  3D masks: {final_scan['masks_3d']}")
    formatter.print(f"  Complete subjects: {final_scan['complete']}")

    # Next steps
    if stats['error'] > 0:
        formatter.warning(f"{stats['error']} files failed - check error messages and retry")

    formatter.next_steps([
        f"Verify output files in: {output_base}",
        "Check visualizations in: outputs/.visualizations/nifti_processing/template_registration/",
        "Proceed to next substage: labelling"
    ])

    # Save report
    formatter.report_data.update({
        "parameters": reg_cfg,
        "results": {
            "total_tasks": len(tasks['to_process']) + len(tasks['to_skip']),
            "processed": stats['success'],
            "skipped": stats['skip'],
            "failed": stats['error'],
            "total_time_min": round(total_time/60, 1)
        },
        "output_verification": final_scan
    })

    formatter.footer(exit_code=0 if stats['error'] == 0 else 1)
    return 0 if stats['error'] == 0 else 1


def find_test_subject(input_dir: Path, required_visits: List[str]) -> Optional[Tuple]:
    """Find first available subject for testing."""
    for split in ['train', 'val', 'test']:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue

        for subject_dir in split_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name
            for visit in required_visits:
                brain_file = subject_dir / f"{subject}_{visit}_brain.nii.gz"
                if brain_file.exists():
                    return (split, subject, visit, brain_file)

    return None


def scan_for_tasks(
    input_dir: Path,
    output_base: Path,
    required_visits: List[str],
    hippo_3d_dir: str,
    plane_dirs: Dict[str, str]
) -> Tuple[Dict, Dict]:
    """Scan for processing tasks and existing files."""
    tasks = {'to_process': [], 'to_skip': []}
    existing_files = {
        'complete_count': 0,
        'complete_subjects': set(),
        'train': 0, 'val': 0, 'test': 0
    }

    for split in ['train', 'val', 'test']:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue

        for subject_dir in split_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name

            for visit in required_visits:
                brain_file = subject_dir / f"{subject}_{visit}_brain.nii.gz"

                if brain_file.exists():
                    # Check what already exists
                    all_exist = True

                    # Check brain slices
                    for plane in ['axial', 'sagittal', 'coronal']:
                        plane_subdir = plane_dirs.get(plane, plane)
                        slice_dir = output_base / plane_subdir / split / subject
                        slice_pattern = f"{subject}_{visit}_optimal_{plane}_x*.nii.gz"

                        if not any(slice_dir.glob(slice_pattern) if slice_dir.exists() else []):
                            all_exist = False
                            break

                    # Check 3D mask
                    mask_3d_dir = output_base / hippo_3d_dir / split / subject
                    mask_3d_file = mask_3d_dir / f"{subject}_{visit}_hippocampus_3D.nii.gz"

                    if not mask_3d_file.exists():
                        all_exist = False

                    task = (split, subject, visit, brain_file)

                    if all_exist:
                        tasks['to_skip'].append(task)
                        existing_files['complete_count'] += 1
                        existing_files['complete_subjects'].add(subject)
                        existing_files[split] += 1
                    else:
                        tasks['to_process'].append(task)

    return tasks, existing_files


def scan_existing_files(output_base: Path, hippo_3d_dir: str, plane_dirs: Dict[str, str]) -> Dict:
    """Scan existing output files."""
    result = {
        'planes': {'axial': 0, 'sagittal': 0, 'coronal': 0},
        'masks_3d': 0,
        'complete': 0
    }

    # Count brain slices
    for plane in ['axial', 'sagittal', 'coronal']:
        plane_subdir = plane_dirs.get(plane, plane)
        plane_path = output_base / plane_subdir

        if plane_path.exists():
            files = list(plane_path.glob("*/*/*_optimal_*.nii.gz"))
            result['planes'][plane] = len(files)

    # Count 3D masks
    mask_path = output_base / hippo_3d_dir
    if mask_path.exists():
        masks = list(mask_path.glob("*/*/*_hippocampus_3D.nii.gz"))
        result['masks_3d'] = len(masks)

    # Estimate complete subjects (those with all files)
    # This is approximate - counts subjects with at least one file in each category
    subjects_with_all = set()
    for split in ['train', 'val', 'test']:
        for plane in ['axial', 'sagittal', 'coronal']:
            plane_subdir = plane_dirs.get(plane, plane)
            plane_path = output_base / plane_subdir / split

            if plane_path.exists():
                for subject_dir in plane_path.iterdir():
                    if subject_dir.is_dir():
                        subjects_with_all.add(subject_dir.name)

    result['complete'] = len(subjects_with_all)

    return result


def process_parallel(
    tasks: List,
    processor: RegistrationProcessor,
    reg_cfg: Dict,
    output_root: Path,
    formatter: NiftiFormatter,
    stats: Dict,
    progress: Dict,
    max_workers: int = 2
) -> List[Dict]:
    """Process tasks in parallel."""
    results = []
    completed = set(progress.get('completed', []))

    with formatter.progress() as pbar:
        task_id = pbar.add_task("Processing", total=len(tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for task in tasks:
                split, subject, visit, brain_path = task
                key = f"{split}_{subject}_{visit}"

                if key in completed:
                    stats['skip'] += 1
                    pbar.update(task_id, advance=1)
                    continue

                future = executor.submit(
                    process_single_subject,
                    task, processor, reg_cfg, output_root
                )
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                split, subject, visit, _ = task
                key = f"{split}_{subject}_{visit}"

                try:
                    result = future.result(timeout=1200)  # 20 min timeout
                    results.append(result)

                    if result['status'] == 'success':
                        stats['success'] += 1
                        completed.add(key)
                        formatter.print(f"  ✓ {subject}_{visit}: {len(result.get('slices', {}))} slices, "
                                      f"volume: {result.get('hippo_volume', 0):.0f}")
                    else:
                        stats['error'] += 1
                        formatter.warning(f"  ✗ {subject}_{visit}: {result.get('error', 'Unknown')}")

                except Exception as e:
                    stats['error'] += 1
                    formatter.warning(f"  ✗ {subject}_{visit}: {str(e)}")
                    results.append({
                        'subject': subject,
                        'visit': visit,
                        'status': 'error',
                        'error': str(e)
                    })

                pbar.update(task_id, advance=1)

                # Save progress periodically
                if len(results) % 10 == 0:
                    progress['completed'] = list(completed)

    return results


def process_sequential(
    tasks: List,
    processor: RegistrationProcessor,
    reg_cfg: Dict,
    output_root: Path,
    formatter: NiftiFormatter,
    stats: Dict,
    progress: Dict
) -> List[Dict]:
    """Process tasks sequentially."""
    results = []
    completed = set(progress.get('completed', []))

    with formatter.progress() as pbar:
        task_id = pbar.add_task("Processing", total=len(tasks))

        for task in tasks:
            split, subject, visit, brain_path = task
            key = f"{split}_{subject}_{visit}"

            if key in completed:
                stats['skip'] += 1
                pbar.update(task_id, advance=1)
                continue

            result = process_single_subject(task, processor, reg_cfg, output_root)
            results.append(result)

            if result['status'] == 'success':
                stats['success'] += 1
                completed.add(key)
                formatter.print(f"  ✓ {subject}_{visit}: {len(result.get('slices', {}))} slices")
            else:
                stats['error'] += 1
                formatter.warning(f"  ✗ {subject}_{visit}: {result.get('error', 'Unknown')}")

            pbar.update(task_id, advance=1)

            # Save progress periodically
            if len(results) % 10 == 0:
                progress['completed'] = list(completed)

    return results


def process_single_subject(
    task: Tuple,
    processor: RegistrationProcessor,
    reg_cfg: Dict,
    output_root: Path
) -> Dict:
    """Process a single subject."""
    split, subject, visit, brain_path = task

    # Prepare output paths
    output_base = output_root / reg_cfg["output_dir"]
    brain_slices = {}

    for plane in ['axial', 'sagittal', 'coronal']:
        plane_subdir = reg_cfg.get("plane_dirs", {}).get(plane, plane)
        plane_dir = output_base / plane_subdir / split / subject
        # Include slice index in filename
        brain_slices[plane] = plane_dir / f"{subject}_{visit}_optimal_{plane}_x0.nii.gz"

    mask_3d_path = None
    if reg_cfg.get("save_3d_masks", True):
        mask_3d_dir = output_base / reg_cfg.get("hippo_3d_dir", "hippocampus_masks_3D") / split / subject
        mask_3d_path = mask_3d_dir / f"{subject}_{visit}_hippocampus_3D.nii.gz"

    # Process
    result = processor.process_subject(
        brain_path=Path(brain_path),
        output_brain_slices=brain_slices,
        output_mask_3d=mask_3d_path
    )

    # Update slice filenames with actual indices
    if result['status'] == 'success':
        for plane, info in result.get('slices', {}).items():
            if 'slice_idx' in info and 'output_path' in info:
                # Rename file to include actual slice index
                old_path = Path(info['output_path'])
                new_path = old_path.parent / f"{subject}_{visit}_optimal_{plane}_x{info['slice_idx']}.nii.gz"

                if old_path.exists() and old_path != new_path:
                    old_path.rename(new_path)
                    info['output_path'] = str(new_path)

    # Add metadata
    result.update({
        'subject': subject,
        'visit': visit,
        'split': split
    })

    return result


def load_progress(progress_file: Path) -> Dict:
    """Load progress from file."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def save_progress(progress_file: Path, progress: Dict):
    """Save progress to file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)