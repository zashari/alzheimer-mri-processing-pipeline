"""Visualization utilities for center crop sub-stage."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    MATPLOTLIB_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    PIL_AVAILABLE = False


def extract_subject_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract subject ID from temporal filename.

    Args:
        filename: Filename like "002_S_0295_sc_axial_x123.png"

    Returns:
        Subject ID like "002_S_0295" or None if not found
    """
    parts = filename.split('_')
    if len(parts) >= 3 and parts[1] == 'S':
        return f"{parts[0]}_S_{parts[2]}"
    return None


def extract_visit_from_filename(filename: str, required_visits: List[str]) -> Optional[str]:
    """
    Extract visit from temporal filename.

    Args:
        filename: Filename like "002_S_0295_sc_axial_x123.png"
        required_visits: List of required visits

    Returns:
        Visit code like "sc" or None if not found
    """
    for visit in required_visits:
        if f"_{visit}_" in filename:
            return visit
    return None


def get_subject_temporal_files(
    class_dir: Path,
    slice_type: str,
    required_visits: List[str]
) -> Dict[str, Dict[str, Path]]:
    """
    Get temporal files for subjects with complete sequences.

    Args:
        class_dir: Directory containing PNG files
        slice_type: Slice type to filter
        required_visits: List of required visits

    Returns:
        Dictionary: {subject_id: {visit: file_path}}
    """
    if not class_dir.exists():
        return {}

    png_files = list(class_dir.glob("*.png"))
    if not png_files:
        return {}

    # Group files by subject
    subject_files: Dict[str, Dict[str, Path]] = {}
    for png_path in png_files:
        filename = png_path.name
        subject_id = extract_subject_id_from_filename(filename)
        visit = extract_visit_from_filename(filename, required_visits)

        if subject_id and visit and slice_type in filename:
            if subject_id not in subject_files:
                subject_files[subject_id] = {}
            subject_files[subject_id][visit] = png_path

    # Filter to subjects with complete temporal sequences
    complete_subjects = {}
    for subject_id, visit_files in subject_files.items():
        if set(visit_files.keys()) == set(required_visits):
            complete_subjects[subject_id] = visit_files

    return complete_subjects


def select_representative_subject(
    class_dir: Path,
    slice_type: str,
    required_visits: List[str],
    seed: int = 42
) -> tuple[Optional[str], Dict[str, Path]]:
    """
    Select one subject with complete temporal sequence.

    Args:
        class_dir: Directory containing PNG files
        slice_type: Slice type to filter
        required_visits: List of required visits
        seed: Random seed for selection

    Returns:
        Tuple of (subject_id, {visit: file_path})
    """
    complete_subjects = get_subject_temporal_files(class_dir, slice_type, required_visits)

    if not complete_subjects:
        return None, {}

    # Set seed for reproducibility
    random.seed(seed)
    subject_id = random.choice(list(complete_subjects.keys()))
    return subject_id, complete_subjects[subject_id]


def create_temporal_visualization(
    processed_output_dir: Path,
    slice_types: List[str],
    splits: List[str],
    groups: List[str],
    required_visits: List[str],
    visualization_output_dir: Path,
    formatter,
    seed: int = 42
) -> List[Path]:
    """
    Create temporal progression visualization.

    Args:
        processed_output_dir: Directory containing processed PNG files (e.g., outputs/6_center_crop)
        slice_types: List of slice types
        splits: List of splits
        groups: List of groups
        required_visits: List of required visits
        visualization_output_dir: Directory to save visualization
        formatter: Formatter instance for output
        seed: Random seed for subject selection

    Returns:
        List of created visualization file paths
    """
    if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
        if formatter and not formatter.json_only:
            formatter.warning("matplotlib/PIL not available, skipping visualizations")
        return []

    visualization_output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Class names and colors
    class_names = {
        'CN': 'Cognitively Normal',
        'AD': "Alzheimer's Disease"
    }

    class_colors = {
        'CN': '#2E8B57',  # Sea Green
        'AD': '#DC143C'   # Crimson
    }

    visit_colors = {
        'sc': '#2E8B57',   # Dark green for baseline
        'm06': '#4682B4',  # Steel blue for 6 months
        'm12': '#8B0000'   # Dark red for 12 months
    }

    slice_type_names = {
        'axial': 'Axial (Top-Bottom)',
        'coronal': 'Coronal (Front-Back)',
        'sagittal': 'Sagittal (Left-Right)'
    }

    try:
        # Create comprehensive temporal visualization
        fig, axes = plt.subplots(
            len(slice_types),
            len(groups) * len(required_visits),
            figsize=(4 * len(groups) * len(required_visits), 4 * len(slice_types))
        )

        if len(slice_types) == 1:
            axes = axes.reshape(1, -1)
        if len(groups) * len(required_visits) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            'Temporal Disease Progression Across Brain Views\n'
            'Each Row: Different Brain View | Each Column Group: Disease Class Progression',
            fontsize=16, fontweight='bold', y=1
        )

        # Collect subjects and their temporal data
        for slice_idx, slice_type in enumerate(slice_types):
            for group_idx, group_name in enumerate(groups):
                # Use train data for visualization
                class_dir = processed_output_dir / slice_type / 'train' / group_name

                # Select representative subject with complete temporal sequence
                subject_id, temporal_files = select_representative_subject(
                    class_dir, slice_type, required_visits, seed
                )

                # Plot temporal sequence for this class
                for visit_idx, visit in enumerate(required_visits):
                    col = group_idx * len(required_visits) + visit_idx
                    ax = axes[slice_idx, col]

                    if subject_id and visit in temporal_files:
                        try:
                            # Load and display the image
                            img_path = temporal_files[visit]
                            img = np.array(Image.open(img_path))

                            im = ax.imshow(img, cmap='gray', aspect='equal')

                            # Add intensity range info
                            ax.text(
                                0.02, 0.98, f'[{img.min():.0f}, {img.max():.0f}]',
                                transform=ax.transAxes, fontsize=8,
                                verticalalignment='top', horizontalalignment='left',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                            )

                            ax.axis('off')

                        except Exception as e:
                            ax.text(
                                0.5, 0.5, f'{visit.upper()}\nERROR\n{str(e)[:15]}...',
                                ha='center', va='center', transform=ax.transAxes,
                                fontsize=10, color='red'
                            )
                            ax.set_title(f'{visit.upper()}\nError', fontsize=10, color='red')
                            ax.axis('off')
                    else:
                        # Missing data
                        ax.text(
                            0.5, 0.5, f'{visit.upper()}\nMISSING',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=12, color='gray', alpha=0.7
                        )
                        ax.set_title(f'{visit.upper()}\nNo Data', fontsize=10, color='gray')
                        ax.axis('off')

            # Add slice type label on the left
            if slice_idx < len(axes):
                axes[slice_idx, 0].text(
                    -0.15, 0.5, f'{slice_type_names.get(slice_type, slice_type)}',
                    transform=axes[slice_idx, 0].transAxes,
                    fontsize=14, fontweight='bold',
                    rotation=90, va='center', ha='center'
                )

        # Add class and visit labels at the top
        for group_idx, group_name in enumerate(groups):
            for visit_idx, visit in enumerate(required_visits):
                col = group_idx * len(required_visits) + visit_idx
                if col < axes.shape[1]:
                    # Class label (spanning all visits for this class)
                    if visit_idx == len(required_visits) // 2:  # Middle visit for class label
                        axes[0, col].text(
                            0.5, 1.2, f'{class_names.get(group_name, group_name)}\n({group_name})',
                            transform=axes[0, col].transAxes,
                            fontsize=12, fontweight='bold',
                            ha='center', va='center',
                            color=class_colors.get(group_name, 'black'),
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5)
                        )

                    # Visit label
                    axes[0, col].text(
                        0.5, 1.05, visit.upper(),
                        transform=axes[0, col].transAxes,
                        fontsize=11, fontweight='bold',
                        color=visit_colors.get(visit, 'black'),
                        ha='center', va='center'
                    )

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, left=0.08, hspace=0.3, wspace=0.1)

        # Save visualization
        viz_path = visualization_output_dir / "temporal_progression.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        created_files.append(viz_path)

        return created_files

    except Exception as e:
        if formatter and not formatter.json_only:
            formatter.warning(f"Visualization creation failed: {e}")
        return []

