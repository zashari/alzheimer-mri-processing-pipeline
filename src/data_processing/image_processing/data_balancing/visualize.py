"""Visualization utilities for data balancing sub-stage."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    CV2_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    NUMPY_AVAILABLE = False


def create_augmentation_effects_visualization(
    input_root: Path,
    output_root: Path,
    slice_types: List[str],
    groups: List[str],
    required_visits: List[str],
    augmentation_log: Dict,
    visualization_output_dir: Path,
    formatter,
    plane: str = "coronal"
) -> List[Path]:
    """
    Create augmentation effects visualization.

    Args:
        input_root: Directory containing original PNG files
        output_root: Directory containing augmented PNG files
        slice_types: List of slice types
        groups: List of groups
        required_visits: List of required visits
        augmentation_log: Augmentation log dictionary
        visualization_output_dir: Directory to save visualization
        formatter: Formatter instance for output
        plane: Plane to visualize (default: coronal)

    Returns:
        List of created visualization file paths
    """
    if not MATPLOTLIB_AVAILABLE or not CV2_AVAILABLE or not NUMPY_AVAILABLE:
        if formatter and not formatter.json_only:
            formatter.warning("matplotlib/cv2/numpy not available, skipping visualizations")
        return []

    visualization_output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    try:
        # Find first augmented subject from log
        if plane not in augmentation_log:
            plane = slice_types[0] if slice_types else "axial"

        if plane not in augmentation_log:
            return []

        # Get first group with augmentation
        aug_params = None
        selected_group = None
        for group in groups:
            if group in augmentation_log[plane]:
                params_dict = augmentation_log[plane][group].get("augmentation_params", {})
                if params_dict:
                    aug_params = params_dict
                    selected_group = group
                    break

        if not aug_params:
            return []

        # Get first augmented subject
        first_aug_id = sorted(aug_params.keys())[0]
        source_subject = aug_params[first_aug_id]["source"]
        params = aug_params[first_aug_id]["params"]

        # Find original and augmented screening images
        orig_path = input_root / plane / "train" / selected_group
        aug_path = output_root / plane / "train" / selected_group

        orig_files = list(orig_path.glob(f"{source_subject}_sc_*.png"))
        aug_files = list(aug_path.glob(f"AUG_{first_aug_id}_{source_subject}_sc_*.png"))

        if not orig_files or not aug_files:
            return []

        # Load images
        orig_img = cv2.imread(str(orig_files[0]), cv2.IMREAD_GRAYSCALE)
        aug_img = cv2.imread(str(aug_files[0]), cv2.IMREAD_GRAYSCALE)

        if orig_img is None or aug_img is None:
            return []

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Augmentation Effects on Subject {source_subject}", fontsize=16, weight="bold")

        # Original image
        axes[0].imshow(orig_img, cmap="gray")
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")

        # Augmented image
        axes[1].imshow(aug_img, cmap="gray")
        axes[1].set_title("Augmented Image", fontsize=14)
        axes[1].axis("off")

        # Difference map
        diff = cv2.absdiff(orig_img, aug_img)
        im = axes[2].imshow(diff, cmap="hot")
        axes[2].set_title("Difference Map", fontsize=14)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save figure
        viz_path = visualization_output_dir / "augmentation_effects_detail.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()

        created_files.append(viz_path)

        return created_files

    except Exception as e:
        if formatter and not formatter.json_only:
            formatter.warning(f"Visualization creation failed: {e}")
        return []

