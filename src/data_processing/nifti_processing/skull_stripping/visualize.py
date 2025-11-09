"""Visualization utilities for NIfTI processing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

try:
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def plot_skull_stripping_results(
    original_path: Path,
    brain_path: Path,
    mask_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> bool:
    """
    Create before/after visualization of skull stripping.

    Args:
        original_path: Path to original NIfTI file
        brain_path: Path to brain-extracted NIfTI file
        mask_path: Optional path to mask file
        output_path: Optional path to save the figure
        title: Optional title for the figure

    Returns:
        True if successful, False otherwise
    """
    if not VISUALIZATION_AVAILABLE:
        return False

    try:
        # Load NIfTI files
        orig_nii = nib.load(str(original_path))
        brain_nii = nib.load(str(brain_path))

        orig_data = orig_nii.get_fdata()
        brain_data = brain_nii.get_fdata()

        mask_data = None
        if mask_path and mask_path.exists():
            mask_nii = nib.load(str(mask_path))
            mask_data = mask_nii.get_fdata()

        # Get middle slices for visualization
        mid_axial = orig_data.shape[2] // 2
        mid_sagittal = orig_data.shape[0] // 2
        mid_coronal = orig_data.shape[1] // 2

        # Create figure
        n_cols = 3 if mask_data is not None else 2
        fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 4, 12))

        # Plot original
        axes[0, 0].imshow(np.rot90(orig_data[:, :, mid_axial]), cmap="gray")
        axes[0, 0].set_title("Original - Axial")
        axes[0, 0].axis("off")

        axes[1, 0].imshow(np.rot90(orig_data[mid_sagittal, :, :]), cmap="gray")
        axes[1, 0].set_title("Original - Sagittal")
        axes[1, 0].axis("off")

        axes[2, 0].imshow(np.rot90(orig_data[:, mid_coronal, :]), cmap="gray")
        axes[2, 0].set_title("Original - Coronal")
        axes[2, 0].axis("off")

        # Plot brain extracted
        axes[0, 1].imshow(np.rot90(brain_data[:, :, mid_axial]), cmap="gray")
        axes[0, 1].set_title("Brain Extracted - Axial")
        axes[0, 1].axis("off")

        axes[1, 1].imshow(np.rot90(brain_data[mid_sagittal, :, :]), cmap="gray")
        axes[1, 1].set_title("Brain Extracted - Sagittal")
        axes[1, 1].axis("off")

        axes[2, 1].imshow(np.rot90(brain_data[:, mid_coronal, :]), cmap="gray")
        axes[2, 1].set_title("Brain Extracted - Coronal")
        axes[2, 1].axis("off")

        # Plot mask overlay if available
        if mask_data is not None:
            axes[0, 2].imshow(np.rot90(orig_data[:, :, mid_axial]), cmap="gray", alpha=0.7)
            axes[0, 2].imshow(np.rot90(mask_data[:, :, mid_axial]), cmap="hot", alpha=0.3)
            axes[0, 2].set_title("Mask Overlay - Axial")
            axes[0, 2].axis("off")

            axes[1, 2].imshow(np.rot90(orig_data[mid_sagittal, :, :]), cmap="gray", alpha=0.7)
            axes[1, 2].imshow(np.rot90(mask_data[mid_sagittal, :, :]), cmap="hot", alpha=0.3)
            axes[1, 2].set_title("Mask Overlay - Sagittal")
            axes[1, 2].axis("off")

            axes[2, 2].imshow(np.rot90(orig_data[:, mid_coronal, :]), cmap="gray", alpha=0.7)
            axes[2, 2].imshow(np.rot90(mask_data[:, mid_coronal, :]), cmap="hot", alpha=0.3)
            axes[2, 2].set_title("Mask Overlay - Coronal")
            axes[2, 2].axis("off")

        # Add title
        if title:
            fig.suptitle(title, fontsize=16)

        plt.tight_layout()

        # Save or show
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return True

    except Exception:
        return False


def create_batch_visualization(
    results: List[Tuple[Path, Path, Optional[Path]]],
    output_dir: Path,
    max_samples: int = 3
) -> List[Path]:
    """
    Create visualization for multiple skull stripping results.

    Args:
        results: List of (original, brain, mask) path tuples
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize

    Returns:
        List of saved visualization paths
    """
    if not VISUALIZATION_AVAILABLE:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for i, (orig_path, brain_path, mask_path) in enumerate(results[:max_samples]):
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subject_name = brain_path.stem.split("_")[0]  # Extract subject ID
        output_path = output_dir / f"skull_strip_{subject_name}_{timestamp}.png"

        # Create visualization
        success = plot_skull_stripping_results(
            orig_path,
            brain_path,
            mask_path,
            output_path,
            title=f"Skull Stripping Result: {subject_name}"
        )

        if success:
            saved_paths.append(output_path)

    return saved_paths