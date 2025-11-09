"""Visualization utilities for template registration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from skimage import measure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import nibabel as nib
    import numpy as np
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


def visualize_registration_results(
    subject_id: str,
    visit: str,
    brain_slices: Dict[str, Path],
    mask_3d_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    alpha: float = 0.3,
    show_contour: bool = True,
    cmap_brain: str = 'gray',
    cmap_overlay: str = 'Reds'
) -> bool:
    """
    Visualize optimal brain slices with hippocampus overlay.

    Args:
        subject_id: Subject identifier
        visit: Visit identifier
        brain_slices: Dict of plane->path for brain slices
        mask_3d_path: Optional path to 3D hippocampus mask
        output_path: Where to save visualization
        alpha: Overlay transparency
        show_contour: Whether to show hippocampus contour
        cmap_brain: Colormap for brain
        cmap_overlay: Colormap for overlay

    Returns:
        Success status
    """
    if not MATPLOTLIB_AVAILABLE or not NIBABEL_AVAILABLE:
        return False

    try:
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        planes = ['axial', 'sagittal', 'coronal']

        # Load 3D mask if available
        mask_3d_data = None
        hippo_volume = 0
        if mask_3d_path and mask_3d_path.exists():
            mask_3d_img = nib.load(mask_3d_path)
            mask_3d_data = mask_3d_img.get_fdata()
            hippo_volume = np.sum(mask_3d_data)

        found_any = False

        for idx, plane in enumerate(planes):
            if plane not in brain_slices:
                # Empty subplot
                for row in range(2):
                    axes[row, idx].text(0.5, 0.5, 'Not found',
                                       ha='center', va='center')
                    axes[row, idx].axis('off')
                continue

            brain_path = brain_slices[plane]
            if not brain_path.exists():
                continue

            # Load brain slice
            brain_img = nib.load(brain_path)
            brain_data = brain_img.get_fdata().squeeze()

            # Extract slice index from filename
            slice_idx = _extract_slice_index(brain_path)

            # Top row: Brain only
            axes[0, idx].imshow(brain_data.T, cmap=cmap_brain, origin='lower')
            axes[0, idx].set_title(f"{plane.capitalize()} - Slice {slice_idx}")
            axes[0, idx].axis('off')

            # Bottom row: Brain with overlay if mask available
            axes[1, idx].imshow(brain_data.T, cmap=cmap_brain, origin='lower')

            if mask_3d_data is not None and slice_idx is not None:
                # Extract corresponding mask slice
                mask_slice = _extract_mask_slice(mask_3d_data, slice_idx, plane)

                # Add overlay
                masked_hippo = np.ma.masked_where(mask_slice.T < 0.5, mask_slice.T)
                axes[1, idx].imshow(masked_hippo, cmap=cmap_overlay,
                                   alpha=alpha, origin='lower')

                # Add contour if requested
                if show_contour and np.any(mask_slice > 0):
                    contours = measure.find_contours(mask_slice.T, 0.5)
                    for contour in contours:
                        axes[1, idx].plot(contour[:, 1], contour[:, 0],
                                        'lime', linewidth=2)

                # Add area text
                area = np.sum(mask_slice)
                axes[1, idx].text(0.5, -0.05, f"Area: {area:.0f} voxels",
                                transform=axes[1, idx].transAxes,
                                ha='center', fontsize=10)

            axes[1, idx].set_title("With Hippocampus Overlay")
            axes[1, idx].axis('off')

            found_any = True

        # Add title
        title = f"Optimal Hippocampus Slices: {subject_id}_{visit}"
        if hippo_volume > 0:
            title += f"\n3D Hippocampus Volume: {hippo_volume:.0f} voxels"
        title += "\nTop: Brain only | Bottom: Brain with hippocampus overlay"

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        # Save figure
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.close()
        return found_any

    except Exception as e:
        print(f"Visualization error: {e}")
        return False


def create_batch_visualization(
    results: List[Dict],
    output_dir: Path,
    max_samples: int = 3
) -> List[Path]:
    """
    Create visualizations for a batch of processed subjects.

    Args:
        results: List of processing results
        output_dir: Output directory for visualizations
        max_samples: Maximum number to visualize

    Returns:
        List of created visualization paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Filter successful results with 3D masks
    successful = [r for r in results
                  if r.get('status') == 'success' and r.get('hippo_mask_3d')]

    for i, result in enumerate(successful[:max_samples]):
        # Prepare slice paths
        brain_slices = {}
        for plane, info in result.get('slices', {}).items():
            if 'output_path' in info:
                brain_slices[plane] = Path(info['output_path'])

        # Create visualization
        viz_path = output_dir / f"registration_{result['subject']}_{result['visit']}.png"

        mask_3d_path = None
        if result.get('hippo_mask_3d'):
            mask_3d_path = Path(result['hippo_mask_3d'])

        success = visualize_registration_results(
            result['subject'],
            result['visit'],
            brain_slices,
            mask_3d_path=mask_3d_path,
            output_path=viz_path
        )

        if success:
            created_files.append(viz_path)

    return created_files


def _extract_slice_index(filepath: Path) -> Optional[int]:
    """Extract slice index from filename."""
    filename = filepath.stem  # Remove extension

    # Look for pattern _x{number}
    parts = filename.split('_x')
    if len(parts) >= 2:
        try:
            return int(parts[-1].split('_')[0])
        except ValueError:
            pass

    return None


def _extract_mask_slice(mask_3d: np.ndarray, slice_idx: int, plane: str) -> np.ndarray:
    """Extract a 2D slice from 3D mask."""
    if plane == 'axial':
        return mask_3d[:, :, slice_idx]
    elif plane == 'sagittal':
        return mask_3d[slice_idx, :, :]
    elif plane == 'coronal':
        return mask_3d[:, slice_idx, :]
    else:
        raise ValueError(f"Invalid plane: {plane}")