"""Visualization utilities for image enhancement sub-stage."""

from __future__ import annotations

import random
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


def create_enhancement_comparison_visualization(
    input_root: Path,
    output_root: Path,
    slice_types: List[str],
    splits: List[str],
    groups: List[str],
    required_visits: List[str],
    visualization_output_dir: Path,
    formatter,
    num_samples: int = 6,
    seed: int = 42
) -> List[Path]:
    """
    Create enhancement comparison visualization.

    Args:
        input_root: Directory containing original PNG files
        output_root: Directory containing enhanced PNG files
        slice_types: List of slice types
        splits: List of splits
        groups: List of groups
        required_visits: List of required visits
        visualization_output_dir: Directory to save visualization
        formatter: Formatter instance for output
        num_samples: Number of sample images to show
        seed: Random seed for sample selection

    Returns:
        List of created visualization file paths
    """
    if not MATPLOTLIB_AVAILABLE or not CV2_AVAILABLE or not NUMPY_AVAILABLE:
        if formatter and not formatter.json_only:
            formatter.warning("matplotlib/cv2/numpy not available, skipping visualizations")
        return []

    visualization_output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Set seed for reproducibility
    random.seed(seed)

    try:
        samples = []

        # Collect samples from different classes and visits
        for cls in groups:
            # Use coronal/train for visualization (as in reference)
            inp_dir = input_root / "coronal" / "train" / cls
            out_dir = output_root / "coronal" / "train" / cls

            if not inp_dir.exists() or not out_dir.exists():
                continue

            # Find files for different visits
            for visit in required_visits:
                inp_files = [f for f in inp_dir.glob("*.png") if f"_{visit}_" in f.name]
                if inp_files:
                    inp_file = random.choice(inp_files)
                    out_file = out_dir / inp_file.name
                    if out_file.exists():
                        samples.append((inp_file, out_file, f"{cls}_{visit}"))
                        break

        if not samples:
            if formatter and not formatter.json_only:
                formatter.warning("No enhancement samples found for visualization")
            return []

        # Sample random subset
        samples = random.sample(samples, min(num_samples, len(samples)))

        # Create visualization
        fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
        if len(samples) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            "Sequential Temporal Enhancement Comparison",
            fontsize=16,
            weight="bold"
        )

        for i, (orig_p, en_p, label) in enumerate(samples):
            try:
                o = cv2.imread(str(orig_p), cv2.IMREAD_GRAYSCALE)
                e = cv2.imread(str(en_p), cv2.IMREAD_GRAYSCALE)

                if o is None or e is None:
                    continue

                axes[0, i].imshow(o, cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title(f"Original\n{label}")

                axes[1, i].imshow(e, cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title("Enhanced")
            except Exception as e:
                if formatter and formatter.verbose:
                    formatter.warning(f"Error visualizing {orig_p.name}: {e}")
                continue

        plt.tight_layout()

        # Save visualization
        viz_path = visualization_output_dir / "enhancement_comparison.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        created_files.append(viz_path)

        return created_files

    except Exception as e:
        if formatter and not formatter.json_only:
            formatter.warning(f"Visualization creation failed: {e}")
        return []

