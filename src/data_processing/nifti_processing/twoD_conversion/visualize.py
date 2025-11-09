"""Visualization for 2D conversion sub-stage."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    MATPLOTLIB_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    PIL_AVAILABLE = False


def create_visualizations(
    results: Dict[str, Dict],
    output_dir: Path,
    formatter,
    is_test: bool = False
) -> List[Path]:
    """
    Create visualizations for 2D conversion results.

    Args:
        results: Dictionary of results by slice type
        output_dir: Output directory for visualizations
        formatter: Formatter instance for output
        is_test: Whether this is test mode

    Returns:
        List of created visualization file paths
    """
    if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
        if formatter and not formatter.json_only:
            formatter.warning("matplotlib/PIL not available, skipping visualizations")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Create distribution plots
    for slice_type, result in results.items():
        stats = result.get("stats", {})
        processed_subjects = result.get("processed_subjects", {})

        if not stats:
            continue

        # Create distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"2D Conversion Distribution: {slice_type.upper()}", fontsize=14, fontweight="bold")

        # Plot 1: PNG files by Split and Group
        ax1 = axes[0]
        splits = list(processed_subjects.keys()) if processed_subjects else []
        groups = set()
        if processed_subjects:
            for split_data in processed_subjects.values():
                if isinstance(split_data, dict):
                    groups.update(split_data.keys())
        groups = sorted(list(groups)) if groups else []

        if splits and groups:
            x = np.arange(len(splits))
            width = 0.35

            for i, group in enumerate(groups):
                values = [
                    stats.get(group, {}).get("saved", 0)
                    for split in splits
                ]
                offset = (i - len(groups) / 2 + 0.5) * width / len(groups)
                ax1.bar(x + offset, values, width / len(groups), label=group)

            ax1.set_xlabel("Split")
            ax1.set_ylabel("Number of PNG Files")
            ax1.set_title("PNG Files by Split and Group")
            ax1.set_xticks(x)
            ax1.set_xticklabels(splits)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Subjects by Split and Group
        ax2 = axes[1]
        if splits and groups:
            for i, group in enumerate(groups):
                values = [
                    processed_subjects.get(split, {}).get(group, 0)
                    for split in splits
                ]
                offset = (i - len(groups) / 2 + 0.5) * width / len(groups)
                ax2.bar(x + offset, values, width / len(groups), label=group)

            ax2.set_xlabel("Split")
            ax2.set_ylabel("Number of Subjects")
            ax2.set_title("Subjects by Split and Group")
            ax2.set_xticks(x)
            ax2.set_xticklabels(splits)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        filename = f"distribution_{slice_type}.png" if not is_test else f"test_distribution_{slice_type}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        created_files.append(filepath)

    # Create overall summary plot if multiple slice types
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Overall 2D Conversion Summary", fontsize=14, fontweight="bold")

        slice_types = list(results.keys())
        total_png = []
        total_subjects = []

        for slice_type in slice_types:
            result = results[slice_type]
            stats = result.get("stats", {})
            processed_subjects = result.get("processed_subjects", {})

            png = sum(
                sum(group_data.get("saved", 0) for group_data in stats.values())
                if isinstance(stats, dict) else 0
            )
            subjects = sum(
                sum(group_data.values())
                for split_data in processed_subjects.values()
                for group_data in split_data.values()
            )

            total_png.append(png)
            total_subjects.append(subjects)

        x = np.arange(len(slice_types))
        width = 0.35

        ax.bar(x - width/2, total_png, width, label="PNG Files", alpha=0.8)
        ax.bar(x + width/2, total_subjects, width, label="Subjects", alpha=0.8)

        ax.set_xlabel("Slice Type")
        ax.set_ylabel("Count")
        ax.set_title("Total PNG Files and Subjects by Slice Type")
        ax.set_xticks(x)
        ax.set_xticklabels([st.upper() for st in slice_types])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = "overall_summary.png" if not is_test else "test_overall_summary.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        created_files.append(filepath)

    if formatter and not formatter.json_only and created_files:
        formatter.print(f"[blue]📸 Saved {len(created_files)} visualizations to {output_dir}[/blue]")

    return created_files

