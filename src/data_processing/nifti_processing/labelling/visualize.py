"""Visualization for labelling sub-stage."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_visualizations(
    results: Dict[str, Dict],
    output_dir: Path,
    formatter,
    is_test: bool = False
) -> List[Path]:
    """
    Create visualizations for labelling results.

    Args:
        results: Dictionary of results by slice type
        output_dir: Output directory for visualizations
        formatter: Formatter instance for output
        is_test: Whether this is test mode

    Returns:
        List of created visualization file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        if formatter and not formatter.json_only:
            formatter.warning("matplotlib not available, skipping visualizations")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Create distribution plots
    for slice_type, result in results.items():
        temporal_stats = result.get("temporal_stats", {})
        processed_subjects = result.get("processed_subjects", {})

        if not temporal_stats:
            continue

        # Create distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Labelling Distribution: {slice_type.upper()}", fontsize=14, fontweight="bold")

        # Plot 1: Files by Split and Group
        ax1 = axes[0]
        splits = list(temporal_stats.keys())
        groups = set()
        for split_data in temporal_stats.values():
            groups.update(split_data.keys())
        groups = sorted(list(groups))

        x = np.arange(len(splits))
        width = 0.35

        for i, group in enumerate(groups):
            values = [temporal_stats.get(split, {}).get(group, 0) for split in splits]
            offset = (i - len(groups) / 2 + 0.5) * width / len(groups)
            ax1.bar(x + offset, values, width / len(groups), label=group)

        ax1.set_xlabel("Split")
        ax1.set_ylabel("Number of Files")
        ax1.set_title("Files by Split and Group")
        ax1.set_xticks(x)
        ax1.set_xticklabels(splits)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Subjects by Split and Group
        ax2 = axes[1]
        for i, group in enumerate(groups):
            values = [processed_subjects.get(split, {}).get(group, 0) for split in splits]
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
        fig.suptitle("Overall Labelling Summary", fontsize=14, fontweight="bold")

        slice_types = list(results.keys())
        total_files = []
        total_subjects = []

        for slice_type in slice_types:
            result = results[slice_type]
            temporal_stats = result.get("temporal_stats", {})
            processed_subjects = result.get("processed_subjects", {})

            files = sum(
                sum(group_data.values())
                for group_data in temporal_stats.values()
            )
            subjects = sum(
                sum(group_data.values())
                for group_data in processed_subjects.values()
            )

            total_files.append(files)
            total_subjects.append(subjects)

        x = np.arange(len(slice_types))
        width = 0.35

        ax.bar(x - width/2, total_files, width, label="Files", alpha=0.8)
        ax.bar(x + width/2, total_subjects, width, label="Subjects", alpha=0.8)

        ax.set_xlabel("Slice Type")
        ax.set_ylabel("Count")
        ax.set_title("Total Files and Subjects by Slice Type")
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






