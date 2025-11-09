from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def plot_subject_visits(meta, base_dir: str | Path, group_col: str = "Group", output_dir: str | Path | None = None) -> None:
    """
    Plot brain scans for one random subject from each diagnostic group.
    Shows exactly 3 visits (SC, M06, M12) per subject.
    Creates one PNG file per group.
    """
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import random
        import glob
        from datetime import datetime
    except Exception as e:  # pragma: no cover
        print("[viz] Visualization dependencies not available; skipping plots")
        return

    base_dir = os.path.abspath(os.path.normpath(str(base_dir)))
    REQUIRED_VISITS = ["sc", "m06", "m12"]  # The visits we want to visualize

    # Find one subject per group that has complete visits
    groups = meta[group_col].unique()
    selected_subjects: Dict[str, Tuple[str, Dict[str, str]]] = {}

    for group in groups:
        # Get all subjects in this group
        group_subjects = meta.loc[meta[group_col] == group, "Subject"].unique().tolist()
        random.shuffle(group_subjects)

        # Find a subject with complete visits
        for subj in group_subjects:
            # Check if this subject has all required visits
            subject_visits = meta.loc[meta["Subject"] == subj, "Visit"].str.lower().unique()

            # Check if all required visits are present
            if all(visit in subject_visits for visit in REQUIRED_VISITS):
                # Get the subject's metadata for the required visits
                sub_meta = meta.loc[
                    (meta["Subject"] == subj) &
                    (meta["Visit"].str.lower().isin(REQUIRED_VISITS))
                ].copy()

                # Create mapping of visit to date for finding files
                sub_meta["DateStr"] = sub_meta["Acq Date"].dt.strftime("%Y-%m-%d")
                visit_to_date = {}
                for visit in REQUIRED_VISITS:
                    visit_rows = sub_meta[sub_meta["Visit"].str.lower() == visit]
                    if not visit_rows.empty:
                        visit_to_date[visit] = visit_rows.iloc[0]["DateStr"]

                # Verify NIfTI files exist for this subject
                subject_dir = os.path.join(base_dir, subj)
                if os.path.exists(subject_dir):
                    patterns = [
                        os.path.join(subject_dir, "**", "*.nii"),
                        os.path.join(subject_dir, "**", "*.nii.gz"),
                    ]
                    found_files: List[str] = []
                    for pattern in patterns:
                        found_files.extend(glob.glob(pattern, recursive=True))

                    if found_files:  # Subject has NIfTI files
                        selected_subjects[group] = (subj, visit_to_date)
                        break

    # Create visualization for each selected subject
    for group_label, (subject_id, visit_to_date) in selected_subjects.items():
        # Find NIfTI files for this subject
        subject_dir = os.path.join(base_dir, subject_id)
        nii_files: List[str] = []
        for ext in ["*.nii", "*.nii.gz"]:
            pattern = os.path.join(subject_dir, "**", ext)
            nii_files.extend([f for f in glob.glob(pattern, recursive=True) if os.path.exists(f)])

        if not nii_files:
            continue

        # Find the specific files for each required visit
        visit_files = {}
        for visit, date_str in visit_to_date.items():
            # Find file that matches this visit's date
            for nii_file in nii_files:
                if date_str in nii_file:
                    visit_files[visit] = nii_file
                    break

            # If not found by date, try to find by visit code in path
            if visit not in visit_files:
                for nii_file in nii_files:
                    if visit.lower() in nii_file.lower() or visit.upper() in nii_file:
                        visit_files[visit] = nii_file
                        break

        # Only proceed if we found files for all three visits
        if len(visit_files) != 3:
            print(f"[viz] Warning: Could not find all 3 visits for {subject_id} (found {len(visit_files)})")
            continue

        # Get reference slice index from SC visit (use this same slice for all visits)
        ref_file = visit_files.get("sc", list(visit_files.values())[0])
        ref_data = nib.load(ref_file).get_fdata()
        # Use the middle coronal slice as reference
        reference_slice_idx = ref_data.shape[1] // 2

        # Create figure with exactly 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot each visit in order: SC, M06, M12
        for idx, visit in enumerate(REQUIRED_VISITS):
            if visit in visit_files:
                nii_path = visit_files[visit]
                data = nib.load(nii_path).get_fdata()

                # Use the SAME slice index for all visits for better comparison
                # Make sure the slice index is valid for this volume
                slice_idx = min(reference_slice_idx, data.shape[1] - 1)
                coronal_slice = data[:, slice_idx, :]

                axes[idx].imshow(np.rot90(coronal_slice), cmap="gray")
                axes[idx].set_title(f"{visit.upper()}", fontsize=12)
                axes[idx].axis("off")
            else:
                # This shouldn't happen if we checked correctly
                axes[idx].text(0.5, 0.5, "Missing", ha='center', va='center')
                axes[idx].axis("off")

        plt.suptitle(f"{group_label} | Subject: {subject_id}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save or show the figure
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp to avoid overwrites
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"subject_visits_{group_label}_{subject_id}_{timestamp}.png"
            save_path = output_path / filename

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[viz] Saved visualization: {save_path}")
            plt.close()  # Close to free memory
        else:
            plt.show()