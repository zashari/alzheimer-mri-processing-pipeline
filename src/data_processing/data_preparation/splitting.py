from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import pandas as pd


def subjects_with_complete_visits(df: pd.DataFrame, required_visits: List[str]) -> List[str]:
    """
    Find subjects that have all required visits.

    Args:
        df: DataFrame with Subject and Visit columns
        required_visits: List of required visit codes

    Returns:
        List of subject IDs with complete visits
    """
    complete = []
    by_subj = df.groupby("Subject")

    for subj, g in by_subj:
        visits = set(g["Visit"].astype(str).str.lower().tolist())
        if all(v.lower() in visits for v in required_visits):
            complete.append(subj)

    return complete


def analyze_visit_completeness(
    df: pd.DataFrame,
    required_visits: List[str]
) -> Tuple[List[str], List[Tuple[str, str, set, set]], Dict[str, List[str]]]:
    """
    Comprehensive analysis of visit completeness.

    Args:
        df: DataFrame with subject data
        required_visits: List of required visits

    Returns:
        Tuple of:
        - List of complete subject IDs
        - List of incomplete subjects with (subject, group, available_visits, missing_visits)
        - Dictionary of complete subjects by group
    """
    complete_subjects = []
    incomplete_subjects = []
    complete_by_group = defaultdict(list)

    # Get subject groups
    subject_groups = df.groupby("Subject")["Group"].first().to_dict()

    # Analyze each subject
    for subject, group_df in df.groupby("Subject"):
        visits = set(group_df["Visit"].astype(str).str.lower().tolist())
        required_set = set(v.lower() for v in required_visits)
        group = subject_groups[subject]

        if required_set.issubset(visits):
            complete_subjects.append(subject)
            complete_by_group[group].append(subject)
        else:
            available = visits & required_set
            missing = required_set - visits
            incomplete_subjects.append((subject, group, available, missing))

    return complete_subjects, incomplete_subjects, dict(complete_by_group)


def check_split_feasibility(
    complete_by_group: Dict[str, List[str]],
    min_subjects_per_group: int = 7
) -> List[Tuple[str, int, bool]]:
    """
    Check if each group has enough subjects for splitting.

    Args:
        complete_by_group: Dictionary of subjects by group
        min_subjects_per_group: Minimum subjects needed per group

    Returns:
        List of (group, count, is_feasible) tuples
    """
    feasible_groups = []

    for group, subjects in complete_by_group.items():
        count = len(subjects)
        is_feasible = count >= min_subjects_per_group
        feasible_groups.append((group, count, is_feasible))

    return feasible_groups


def stratified_split_with_redistribution(
    subjects: List[str],
    df: pd.DataFrame,
    stratify_by: str,
    ratios: Tuple[float, float, float],
    seed: Optional[int],
    shuffle: bool = True
) -> Dict[str, List[str]]:
    """
    Enhanced stratified split with redistribution for small groups.

    Args:
        subjects: List of subject IDs to split
        df: DataFrame with subject metadata
        stratify_by: Column name to stratify by
        ratios: (train_ratio, val_ratio, test_ratio)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle subjects before splitting

    Returns:
        Dictionary with train/val/test subject lists
    """
    # Map subject -> group value
    sub_group = {
        subj: df.loc[df["Subject"] == subj, stratify_by].iloc[0]
        for subj in subjects
    }

    by_group: Dict[str, List[str]] = defaultdict(list)
    for subj, grp in sub_group.items():
        by_group[str(grp)].append(subj)

    if seed is not None:
        random.seed(seed)

    train_ratio, val_ratio, test_ratio = ratios
    out = {"train": [], "val": [], "test": []}

    # Process each group
    for grp, subs in by_group.items():
        items = subs[:]
        if shuffle:
            random.shuffle(items)

        n_total = len(items)

        # Calculate initial allocation
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Remaining subjects go to test

        # Redistribution logic for small groups
        if n_total >= 3:
            # Ensure each split gets at least 1 subject if possible
            if n_train == 0 and n_total >= 3:
                n_train = 1
            if n_val == 0 and n_total >= 3:
                n_val = 1
            if n_test == 0 and n_total >= 3:
                n_test = 1

            # Rebalance if we over-allocated
            total_allocated = n_train + n_val + n_test
            if total_allocated > n_total:
                # Reduce largest allocation first
                diffs = [
                    (n_train - int(n_total * train_ratio), "train"),
                    (n_val - int(n_total * val_ratio), "val"),
                    (n_test - int(n_total * test_ratio), "test")
                ]
                diffs.sort(reverse=True)

                excess = total_allocated - n_total
                for _, split_name in diffs:
                    if excess <= 0:
                        break
                    if split_name == "train" and n_train > 1:
                        reduce = min(excess, n_train - 1)
                        n_train -= reduce
                        excess -= reduce
                    elif split_name == "val" and n_val > 1:
                        reduce = min(excess, n_val - 1)
                        n_val -= reduce
                        excess -= reduce
                    elif split_name == "test" and n_test > 1:
                        reduce = min(excess, n_test - 1)
                        n_test -= reduce
                        excess -= reduce

        elif n_total == 2:
            # Special case: 2 subjects
            n_train = 1
            n_val = 1
            n_test = 0
        elif n_total == 1:
            # Special case: 1 subject (put in train)
            n_train = 1
            n_val = 0
            n_test = 0

        # Allocate subjects
        out["train"].extend(items[:n_train])
        out["val"].extend(items[n_train:n_train + n_val])
        out["test"].extend(items[n_train + n_val:])

    return out


def calculate_split_statistics(
    split_dict: Dict[str, List[str]],
    df: pd.DataFrame,
    stratify_by: str
) -> Dict[str, Dict[str, Tuple[int, float]]]:
    """
    Calculate detailed statistics for the split.

    Args:
        split_dict: Dictionary with train/val/test subject lists
        df: DataFrame with subject metadata
        stratify_by: Column used for stratification

    Returns:
        Dictionary with statistics per split and group
    """
    stats = {}

    # Get unique groups
    groups = df[stratify_by].unique()

    for split_name, subjects in split_dict.items():
        split_df = df[df["Subject"].isin(subjects)]
        group_counts = split_df.groupby(stratify_by)["Subject"].nunique()

        stats[split_name] = {}
        total_in_split = len(subjects)

        for group in groups:
            if group in group_counts.index:
                count = group_counts[group]
                percentage = (count / total_in_split * 100) if total_in_split > 0 else 0
            else:
                count = 0
                percentage = 0

            stats[split_name][group] = (count, percentage)

    return stats


def check_leakage(split_dict: Dict[str, List[str]]) -> int:
    """
    Check for subject leakage across splits.

    Args:
        split_dict: Dictionary with train/val/test subject lists

    Returns:
        Number of subjects that appear in multiple splits
    """
    train_set = set(split_dict["train"])
    val_set = set(split_dict["val"])
    test_set = set(split_dict["test"])

    # Check for overlaps
    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    total_overlap = len(train_val | train_test | val_test)

    return total_overlap


def materialize_manifests(
    df: pd.DataFrame,
    split: Dict[str, List[str]],
    required_visits: List[str],
    output_dir: Path
) -> Dict[str, int]:
    """
    Write manifest CSV files for each split.

    Args:
        df: DataFrame with all data
        split: Dictionary with train/val/test subject lists
        required_visits: List of required visits to include
        output_dir: Directory to save manifests

    Returns:
        Dictionary with row counts per file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to subjects in splits and required visits
    all_subjects = set(split["train"]) | set(split["val"]) | set(split["test"])
    df_filtered = df[df["Subject"].isin(all_subjects)]

    # Filter to required visits (case-insensitive)
    required_lower = [v.lower() for v in required_visits]
    df_filtered = df_filtered[df_filtered["Visit"].str.lower().isin(required_lower)]

    # Add Split column based on subject
    def _split_label(subj: str) -> str:
        if subj in split["train"]:
            return "train"
        elif subj in split["val"]:
            return "val"
        else:
            return "test"

    df_filtered = df_filtered.copy()
    df_filtered["Split"] = df_filtered["Subject"].apply(_split_label)

    file_counts = {}

    # Write metadata_split.csv (all data with Split column)
    metadata_path = output_dir / "metadata_split.csv"
    df_filtered.to_csv(metadata_path, index=False)
    file_counts["metadata_split.csv"] = len(df_filtered)

    # Write per-split CSVs
    for split_name in ["train", "val", "test"]:
        sub_df = df_filtered[df_filtered["Split"] == split_name]
        file_path = output_dir / f"{split_name}.csv"
        sub_df.to_csv(file_path, index=False)
        file_counts[f"{split_name}.csv"] = len(sub_df)

    return file_counts


# Keep original function name for backwards compatibility
def stratified_split(
    subjects: List[str],
    df: pd.DataFrame,
    stratify_by: str,
    ratios: Tuple[float, float, float],
    seed: Optional[int],
    shuffle: bool = True
) -> Dict[str, List[str]]:
    """
    Original stratified split function (kept for compatibility).
    Delegates to enhanced version.
    """
    return stratified_split_with_redistribution(
        subjects, df, stratify_by, ratios, seed, shuffle
    )