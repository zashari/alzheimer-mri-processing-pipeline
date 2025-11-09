"""Deduplication utilities for data preparation."""

from __future__ import annotations

import pandas as pd
from typing import Optional
import warnings


def keep_scaled2(group: pd.DataFrame) -> pd.DataFrame:
    """
    Within (Subject, Visit) keep only *Scaled_2* rows if they exist.
    Otherwise keep *Scaled* rows. This ensures we use the highest quality scans.

    Args:
        group: DataFrame group for a specific (Subject, Visit) combination

    Returns:
        DataFrame with deduplicated rows
    """
    # Check if we have Scaled_2 rows
    if "Description" in group.columns:
        has_scaled2 = group["Description"].str.contains(r"Scaled_2", na=False).any()
        if has_scaled2:
            # Return only Scaled_2 rows
            return group[group["Description"].str.contains(r"Scaled_2", na=False)]

    # If no Scaled_2, return all rows (could be Scaled or other)
    return group


def deduplicate_scans(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Remove duplicate scans, preferring Scaled_2 over Scaled versions.

    Args:
        df: Input DataFrame with scan metadata
        verbose: If True, print deduplication statistics

    Returns:
        DataFrame with duplicates removed
    """
    initial_rows = len(df)

    # Check if there are actually duplicates
    duplicate_mask = df.duplicated(subset=["Subject", "Visit"], keep=False)
    if not duplicate_mask.any():
        if verbose:
            print("  No duplicate scans found")
        return df

    # Apply deduplication with warnings suppressed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="DataFrame.groupby with axis=1 is deprecated")

        # Use a different approach to avoid the include_groups issue
        result_list = []
        for (subject, visit), group in df.groupby(["Subject", "Visit"]):
            # Apply keep_scaled2 logic
            if "Description" in group.columns:
                has_scaled2 = group["Description"].str.contains(r"Scaled_2", na=False).any()
                if has_scaled2:
                    # Keep only Scaled_2 rows
                    filtered_group = group[group["Description"].str.contains(r"Scaled_2", na=False)]
                else:
                    # Keep all rows for this group
                    filtered_group = group
            else:
                filtered_group = group

            # If multiple rows still exist, keep the first one
            if len(filtered_group) > 1:
                filtered_group = filtered_group.iloc[[0]]

            result_list.append(filtered_group)

        # Combine all groups back together
        if result_list:
            df_dedup = pd.concat(result_list, ignore_index=True)
        else:
            df_dedup = pd.DataFrame()

    removed_rows = initial_rows - len(df_dedup)

    if verbose and removed_rows > 0:
        print(f"✓ Removed {removed_rows} duplicate scans (kept Scaled_2 where available)")
        print(f"  Rows after deduplication: {len(df_dedup)}")

    return df_dedup


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """
    Analyze duplicate scans in the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with duplicate analysis
    """
    duplicates_info = {}

    # Check for Subject+Visit duplicates
    if "Subject" not in df.columns or "Visit" not in df.columns:
        return {
            "total_duplicates": 0,
            "unique_subject_visit_pairs": 0,
            "scaled2_duplicates": 0,
            "scaled_duplicates": 0,
            "other_duplicates": 0
        }

    duplicate_mask = df.duplicated(subset=["Subject", "Visit"], keep=False)
    duplicate_df = df[duplicate_mask]

    duplicates_info["total_duplicates"] = len(duplicate_df)

    if len(duplicate_df) > 0:
        duplicates_info["unique_subject_visit_pairs"] = duplicate_df.groupby(["Subject", "Visit"]).ngroup().nunique()
    else:
        duplicates_info["unique_subject_visit_pairs"] = 0

    # Analyze by Description type
    if "Description" in df.columns and len(duplicate_df) > 0:
        scaled2_count = duplicate_df["Description"].str.contains(r"Scaled_2", na=False).sum()

        # Find Scaled but not Scaled_2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            scaled_only_mask = (
                duplicate_df["Description"].str.contains(r"Scaled", na=False) &
                ~duplicate_df["Description"].str.contains(r"Scaled_2", na=False)
            )
            scaled_count = scaled_only_mask.sum()

        other_count = len(duplicate_df) - scaled2_count - scaled_count

        duplicates_info["scaled2_duplicates"] = scaled2_count
        duplicates_info["scaled_duplicates"] = scaled_count
        duplicates_info["other_duplicates"] = other_count
    else:
        duplicates_info["scaled2_duplicates"] = 0
        duplicates_info["scaled_duplicates"] = 0
        duplicates_info["other_duplicates"] = 0

    return duplicates_info