"""Output formatting utilities for data preparation using Rich library."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.padding import Padding
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn


class DataPrepFormatter:
    """Handles formatted output for data preparation stage using Rich."""

    def __init__(self, verbose: bool = False, quiet: bool = False, json_only: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.json_only = json_only
        self.start_time = time.time()
        self.report_data: Dict[str, Any] = {}

        # Initialize Rich console
        self.console = Console()

        # Track warnings for final outcome
        self.has_warnings = False
        self.warnings_list = []

    def header(self, action: str, **kwargs) -> None:
        """Print header section."""
        if self.json_only or self.quiet:
            return

        # Build header lines
        header_lines = []
        header_lines.append("[cyan]═" * 70 + "[/cyan]")
        header_lines.append(f"[cyan] Data Preparation • {action.title()}[/cyan]")

        # Build info line based on action
        timestamp = datetime.now().strftime("%H:%M:%S")
        info_parts = [f"Started: {timestamp}"]

        if action == "split":
            seed = kwargs.get('seed', 42)
            stratify_by = kwargs.get('stratify_by', 'Group')
            info_parts.append(f"Seed: {seed}")
            info_parts.append(f"Strategy: subject-level stratified({stratify_by})")

        info_line = "  •  ".join(info_parts)
        header_lines.append(f" [dim]{info_line}[/dim]")
        header_lines.append("[cyan]═" * 70 + "[/cyan]")

        for line in header_lines:
            self.console.print(line)
        self.console.print()

        # Store in report
        self.report_data.update({
            "tool": "data_preparation",
            "action": action,
            "started_at": datetime.now().isoformat(),
            "parameters": kwargs
        })

    def dataset_info(self, subjects: int, rows: int, modality: str = "MRI",
                    manifest_dir: Optional[str] = None) -> None:
        """Print dataset information."""
        if self.json_only:
            return

        self.console.print("[blue][INFO][/blue] Dataset")
        self.console.print(f"  • Subjects: {subjects:,}   • Rows: {rows:,}   • Modality: {modality}")
        if manifest_dir:
            self.console.print(f"  • Manifest dir: {manifest_dir}")
        self.console.print()

        # Store in report
        self.report_data["dataset"] = {
            "subjects": subjects,
            "rows": rows,
            "modality": modality,
            "manifest_dir": manifest_dir
        }

    def overview(self, rows: int, subjects: int, modalities: Dict[str, float]) -> None:
        """Print overview for analyze action."""
        if self.json_only:
            return

        self.console.print("[blue][OVERVIEW][/blue]")

        # Format modalities
        mod_str = ", ".join([f"{k}({v:.0f}%)" for k, v in modalities.items()])
        self.console.print(f"  Rows: {rows:,}   • Subjects: {subjects:,}   • Modalities: {mod_str}")
        self.console.print()

    def group_distribution(self, groups: Dict[str, Tuple[int, float]]) -> None:
        """Print group distribution."""
        if self.json_only:
            return

        self.console.print("[blue][GROUP][/blue] Distribution (rows)")
        parts = []
        for group, (count, pct) in groups.items():
            parts.append(f"{group}: {count:,} ({pct:.1f}%)")
        self.console.print("  " + "   ".join(parts))
        self.console.print()

        # Store in report
        self.report_data["groups"] = groups

    def sex_distribution(self, sex_dist: Dict[str, Tuple[int, float]]) -> None:
        """Print sex distribution."""
        if self.json_only:
            return

        parts = []
        for sex, (count, pct) in sex_dist.items():
            parts.append(f"{sex}: {count:,} ({pct:.1f}%)")

        self.console.print("[blue][SEX][/blue] " + "   ".join(parts))
        self.console.print()

    def visit_distribution(self, visits: Dict[str, int]) -> None:
        """Print visit distribution."""
        if self.json_only:
            return

        self.console.print("[blue][VISIT][/blue] ", end="")
        parts = []
        for visit, count in visits.items():
            pct = count / sum(visits.values()) * 100
            parts.append(f"{visit}: {count:,} ({pct:.1f}%)")
        self.console.print("  ".join(parts))
        self.console.print()

    def description_summary(self, descriptions: List[Tuple[str, int]],
                           show_all: bool = False) -> None:
        """Print description summary."""
        if self.json_only:
            return

        self.console.print("[blue][DESCRIPTION][/blue] Top levels by count")

        limit = len(descriptions) if show_all else min(6, len(descriptions))
        for i, (desc, count) in enumerate(descriptions[:limit], 1):
            # Truncate long descriptions
            if len(desc) > 50:
                desc = desc[:47] + "..."
            self.console.print(f"  {i}) {desc} {'.' * (55 - len(desc))} {count:,}")

        remaining = len(descriptions) - limit
        if remaining > 0 and not show_all:
            self.console.print(f"  (+{remaining} more — use --show-all to expand)")
        self.console.print()

    def validation_summary(self, rare_categories: int = 0, missing_values: int = 0,
                         class_imbalance: Optional[str] = None) -> None:
        """Print validation summary."""
        if self.json_only:
            return

        self.console.print("[blue][VALIDATION][/blue]")

        if rare_categories > 0:
            self.console.print(f"  • Rare categories (<0.5%): {rare_categories}  (flagged)")
            self.has_warnings = True

        missing_str = "none" if missing_values == 0 else str(missing_values)
        self.console.print(f"  • Missing values: {missing_str}")

        if class_imbalance:
            self.console.print(f"  • Class imbalance (Group): {class_imbalance}")
            if "high" in class_imbalance.lower():
                self.has_warnings = True

        self.console.print()

    def split_results(self, train_count: int, val_count: int, test_count: int,
                     total_subjects: int) -> None:
        """Print split results."""
        if self.json_only:
            return

        self.console.print("[blue][RESULT][/blue] Split sizes")

        train_pct = train_count / total_subjects * 100
        val_pct = val_count / total_subjects * 100
        test_pct = test_count / total_subjects * 100

        self.console.print(f"  train: {train_count:,}  ({train_pct:.1f}%)")
        self.console.print(f"  val:   {val_count:,}  ({val_pct:.1f}%)")
        self.console.print(f"  test:  {test_count:,}  ({test_pct:.1f}%)")
        self.console.print()

        # Store in report
        self.report_data["split_sizes"] = {
            "train": {"count": train_count, "percentage": train_pct},
            "val": {"count": val_count, "percentage": val_pct},
            "test": {"count": test_count, "percentage": test_pct}
        }

    def class_balance_table(self, split_groups: Dict[str, Dict[str, Tuple[int, float]]],
                           overall_dist: Dict[str, float]) -> None:
        """Print class balance table."""
        if self.json_only:
            return

        self.console.print("[blue][CHECK][/blue] Class balance by Group (subjects, % of split)")

        # Create table
        table = Table(box=None, padding=(0, 2))
        table.add_column("", style="cyan")
        table.add_column("train", justify="right")
        table.add_column("val", justify="right")
        table.add_column("test", justify="right")
        table.add_column("total", justify="right", style="dim")

        # Add rows for each group
        for group in sorted(overall_dist.keys()):
            row = [group]
            for split in ["train", "val", "test"]:
                if split in split_groups and group in split_groups[split]:
                    count, pct = split_groups[split][group]
                    row.append(f"{count} ({pct:.1f})")
                else:
                    row.append("0 (0.0)")

            # Add total column
            total_pct = overall_dist[group]
            row.append(f"({total_pct:.1f}%)")
            table.add_row(*row)

        self.console.print(Padding(table, (0, 2)))

        # Check for imbalance
        max_delta = 0
        max_delta_info = None
        for split, groups in split_groups.items():
            for group, (count, pct) in groups.items():
                delta = abs(pct - overall_dist[group])
                if delta > max_delta:
                    max_delta = delta
                    max_delta_info = (group, split, delta)

        if max_delta > 5.0:  # Warning threshold
            group, split, delta = max_delta_info
            self.console.print(f"[yellow][WARN][/yellow] Max delta vs overall distribution: {delta:.1f}% ({group} in {split})")
            self.has_warnings = True
            self.warnings_list.append(f"imbalance in {split}")

        self.console.print()

    def data_availability_analysis(self, complete_subjects: List[str],
                                  incomplete_subjects: List[Tuple[str, str, set, set]],
                                  complete_by_group: Dict[str, List[str]],
                                  initial_subjects: int) -> None:
        """Print comprehensive data availability analysis."""
        if self.json_only:
            return

        self.console.print("[blue]🔍 DATA AVAILABILITY ANALYSIS[/blue]")
        self.console.print("=" * 60)

        total_subjects = len(complete_subjects) + len(incomplete_subjects)
        retention_rate = len(complete_subjects) / initial_subjects * 100

        self.console.print(f"[blue][cyan]📊 VISIT COMPLETENESS ANALYSIS:[/cyan][/blue]")
        self.console.print(f"   • Total subjects analyzed: {total_subjects:,}")
        self.console.print(f"   • Complete sequences: {len(complete_subjects):,}")
        self.console.print(f"   • Incomplete sequences: {len(incomplete_subjects):,}")
        self.console.print(f"   • Data retention rate: [green]{retention_rate:.1f}%[/green]")
        self.console.print()

        # Complete subjects by group
        self.console.print(f"[blue][cyan]📋 COMPLETE SUBJECTS BY DIAGNOSTIC GROUP:[/cyan][/blue]")
        total_complete = 0
        for group in sorted(complete_by_group.keys()):
            count = len(complete_by_group[group])
            total_complete += count
            self.console.print(f"   • {group}: {count} subjects")
        self.console.print(f"   • [bold]Total: {total_complete} subjects[/bold]")
        self.console.print()

        # Show sample of incomplete subjects
        if incomplete_subjects:
            self.console.print(f"[yellow]⚠️  INCOMPLETE SUBJECTS (sample):[/yellow]")
            for i, (subj, group, available, missing) in enumerate(incomplete_subjects[:3]):
                self.console.print(f"   • {subj} ({group}): has {sorted(available)}, missing [red]{sorted(missing)}[/red]")
            if len(incomplete_subjects) > 3:
                self.console.print(f"   [dim]... and {len(incomplete_subjects) - 3} more[/dim]")
            self.console.print()

        # Store in report
        self.report_data["data_availability"] = {
            "complete_subjects": len(complete_subjects),
            "incomplete_subjects": len(incomplete_subjects),
            "retention_rate": retention_rate,
            "complete_by_group": {k: len(v) for k, v in complete_by_group.items()}
        }

    def feasibility_check(self, feasible_groups: List[Tuple[str, int, bool]],
                         min_needed: int = 7) -> None:
        """Print split feasibility check."""
        if self.json_only:
            return

        self.console.print("[blue]🎯 SPLIT FEASIBILITY CHECK:[/blue]")

        all_feasible = True
        for group, count, feasible in feasible_groups:
            status = "[green]✅ FEASIBLE[/green]" if feasible else "[red]❌ INSUFFICIENT[/red]"
            self.console.print(f"   • {group}: {count} subjects (need ≥{min_needed}) - {status}")
            if not feasible:
                all_feasible = False

        if not all_feasible:
            self.console.print()
            self.console.print("[red]❌ CRITICAL: Some groups have insufficient subjects![/red]")
            self.console.print("   Consider using a different split strategy for groups with few subjects.")
            self.has_warnings = True
        else:
            self.console.print()
            self.console.print("[green]✅ SUCCESS: All groups have sufficient subjects for splitting![/green]")

        self.console.print()

    def integrity_check(self, subject_overlap: int = 0, duplicate_scans: int = 0,
                       missing_metadata: int = 0) -> None:
        """Print integrity check results."""
        if self.json_only:
            return

        self.console.print("[blue][CHECK][/blue] Leakage & integrity")

        overlap_str = "NONE" if subject_overlap == 0 else str(subject_overlap)
        check1 = "[green]✔[/green]" if subject_overlap == 0 else "[red]✖[/red]"
        self.console.print(f"  • Subject overlap across splits: {overlap_str}  {check1}")

        check2 = "[green]✔[/green]" if duplicate_scans == 0 else "[red]✖[/red]"
        self.console.print(f"  • Duplicate scans within a split: {duplicate_scans}    {check2}")

        check3 = "[green]✔[/green]" if missing_metadata == 0 else "[red]✖[/red]"
        self.console.print(f"  • Empty files / missing metadata: {missing_metadata}    {check3}")

        if subject_overlap > 0 or duplicate_scans > 0 or missing_metadata > 0:
            self.has_warnings = True

        self.console.print()

    def file_copy_progress(self, total_files: int) -> Progress:
        """Create and return a progress bar for file copying."""
        if self.json_only or self.quiet:
            return None

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        return progress

    def file_operations_summary(self, copied: int, skipped: int, errors: int,
                               error_details: Optional[List[str]] = None) -> None:
        """Print file operations summary."""
        if self.json_only:
            return

        self.console.print("[blue]📁 FILE OPERATIONS[/blue]")
        self.console.print(f"  • Copied: {copied:,}")
        self.console.print(f"  • Skipped (existing): {skipped:,}")

        if errors > 0:
            self.console.print(f"  • [red]Errors: {errors:,}[/red]")
            self.has_warnings = True

            if error_details and self.verbose:
                self.console.print()
                self.console.print("[yellow]Error details:[/yellow]")
                for i, error in enumerate(error_details[:5], 1):
                    self.console.print(f"  {i}. {error}")
                if len(error_details) > 5:
                    self.console.print(f"  [dim]... and {len(error_details) - 5} more[/dim]")

        total = copied + skipped + errors
        if total > 0:
            success_rate = (copied + skipped) / total * 100
            self.console.print(f"  • Success rate: {success_rate:.1f}%")

        self.console.print()

        # Store in report
        self.report_data["file_operations"] = {
            "copied": copied,
            "skipped": skipped,
            "errors": errors,
            "error_details": error_details if error_details else []
        }

    def files_summary(self, files: Dict[str, int], output_dir: str) -> None:
        """Print files summary."""
        if self.json_only:
            return

        self.console.print("[blue][FILES][/blue]")
        for filename, count in files.items():
            self.console.print(f"  {filename:<15} {count:,} rows")
        self.console.print(f"  [dim](saved in {output_dir})[/dim]")
        self.console.print()

    def report_saved(self, report_path: str) -> None:
        """Print report saved message."""
        if self.json_only or self.quiet:
            return

        self.console.print(f"[blue][REPORT][/blue] Saved detailed JSON: {report_path}")

    def warning(self, message: str) -> None:
        """Print warning message."""
        if self.json_only:
            return

        self.console.print(f"[yellow][WARN][/yellow] {message}")
        self.has_warnings = True
        self.warnings_list.append(message)

    def error(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Print error message."""
        if self.json_only:
            return

        self.console.print(f"[red][ERROR][/red] {message}")

        if details and not self.quiet:
            if "cause" in details:
                self.console.print(f"  Cause: {details['cause']}")
            if "next_steps" in details:
                self.console.print(f"  Next steps: {details['next_steps']}")

    def next_steps(self, steps: List[str]) -> None:
        """Print next steps."""
        if self.json_only or self.quiet:
            return

        self.console.print("[blue]🔄 NEXT STEPS:[/blue]")
        for i, step in enumerate(steps, 1):
            self.console.print(f"   {i}. {step}")
        self.console.print()

    def footer(self, exit_code: int = 0) -> None:
        """Print footer with outcome."""
        duration = time.time() - self.start_time

        # Complete report data
        self.report_data.update({
            "finished_at": datetime.now().isoformat(),
            "duration_sec": round(duration, 1),
            "exit_code": exit_code,
            "has_warnings": self.has_warnings,
            "warnings": self.warnings_list,
            "report_version": "1.1.0"
        })

        # Save JSON report
        report_path = self._save_report()

        if not self.json_only and not self.quiet:
            if report_path:
                self.report_saved(report_path)

            # Determine outcome
            if exit_code != 0:
                outcome = "[red]FAIL[/red]"
            elif self.has_warnings:
                warnings_str = ", ".join(self.warnings_list[:2])
                if len(self.warnings_list) > 2:
                    warnings_str += ", ..."
                outcome = f"[yellow]PASS with warnings ({warnings_str})[/yellow]"
            else:
                outcome = "[green]PASS[/green]"

            self.console.print(f"[dim]Finished in {duration:.1f}s • Outcome: {outcome}[/dim]")

    def _save_report(self) -> Optional[Path]:
        """Save JSON report to file."""
        try:
            # Create reports directory
            reports_dir = Path(".reports")
            reports_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            report_file = reports_dir / f"data_preparation-{timestamp}.json"

            # Write JSON
            with open(report_file, "w") as f:
                json.dump(self.report_data, f, indent=2)

            return report_file
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow][WARN][/yellow] Could not save report: {e}")
            return None