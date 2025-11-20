"""Output formatter for image processing stage."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn


class ImageProcessingFormatter:
    """Handles formatted output for image processing stage."""

    def __init__(self, verbose=False, quiet=False, json_only=False):
        self.verbose = verbose
        self.quiet = quiet
        self.json_only = json_only
        self.console = Console()
        self.report_data = {}
        self.has_warnings = False
        self.warnings_list = []
        self.start_time = time.time()

    def header(self, action: str, substage: str, **kwargs) -> None:
        """Print header section."""
        if self.json_only or self.quiet:
            return

        self.console.print("[cyan]═" * 70 + "[/cyan]")
        self.console.print(f"[cyan] Image Processing • {substage.replace('_', ' ').title()} • {action.title()}[/cyan]")

        # Build context line
        timestamp = datetime.now().strftime("%H:%M:%S")
        info_parts = [f"Started: {timestamp}"]

        # Add action-specific context
        for key, value in kwargs.items():
            info_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        info_line = "  •  ".join(info_parts)
        self.console.print(f" [dim]{info_line}[/dim]")
        self.console.print("[cyan]═" * 70 + "[/cyan]")
        self.console.print()

        # Initialize report
        self.report_data = {
            "tool": "image_processing",
            "substage": substage,
            "action": action,
            "started_at": datetime.now().isoformat(),
            "parameters": kwargs
        }

    def info(self, message: str) -> None:
        """Print info message."""
        if not self.json_only:
            self.console.print(f"[blue][INFO][/blue] {message}")

    def success(self, message: str) -> None:
        """Print success message."""
        if not self.json_only:
            self.console.print(f"[green]✅[/green] {message}")

    def warning(self, message: str) -> None:
        """Print warning and track it."""
        if not self.json_only:
            self.console.print(f"[yellow]⚠️[/yellow] {message}")
        self.has_warnings = True
        self.warnings_list.append(message)

    def error(self, message: str, details: Optional[Dict] = None) -> None:
        """Print error with optional details."""
        if not self.json_only:
            self.console.print(f"[red]❌[/red] {message}")

            if details and not self.quiet:
                if "cause" in details:
                    self.console.print(f"  Cause: {details['cause']}")
                if "next_steps" in details:
                    self.console.print(f"  Next steps: {details['next_steps']}")

    def print(self, message: str) -> None:
        """Print a message."""
        if not self.json_only:
            self.console.print(message)

    def input_distribution(self, distribution: Dict[str, Dict[str, Dict[str, int]]]) -> None:
        """Print input file distribution."""
        if self.json_only:
            return

        self.console.print("[blue]📊 INPUT DISTRIBUTION[/blue]")
        total_files = 0

        for slice_type in ["axial", "coronal", "sagittal"]:
            if slice_type not in distribution:
                continue

            self.console.print(f"\n  {slice_type.upper()} SLICES:")
            slice_total = 0

            for split in ["train", "val", "test"]:
                if split not in distribution[slice_type]:
                    continue

                split_total = 0
                for group in sorted(distribution[slice_type][split].keys()):
                    count = distribution[slice_type][split][group]
                    self.console.print(f"    {split}/{group}: {count:,} files")
                    split_total += count

                if split_total > 0:
                    self.console.print(f"    {split} total: {split_total:,} files")
                slice_total += split_total

            if slice_total > 0:
                self.console.print(f"  {slice_type} total: {slice_total:,} files")
            total_files += slice_total

        self.console.print(f"\n  [bold]GRAND TOTAL: {total_files:,} files[/bold]")
        self.console.print()

        # Store in report
        self.report_data["input_distribution"] = distribution
        self.report_data["input_total"] = total_files

    def output_distribution(self, distribution: Dict[str, Dict[str, Dict[str, int]]]) -> None:
        """Print output file distribution."""
        if self.json_only:
            return

        self.console.print("[blue]📊 OUTPUT DISTRIBUTION[/blue]")
        total_files = 0

        for slice_type in ["axial", "coronal", "sagittal"]:
            if slice_type not in distribution:
                continue

            self.console.print(f"\n  {slice_type.upper()} SLICES:")
            slice_total = 0

            for split in ["train", "val", "test"]:
                if split not in distribution[slice_type]:
                    continue

                split_data = distribution[slice_type][split]
                split_total = sum(split_data.values())
                
                if split_total > 0:
                    # Build compact group list: "AD: 279, MCI: 651, CN: 408"
                    # Use custom order: AD, MCI, CN (matching user preference)
                    group_order = ["AD", "MCI", "CN"]
                    group_parts = []
                    for group in group_order:
                        if group in split_data:
                            count = split_data[group]
                            group_parts.append(f"{group}: {count:,}")
                    # Add any remaining groups not in the standard order
                    for group in sorted(split_data.keys()):
                        if group not in group_order:
                            count = split_data[group]
                            group_parts.append(f"{group}: {count:,}")
                    group_str = ", ".join(group_parts)
                    
                    self.console.print(f"    {split} total: {split_total:,} files | {group_str} |")
                
                slice_total += split_total

            if slice_total > 0:
                self.console.print(f"  {slice_type} total: {slice_total:,} files")
            total_files += slice_total

        self.console.print(f"\n  [bold]OUTPUT GRAND TOTAL: {total_files:,} files[/bold]")
        self.console.print()

        # Store in report
        self.report_data["output_distribution"] = distribution
        self.report_data["output_total"] = total_files

    def configuration(self, crop_padding: int, target_size: tuple, rotation_angle: int) -> None:
        """Print configuration summary."""
        if self.json_only or self.quiet:
            return

        self.console.print("[blue]⚙️  CONFIGURATION[/blue]")
        self.console.print(f"  • Crop padding: {crop_padding} pixels")
        self.console.print(f"  • Target size: {target_size[0]}×{target_size[1]} pixels")
        self.console.print(f"  • Rotation angle: {rotation_angle}°")
        self.console.print()

    def processing_summary(self, processed: int, skipped: int, errors: int, total: int) -> None:
        """Print processing summary."""
        if self.json_only:
            return

        self.console.print("[blue]📊 PROCESSING SUMMARY[/blue]")
        self.console.print(f"  • Processed: {processed:,}")
        self.console.print(f"  • Skipped (existing): {skipped:,}")

        if errors > 0:
            self.console.print(f"  • [red]Errors: {errors:,}[/red]")
            self.has_warnings = True

        if total > 0:
            success_rate = (processed + skipped) / total * 100
            self.console.print(f"  • Success rate: {success_rate:.1f}%")

        self.console.print()

        # Store in report
        self.report_data["results"] = {
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "total": total,
            "success_rate": success_rate if total > 0 else 0
        }

    def create_progress_bar(self) -> Progress:
        """Create and return a progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )

    def next_steps(self, steps: List[str]) -> None:
        """Print next steps."""
        if self.json_only or self.quiet:
            return

        self.console.print("\n[blue]🔄 NEXT STEPS:[/blue]")
        for i, step in enumerate(steps, 1):
            self.console.print(f"  {i}. {step}")
        self.console.print()

    def footer(self, exit_code: int = 0) -> None:
        """Print footer and save report."""
        duration = time.time() - self.start_time

        # Complete report
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
                self.console.print(f"[blue][REPORT][/blue] Saved detailed JSON: {report_path}")

            # Determine outcome
            if exit_code != 0:
                outcome = "[red]FAIL[/red]"
            elif self.has_warnings:
                outcome = "[yellow]PASS with warnings[/yellow]"
            else:
                outcome = "[green]PASS[/green]"

            self.console.print(f"[dim]Finished in {duration:.1f}s • Outcome: {outcome}[/dim]")

    def _save_report(self) -> Optional[Path]:
        """Save JSON report to file."""
        try:
            reports_dir = Path(".reports")
            reports_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            substage = self.report_data.get("substage", "image_processing")
            report_file = reports_dir / f"image_processing_{substage}-{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(self.report_data, f, indent=2)

            return report_file
        except Exception as e:
            if self.verbose:
                self.warning(f"Could not save report: {e}")
            return None

